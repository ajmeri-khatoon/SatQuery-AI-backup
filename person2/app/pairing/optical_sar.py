"""Sentinel-2 optical and Sentinel-1 SAR pairing workflow."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as transform_geometry

from ..acquisition.sentinel1 import Sentinel1Acquirer, Sentinel1Scene
from ..acquisition.sentinel2 import Sentinel2Acquirer, Sentinel2Scene
from ..models.metadata import ImageMetadata, ProcessingConfig
from ..metadata.extractor import extract_image_metadata
from ..models.optical_sar import OpticalSARPairReport, OpticalSARRequest
from ..models.request import SatelliteRequest, SatelliteSensor
from ..models.sentinel1 import Sentinel1Request
from ..preprocessing.alignment import AlignmentError, align_images
from ..preprocessing.sentinel1 import Sentinel1PreprocessingError, prepare_sentinel1
from ..preprocessing.sentinel2 import Sentinel2PreprocessingError, prepare_sentinel2

LOGGER = logging.getLogger(__name__)


class OpticalSARPairingError(RuntimeError):
    """Raised when optical and SAR imagery cannot form a compatible pair."""


class OpticalSARPairResult:
    """Aligned paths and report returned by the optical/SAR workflow."""

    def __init__(self, report: OpticalSARPairReport, report_path: Path) -> None:
        self.report = report
        self.report_path = report_path


def _coverage(path: Path, request: OpticalSARRequest) -> bool:
    try:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise OpticalSARPairingError(f"Raster '{path}' has no CRS")
            aoi_geometry = request.aoi.shapely_geometry
            converter = Transformer.from_crs(request.aoi.crs, dataset.crs, always_xy=True)
            aoi_geometry = transform_geometry(converter.transform, aoi_geometry)
            raster_geometry = box(*dataset.bounds)
            return raster_geometry.covers(aoi_geometry)
    except OpticalSARPairingError:
        raise
    except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
        raise OpticalSARPairingError(f"Unable to validate AOI coverage for '{path}': {exc}") from exc


def _metadata_for_aligned(base: ImageMetadata, path: Path) -> ImageMetadata:
    return extract_image_metadata(
        path,
        sensor=base.sensor,
        satellite=base.satellite,
        acquisition_date=base.acquisition_date,
        source=base.source,
        aoi=base.aoi,
        polarization=base.polarization,
        parent_image=str(path),
        preprocessing_steps=[*base.preprocessing_steps, "aligned_metadata_extracted"],
        extra=base.extra,
    )


def _scene_dates(
    optical_scene: Sentinel2Scene | None,
    sar_scene: Sentinel1Scene | None,
    request: OpticalSARRequest,
) -> tuple[date, date, str, str]:
    optical_date = optical_scene.acquisition_date if optical_scene else request.optical_start_date
    sar_date = sar_scene.acquisition_date if sar_scene else request.sar_start_date
    optical_id = optical_scene.image_id if optical_scene else "local:sentinel2"
    sar_id = sar_scene.image_id if sar_scene else "local:sentinel1"
    return optical_date, sar_date, optical_id, sar_id


def process_optical_sar(
    request: OpticalSARRequest,
    output_dir: str | Path,
    *,
    optical_source: str | Path | None = None,
    sar_source: str | Path | None = None,
    optical_scene: Sentinel2Scene | None = None,
    sar_scene: Sentinel1Scene | None = None,
    optical_acquirer: Any | None = None,
    sar_acquirer: Any | None = None,
    resampling: str = "nearest",
    nodata: float | int | None = None,
    overwrite: bool = False,
) -> OpticalSARPairResult:
    """Acquire or process optical/SAR images and prepare one common grid.

    Acquisition is performed independently for each sensor when source paths
    are omitted. Scene timestamps are compared after acquisition; they need not
    be identical, but their absolute difference must fit the request limit.
    """

    if (optical_source is None) != (sar_source is None):
        raise OpticalSARPairingError("Provide both optical_source and sar_source, or neither")
    root = Path(output_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    aligned_dir = root / "aligned"
    optical_date, sar_date, optical_id, sar_id = _scene_dates(optical_scene, sar_scene, request)

    if optical_source is None:
        optical_adapter = optical_acquirer or Sentinel2Acquirer()
        sar_adapter = sar_acquirer or Sentinel1Acquirer()
        optical_request = SatelliteRequest(
            aoi=request.aoi,
            sensor=SatelliteSensor.SENTINEL_2,
            start_date=request.optical_start_date,
            end_date=request.optical_end_date,
            bands=request.optical_bands,
            resolution_m=request.resolution_m,
            max_cloud_cover=request.max_cloud_cover,
            source=request.source,
        )
        sar_request = Sentinel1Request(
            aoi=request.aoi,
            start_date=request.sar_start_date,
            end_date=request.sar_end_date,
            polarizations=request.sar_polarizations,
            resolution_m=request.resolution_m,
            source=request.source,
        )
        optical_scene = optical_adapter.search(optical_request)
        sar_scene = sar_adapter.search(sar_request)
        optical_source = raw_dir / "sentinel2.tif"
        sar_source = raw_dir / "sentinel1.tif"
        optical_adapter.download(optical_request, optical_source, scene=optical_scene)
        sar_adapter.download(sar_request, sar_source, scene=sar_scene)
        optical_date, sar_date, optical_id, sar_id = _scene_dates(optical_scene, sar_scene, request)

    date_difference = abs((optical_date - sar_date).days)
    if date_difference > request.max_date_difference_days:
        raise OpticalSARPairingError(
            f"Sentinel-1 and Sentinel-2 acquisition dates differ by {date_difference} days; "
            f"maximum allowed is {request.max_date_difference_days} days"
        )
    optical_path, sar_path = Path(optical_source), Path(sar_source)
    coverage = {
        "sentinel-2": _coverage(optical_path, request),
        "sentinel-1": _coverage(sar_path, request),
    }
    if not all(coverage.values()):
        raise OpticalSARPairingError(f"AOI is not fully covered by both images: {coverage}")

    target_crs = request.target_crs or request.aoi.crs
    processing = ProcessingConfig(
        target_crs=target_crs,
        resolution_m=request.resolution_m,
        resampling=resampling,
        nodata=nodata,
    )
    try:
        optical_metadata = prepare_sentinel2(
            optical_path, processed_dir / "sentinel2.tif", aoi=request.aoi,
            acquisition_date=optical_date, processing=processing,
            bands=request.optical_bands, source=request.source,
        )
        sar_metadata = prepare_sentinel1(
            sar_path, processed_dir / "sentinel1.tif", aoi=request.aoi,
            acquisition_date=sar_date, processing=processing,
            polarizations=request.sar_polarizations, source=request.source,
        )
        alignment = align_images(
            processed_dir / "sentinel2.tif", processed_dir / "sentinel1.tif",
            aligned_dir / "sentinel2.tif", aligned_dir / "sentinel1.tif",
            target_crs=target_crs, resolution=request.resolution_m,
            resampling=resampling, nodata=nodata, overwrite=overwrite,
        )
    except (AlignmentError, Sentinel1PreprocessingError, Sentinel2PreprocessingError, OSError, ValueError) as exc:
        raise OpticalSARPairingError(f"Optical/SAR common-grid preparation failed: {exc}") from exc

    optical_aligned = _metadata_for_aligned(optical_metadata, aligned_dir / "sentinel2.tif")
    sar_aligned = _metadata_for_aligned(sar_metadata, aligned_dir / "sentinel1.tif")
    pair_id = f"sentinel2_sentinel1_{optical_date}_{sar_date}"
    report = OpticalSARPairReport(
        pair_id=pair_id,
        optical_scene_id=optical_id,
        sar_scene_id=sar_id,
        optical_acquisition_date=optical_date,
        sar_acquisition_date=sar_date,
        date_difference_days=date_difference,
        max_date_difference_days=request.max_date_difference_days,
        aoi=request.aoi.model_dump(mode="json"),
        aoi_coverage=coverage,
        optical_bands=request.optical_bands,
        sar_polarizations=[polarization.value for polarization in request.sar_polarizations],
        optical_source_path=str(optical_path),
        sar_source_path=str(sar_path),
        optical_processed_path=str(processed_dir / "sentinel2.tif"),
        sar_processed_path=str(processed_dir / "sentinel1.tif"),
        optical_aligned_path=str(aligned_dir / "sentinel2.tif"),
        sar_aligned_path=str(aligned_dir / "sentinel1.tif"),
        common_crs=alignment.target_crs,
        common_resolution=alignment.target_resolution,
        common_bounds=alignment.common_bounds,
        common_dimensions=alignment.target_dimensions,
        common_transform=alignment.target_transform,
        preprocessing_steps=[
            "sentinel-2 optical preprocessing", "sentinel-1 SAR preprocessing",
            "AOI coverage validated", "acquisition dates compared",
            "reprojected to common CRS", "resampled to common resolution",
            "cropped to common valid bounds", "aligned outputs written",
        ],
    )
    report_path = root / "optical_sar_pair_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    LOGGER.info("Prepared optical/SAR pair %s", pair_id)
    return OpticalSARPairResult(report, report_path)