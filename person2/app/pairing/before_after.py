"""Before/after acquisition, preprocessing, alignment, and tiling workflow."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import rasterio

from ..acquisition.sentinel1 import Sentinel1Acquirer
from ..acquisition.sentinel2 import Sentinel2Acquirer
from ..models.before_after import BeforeAfterRequest, PairMetadata
from ..models.metadata import ImageMetadata, ProcessingConfig
from ..metadata.extractor import extract_image_metadata
from ..models.request import SatelliteRequest
from ..models.sentinel1 import Polarization, Sentinel1Request
from ..preprocessing.alignment import AlignmentError, align_images
from ..preprocessing.sentinel1 import Sentinel1PreprocessingError, prepare_sentinel1
from ..preprocessing.sentinel2 import Sentinel2PreprocessingError, prepare_sentinel2
from ..tiling.tiler import TileMetadata, create_tiles

LOGGER = logging.getLogger(__name__)


class BeforeAfterError(RuntimeError):
    """Raised when a before/after pair cannot be prepared safely."""


class BeforeAfterResult:
    """Paths and metadata returned by the before/after workflow."""

    def __init__(
        self,
        pair_metadata: PairMetadata,
        before_tiles: list[TileMetadata],
        after_tiles: list[TileMetadata],
        pair_metadata_path: Path,
    ) -> None:
        self.pair_metadata = pair_metadata
        self.before_tiles = before_tiles
        self.after_tiles = after_tiles
        self.pair_metadata_path = pair_metadata_path


def _sensor_request(request: BeforeAfterRequest, start: date, end: date | None):
    if request.sensor.value == "sentinel-2":
        return SatelliteRequest(
            aoi=request.aoi,
            sensor=request.sensor,
            start_date=start,
            end_date=end,
            bands=request.bands,
            resolution_m=request.resolution_m,
            max_cloud_cover=request.max_cloud_cover,
            source=request.source,
        )
    return Sentinel1Request(
        aoi=request.aoi,
        start_date=start,
        end_date=end,
        polarizations=[Polarization(band.upper()) for band in (request.bands or ["VV", "VH"])],
        resolution_m=request.resolution_m,
        source=request.source,
    )


def _aligned_metadata(base: ImageMetadata, path: Path) -> ImageMetadata:
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


def _prepare_local(
    request: BeforeAfterRequest,
    source_path: Path,
    output_path: Path,
    acquisition_date: date,
    processing: ProcessingConfig,
) -> ImageMetadata:
    if request.sensor.value == "sentinel-2":
        return prepare_sentinel2(
            source_path,
            output_path,
            aoi=request.aoi,
            acquisition_date=acquisition_date,
            processing=processing,
            bands=request.bands or None,
            source=request.source,
        )
    return prepare_sentinel1(
        source_path,
        output_path,
        aoi=request.aoi,
        acquisition_date=acquisition_date,
        processing=processing,
        polarizations=[Polarization(band.upper()) for band in (request.bands or ["VV", "VH"])],
        source=request.source,
    )


def process_before_after(
    request: BeforeAfterRequest,
    output_dir: str | Path,
    *,
    before_source: str | Path | None = None,
    after_source: str | Path | None = None,
    target_crs: str | None = None,
    resampling: str = "nearest",
    nodata: float | int | None = None,
    tile_width: int = 256,
    tile_height: int = 256,
    overlap: int = 0,
    padding: bool = False,
    tile: bool = True,
    overwrite: bool = False,
    acquirer: Any | None = None,
) -> BeforeAfterResult:
    """Produce aligned, matching before/after tiles and pair metadata.

    When source paths are supplied, acquisition is skipped for offline or test
    workflows. Otherwise the appropriate Earth Engine Sentinel-1 or Sentinel-2
    adapter acquires both date ranges before local preprocessing.
    """

    if (before_source is None) != (after_source is None):
        raise BeforeAfterError("Provide both before_source and after_source, or neither")
    root = Path(output_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    aligned_dir = root / "aligned"
    pair_id = f"{request.sensor.value}_{request.before_start_date}_{request.after_start_date}"
    before_date = request.before_start_date
    after_date = request.after_start_date

    if before_source is None:
        adapter = acquirer
        if adapter is None:
            adapter = Sentinel2Acquirer() if request.sensor.value == "sentinel-2" else Sentinel1Acquirer()
        before_request = _sensor_request(request, request.before_start_date, request.before_end_date)
        after_request = _sensor_request(request, request.after_start_date, request.after_end_date)
        if request.sensor.value == "sentinel-2":
            before_scene = adapter.download(before_request, raw_dir / "before.tif")
            after_scene = adapter.download(after_request, raw_dir / "after.tif")
        else:
            before_scene = adapter.download(before_request, raw_dir / "before.tif")
            after_scene = adapter.download(after_request, raw_dir / "after.tif")
        before_source = raw_dir / "before.tif"
        after_source = raw_dir / "after.tif"
        before_date = before_scene.acquisition_date
        after_date = after_scene.acquisition_date

    processing = ProcessingConfig(
        target_crs=target_crs or "EPSG:4326",
        resolution_m=request.resolution_m,
        resampling=resampling,
        nodata=nodata,
    )
    try:
        before_metadata = _prepare_local(
            request, Path(before_source), processed_dir / "before.tif", before_date, processing
        )
        after_metadata = _prepare_local(
            request, Path(after_source), processed_dir / "after.tif", after_date, processing
        )
        alignment = align_images(
            processed_dir / "before.tif",
            processed_dir / "after.tif",
            aligned_dir / "before.tif",
            aligned_dir / "after.tif",
            target_crs=target_crs or processing.target_crs,
            resolution=request.resolution_m,
            resampling=resampling,
            nodata=nodata,
            overwrite=overwrite,
        )
    except (
        AlignmentError,
        Sentinel1PreprocessingError,
        Sentinel2PreprocessingError,
        OSError,
        ValueError,
    ) as exc:
        raise BeforeAfterError(
            f"Before/after images cannot be aligned or preprocessed: {exc}"
        ) from exc

    aligned_before = _aligned_metadata(before_metadata, aligned_dir / "before.tif")
    aligned_after = _aligned_metadata(after_metadata, aligned_dir / "after.tif")
    if tile:
        before_tiles = create_tiles(
            aligned_dir / "before.tif", root / "before", tile_width=tile_width,
            tile_height=tile_height, overlap=overlap, padding=padding, overwrite=overwrite,
            aoi=request.aoi.model_dump(mode="json"),
        )
        after_tiles = create_tiles(
            aligned_dir / "after.tif", root / "after", tile_width=tile_width,
            tile_height=tile_height, overlap=overlap, padding=padding, overwrite=overwrite,
            aoi=request.aoi.model_dump(mode="json"),
        )
    else:
        before_tiles = []
        after_tiles = []
    if len(before_tiles) != len(after_tiles) or [tile.tile_id for tile in before_tiles] != [tile.tile_id for tile in after_tiles]:
        raise BeforeAfterError("Before and after tiling produced different tile grids")
    pair_metadata = PairMetadata(
        pair_id=pair_id,
        sensor=request.sensor.value,
        before=aligned_before,
        after=aligned_after,
        before_aligned_path=str(aligned_dir / "before.tif"),
        after_aligned_path=str(aligned_dir / "after.tif"),
        common_crs=alignment.target_crs,
        common_resolution=alignment.target_resolution,
        common_bounds=alignment.common_bounds,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        before_tiles=[tile.model_dump(mode="json") for tile in before_tiles],
        after_tiles=[tile.model_dump(mode="json") for tile in after_tiles],
        preprocessing_steps=[
            "before_acquired", "after_acquired", "same_preprocessing_configuration",
            "aligned_to_common_grid", "cropped_to_common_valid_area", "matching_tiles_generated",
        ],
    )
    metadata_path = root / "pair_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(pair_metadata.model_dump_json(indent=2), encoding="utf-8")
    LOGGER.info("Prepared before/after pair %s with %d matching tiles", pair_id, len(before_tiles))
    return BeforeAfterResult(pair_metadata, before_tiles, after_tiles, metadata_path)