"""High-level facade over the independent Person 2 pipeline modules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..acquisition.sentinel1 import Sentinel1Acquirer
from ..acquisition.sentinel2 import Sentinel2Acquirer
from ..config import get_config
from ..models.before_after import BeforeAfterRequest
from ..models.metadata import ProcessingConfig
from ..models.optical_sar import OpticalSARRequest
from ..models.pipeline import PipelineResult, PipelineWorkflow, SatellitePipelineRequest
from ..models.request import SatelliteRequest, SatelliteSensor
from ..models.sentinel1 import Polarization, Sentinel1Request
from ..pairing.before_after import process_before_after
from ..pairing.optical_sar import process_optical_sar
from ..preprocessing.sentinel1 import prepare_sentinel1
from ..preprocessing.sentinel2 import prepare_sentinel2
from ..tiling import create_tiles

LOGGER = logging.getLogger(__name__)


def _validate_local_source(source: str | Path) -> None:
    """Reuse the preserved Person 2 raster inspection boundary for local inputs."""
    try:
        from person2.preprocessing import validate_raster
    except ModuleNotFoundError:
        from preprocessing import validate_raster

    validate_raster(source)


def _write_json(path: Path, model: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


def _base_result(request: SatellitePipelineRequest) -> PipelineResult:
    return PipelineResult(status="success", workflow=request.workflow.value)


def _single(request: SatellitePipelineRequest, root: Path) -> PipelineResult:
    result = _base_result(request)
    is_sentinel2 = request.workflow == PipelineWorkflow.SINGLE_SENTINEL_2
    sensor = SatelliteSensor.SENTINEL_2 if is_sentinel2 else SatelliteSensor.SENTINEL_1
    raw_path = root / "raw" / ("sentinel2.tif" if is_sentinel2 else "sentinel1.tif")
    processed_path = root / "processed" / raw_path.name
    acquisition_date = request.start_date
    bands = request.bands if is_sentinel2 else []
    polarizations = request.polarizations or [Polarization.VV, Polarization.VH]

    if request.source_path is not None:
        source_path = request.source_path
        _validate_local_source(source_path)
    elif is_sentinel2:
        acquisition_request = SatelliteRequest(
            aoi=request.aoi, sensor=sensor, start_date=request.start_date,
            end_date=request.end_date, bands=request.bands,
            resolution_m=request.resolution_m, max_cloud_cover=request.max_cloud_cover,
            source=request.source,
        )
        scene = Sentinel2Acquirer().download(acquisition_request, raw_path)
        source_path, acquisition_date = raw_path, scene.acquisition_date
    else:
        acquisition_request = Sentinel1Request(
            aoi=request.aoi, start_date=request.start_date, end_date=request.end_date,
            polarizations=polarizations, resolution_m=request.resolution_m, source=request.source,
        )
        scene = Sentinel1Acquirer().download(acquisition_request, raw_path)
        source_path, acquisition_date = raw_path, scene.acquisition_date

    processing = ProcessingConfig(
        target_crs=request.target_crs or request.aoi.crs,
        resolution_m=request.resolution_m,
        resampling=request.resampling,
        nodata=request.nodata,
    )
    if is_sentinel2:
        metadata = prepare_sentinel2(
            source_path, processed_path, aoi=request.aoi, acquisition_date=acquisition_date,
            processing=processing, bands=bands or None, source=request.source,
        )
    else:
        metadata = prepare_sentinel1(
            source_path, processed_path, aoi=request.aoi, acquisition_date=acquisition_date,
            processing=processing, polarizations=polarizations, source=request.source,
        )
    metadata_path = _write_json(root / "metadata" / "image_metadata.json", metadata)
    result.output_paths["processed"] = str(processed_path)
    result.metadata_paths.append(metadata_path)
    result.processing_information.update({
        "sensor": sensor.value,
        "acquisition_date": acquisition_date.isoformat(),
        "preprocessing": metadata.preprocessing_steps,
    })
    if request.tile:
        tile_dir = root / "tiles"
        tiles = create_tiles(
            processed_path, tile_dir, tile_width=request.tile_width,
            tile_height=request.tile_height, overlap=request.overlap,
            padding=request.padding, metadata_path=tile_dir / "tiles.json",
            aoi=request.aoi.model_dump(mode="json"),
            polarization=polarizations if not is_sentinel2 else (),
        )
        result.tile_paths["image"] = [tile.output_path for tile in tiles]
        result.metadata_paths.append(str(tile_dir / "tiles.json"))
        result.processing_information["tile_count"] = len(tiles)
    return result


def _before_after(request: SatellitePipelineRequest, root: Path) -> PipelineResult:
    is_sentinel2 = request.workflow == PipelineWorkflow.BEFORE_AFTER_SENTINEL_2
    sensor = SatelliteSensor.SENTINEL_2 if is_sentinel2 else SatelliteSensor.SENTINEL_1
    lower_request = BeforeAfterRequest(
        aoi=request.aoi, sensor=sensor,
        before_start_date=request.before_start_date,
        before_end_date=request.before_end_date,
        after_start_date=request.after_start_date,
        after_end_date=request.after_end_date,
        bands=request.bands if is_sentinel2 else [item.value for item in request.polarizations],
        resolution_m=request.resolution_m,
        max_cloud_cover=request.max_cloud_cover,
        source=request.source,
    )
    if request.before_source is not None and request.after_source is not None:
        _validate_local_source(request.before_source)
        _validate_local_source(request.after_source)
    workflow_result = process_before_after(
        lower_request, root,
        before_source=request.before_source,
        after_source=request.after_source,
        target_crs=request.target_crs,
        tile_width=request.tile_width,
        tile_height=request.tile_height,
        overlap=request.overlap,
        padding=request.padding,
        tile=request.tile,
        resampling=request.resampling,
        nodata=request.nodata,
        overwrite=False,
    )
    result = _base_result(request)
    result.output_paths.update({
        "before_aligned": workflow_result.pair_metadata.before_aligned_path,
        "after_aligned": workflow_result.pair_metadata.after_aligned_path,
    })
    result.metadata_paths.append(str(workflow_result.pair_metadata_path))
    result.tile_paths["before"] = [tile.output_path for tile in workflow_result.before_tiles]
    result.tile_paths["after"] = [tile.output_path for tile in workflow_result.after_tiles]
    result.processing_information.update({
        "common_crs": workflow_result.pair_metadata.common_crs,
        "common_resolution": workflow_result.pair_metadata.common_resolution,
        "tile_count": len(workflow_result.before_tiles),
    })
    return result


def _optical_sar(request: SatellitePipelineRequest, root: Path) -> PipelineResult:
    lower_request = OpticalSARRequest(
        aoi=request.aoi,
        optical_start_date=request.optical_start_date,
        optical_end_date=request.optical_end_date,
        sar_start_date=request.sar_start_date,
        sar_end_date=request.sar_end_date,
        max_date_difference_days=request.max_date_difference_days,
        optical_bands=request.optical_bands or ["B04", "B03", "B02", "B08"],
        sar_polarizations=request.polarizations or [Polarization.VV, Polarization.VH],
        resolution_m=request.resolution_m,
        target_crs=request.target_crs,
        max_cloud_cover=request.max_cloud_cover,
        source=request.source,
    )
    if request.optical_source is not None and request.sar_source is not None:
        _validate_local_source(request.optical_source)
        _validate_local_source(request.sar_source)
    workflow_result = process_optical_sar(
        lower_request, root,
        optical_source=request.optical_source,
        sar_source=request.sar_source,
        resampling=request.resampling,
        nodata=request.nodata,
    )
    result = _base_result(request)
    report = workflow_result.report
    result.output_paths.update({
        "optical_aligned": report.optical_aligned_path,
        "sar_aligned": report.sar_aligned_path,
    })
    result.metadata_paths.append(str(workflow_result.report_path))
    result.processing_information.update({
        "common_crs": report.common_crs,
        "common_resolution": report.common_resolution,
        "date_difference_days": report.date_difference_days,
    })
    if request.tile:
        for key, source_path in (
            ("optical", report.optical_aligned_path),
            ("sar", report.sar_aligned_path),
        ):
            tile_dir = root / f"{key}_tiles"
            tiles = create_tiles(
                source_path, tile_dir, tile_width=request.tile_width,
                tile_height=request.tile_height, overlap=request.overlap,
                padding=request.padding, metadata_path=tile_dir / "tiles.json",
                aoi=request.aoi.model_dump(mode="json"),
                polarization=report.sar_polarizations if key == "sar" else (),
            )
            result.tile_paths[key] = [tile.output_path for tile in tiles]
            result.metadata_paths.append(str(tile_dir / "tiles.json"))
        result.processing_information["tile_count"] = len(result.tile_paths["optical"])
    return result


def process_satellite_request(
    request: SatellitePipelineRequest | dict[str, Any],
) -> PipelineResult:
    """Validate and execute one complete satellite processing workflow.

    Lower-level modules remain independently usable. This facade returns a
    structured failed result with the exception type and message when a
    workflow fails; it never converts an error into a false success.
    """

    try:
        validated = (
            request
            if isinstance(request, SatellitePipelineRequest)
            else SatellitePipelineRequest.model_validate(request)
        )
    except (ValidationError, TypeError, ValueError) as exc:
        workflow = request.get("workflow", "unknown") if isinstance(request, dict) else "unknown"
        return PipelineResult(
            status="failed",
            workflow=str(workflow),
            errors=[f"ValidationError: {exc}"],
        )

    root = (
        validated.output_dir
        or get_config().resolved_data_root / "processed" / validated.workflow.value
    )
    try:
        if validated.workflow in {
            PipelineWorkflow.SINGLE_SENTINEL_1,
            PipelineWorkflow.SINGLE_SENTINEL_2,
        }:
            return _single(validated, root)
        if validated.workflow in {
            PipelineWorkflow.BEFORE_AFTER_SENTINEL_1,
            PipelineWorkflow.BEFORE_AFTER_SENTINEL_2,
        }:
            return _before_after(validated, root)
        if validated.workflow == PipelineWorkflow.OPTICAL_SAR:
            return _optical_sar(validated, root)
        return PipelineResult(
            status="failed",
            workflow=validated.workflow.value,
            errors=[
                "before-after optical+SAR is not supported by the existing modules; "
                "run separate before/after optical+SAR requests when those workflows "
                "are implemented"
            ],
        )
    except Exception as exc:
        LOGGER.exception("Satellite pipeline failed for %s", validated.workflow.value)
        return PipelineResult(
            status="failed",
            workflow=validated.workflow.value,
            errors=[f"{type(exc).__name__}: {exc}"],
            processing_information={"output_dir": str(root)},
        )