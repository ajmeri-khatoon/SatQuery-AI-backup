from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from app.models import PipelineWorkflow, SatellitePipelineRequest
from app.pipeline import process_satellite_request


def _write_raster(path: Path, bands: list[str], value: float, dtype: str = "uint16") -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=10, count=len(bands),
        dtype=dtype, crs="EPSG:32643",
        transform=rasterio.Affine.translation(363000, 2068000) * rasterio.Affine.scale(10, -10),
        nodata=0,
    ) as dataset:
        dataset.write(np.full((len(bands), 10, 10), value, dtype=dtype))
        dataset.descriptions = tuple(bands)


def _base_request(workflow: str, output_dir: Path, **values):
    payload = {
        "workflow": workflow,
        "aoi": {"bbox": (363000, 2067900, 363100, 2068000), "crs": "EPSG:32643"},
        "resolution_m": 10,
        "target_crs": "EPSG:32643",
        "output_dir": output_dir,
    }
    payload.update(values)
    return payload


def test_pipeline_processes_single_sentinel2_and_tiles(tmp_path: Path):
    source = tmp_path / "s2.tif"
    _write_raster(source, ["B04"], 100)
    result = process_satellite_request(_base_request(
        PipelineWorkflow.SINGLE_SENTINEL_2.value, tmp_path / "single-s2",
        start_date=date(2026, 1, 1), bands=["B04"], source_path=source,
        tile=True, tile_width=4, tile_height=4,
    ))

    assert result.status == "success"
    assert result.output_paths["processed"]
    assert result.metadata_paths
    assert result.tile_paths["image"]
    assert not result.errors


def test_pipeline_processes_single_sentinel1(tmp_path: Path):
    source = tmp_path / "s1.tif"
    _write_raster(source, ["VV", "VH"], 3.0, dtype="float32")
    result = process_satellite_request(_base_request(
        PipelineWorkflow.SINGLE_SENTINEL_1.value, tmp_path / "single-s1",
        start_date=date(2026, 1, 1), polarizations=["VV", "VH"], source_path=source,
    ))

    assert result.status == "success"
    assert result.processing_information["sensor"] == "sentinel-1"


def test_pipeline_processes_before_after_sentinel2(tmp_path: Path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    _write_raster(before, ["B04"], 10)
    _write_raster(after, ["B04"], 20)
    result = process_satellite_request(_base_request(
        PipelineWorkflow.BEFORE_AFTER_SENTINEL_2.value, tmp_path / "before-after",
        before_start_date=date(2026, 1, 1), before_end_date=date(2026, 1, 2),
        after_start_date=date(2026, 2, 1), after_end_date=date(2026, 2, 2),
        bands=["B04"], before_source=before, after_source=after,
        tile=False,
    ))

    assert result.status == "success"
    assert "before_aligned" in result.output_paths
    assert result.tile_paths == {"before": [], "after": []}


def test_pipeline_processes_optical_sar_pair(tmp_path: Path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"
    _write_raster(optical, ["B04"], 10)
    _write_raster(sar, ["VV", "VH"], 2.0, dtype="float32")
    result = process_satellite_request(_base_request(
        PipelineWorkflow.OPTICAL_SAR.value, tmp_path / "optical-sar",
        optical_start_date=date(2026, 1, 1), sar_start_date=date(2026, 1, 3),
        optical_bands=["B04"], polarizations=["VV", "VH"],
        optical_source=optical, sar_source=sar, tile=True,
        tile_width=4, tile_height=4,
    ))

    assert result.status == "success"
    assert "optical_aligned" in result.output_paths
    assert len(result.tile_paths["optical"]) == len(result.tile_paths["sar"])


def test_pipeline_returns_validation_error_without_throwing(tmp_path: Path):
    result = process_satellite_request({
        "workflow": "single-sentinel-2",
        "aoi": {"bbox": (0, 0, 1, 1)},
    })

    assert result.status == "failed"
    assert result.errors
    assert "ValidationError" in result.errors[0]


def test_pipeline_reports_unsupported_before_after_optical_sar(tmp_path: Path):
    result = process_satellite_request(_base_request(
        PipelineWorkflow.BEFORE_AFTER_OPTICAL_SAR.value, tmp_path / "unsupported"
    ))

    assert result.status == "failed"
    assert "not supported" in result.errors[0]