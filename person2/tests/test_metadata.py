from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from pydantic import ValidationError

from app.metadata import extract_image_metadata, extract_tile_metadata
from app.models import ImageMetadata


def _write_raster(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=2,
        dtype="float32",
        crs="EPSG:32643",
        transform=Affine.translation(363000, 2068000) * Affine.scale(10, -10),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.ones((2, 3, 4), dtype=np.float32))
        dataset.descriptions = ("VV", "VH")


def test_image_metadata_extractor_contains_complete_json_schema(tmp_path: Path):
    raster = tmp_path / "processed.tif"
    _write_raster(raster)

    metadata = extract_image_metadata(
        raster,
        sensor="sentinel-1-sar",
        satellite="Sentinel-1A",
        acquisition_date=date(2026, 1, 2),
        source="test",
        aoi={"type": "Point", "coordinates": [73.8, 18.5]},
        polarization=["VV", "VH"],
        parent_image="raw.tif",
        preprocessing_steps=["reprojected", "geotiff_written"],
    )

    assert metadata.epsg == 32643
    assert metadata.resolution == (10.0, 10.0)
    assert metadata.polarization == ["VV", "VH"]
    assert metadata.parent_image == "raw.tif"
    serialized = metadata.model_dump_json()
    assert '"sensor":"sentinel-1-sar"' in serialized
    assert '"epsg":32643' in serialized


def test_tile_metadata_uses_same_extractor_and_parent_image(tmp_path: Path):
    raster = tmp_path / "tile.tif"
    _write_raster(raster)

    metadata = extract_tile_metadata(
        raster,
        tile_id="tile_0001",
        source_image="aligned.tif",
        output_path=raster,
        row=0,
        column=0,
        aoi={"type": "Point", "coordinates": [73.8, 18.5]},
        parent_image="aligned.tif",
        polarization=["VV", "VH"],
    )

    assert metadata.epsg == 32643
    assert metadata.parent_image == "aligned.tif"
    assert metadata.polarization == ["VV", "VH"]
    assert metadata.preprocessing_steps


def test_incomplete_image_metadata_is_rejected():
    with pytest.raises(ValidationError):
        ImageMetadata(
            satellite="Sentinel-2A",
            acquisition_date=date(2026, 1, 1),
            processing_date="2026-01-02T00:00:00Z",
            crs="EPSG:32643",
            resolution_m=10,
            width=1,
            height=1,
            bounds=(0, 0, 1, 1),
            transform=(1, 0),
            bands=["B04"],
            dtype="uint16",
            aoi={},
            source="test",
        )