from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from app.models import AOI, BeforeAfterRequest, SatelliteSensor
from app.pairing import BeforeAfterError, process_before_after


def _write_image(path: Path, transform: Affine, value: int) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint16",
        crs="EPSG:32643",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(np.full((1, 10, 10), value, dtype=np.uint16))
        dataset.set_band_description(1, "B04")


def _request() -> BeforeAfterRequest:
    return BeforeAfterRequest(
        aoi=AOI(bbox=(363000, 2067900, 363100, 2068000), crs="EPSG:32643"),
        sensor=SatelliteSensor.SENTINEL_2,
        before_start_date=date(2026, 1, 1),
        before_end_date=date(2026, 1, 3),
        after_start_date=date(2026, 2, 1),
        after_end_date=date(2026, 2, 3),
        bands=["B04"],
        resolution_m=10,
    )


def test_compatible_pair_generates_common_grid_tiles_and_metadata(tmp_path: Path):
    before = tmp_path / "before_source.tif"
    after = tmp_path / "after_source.tif"
    transform = Affine.translation(363000, 2068000) * Affine.scale(10, -10)
    _write_image(before, transform, 10)
    _write_image(after, transform, 20)

    result = process_before_after(
        _request(),
        tmp_path / "pair",
        before_source=before,
        after_source=after,
        target_crs="EPSG:32643",
        tile_width=4,
        tile_height=4,
        overlap=1,
    )

    pair = result.pair_metadata
    assert pair.common_crs == "EPSG:32643"
    assert pair.common_resolution == (10.0, 10.0)
    assert pair.common_bounds == pair.before.bounds == pair.after.bounds
    assert len(result.before_tiles) == len(result.after_tiles)
    assert [tile.tile_id for tile in result.before_tiles] == [tile.tile_id for tile in result.after_tiles]
    assert pair.tile_width == 4
    assert pair.overlap == 1
    assert (tmp_path / "pair" / "pair_metadata.json").exists()
    with rasterio.open(result.before_tiles[0].output_path) as before_tile:
        with rasterio.open(result.after_tiles[0].output_path) as after_tile:
            assert before_tile.crs == after_tile.crs
            assert before_tile.transform == after_tile.transform
            assert before_tile.bounds == after_tile.bounds
            assert before_tile.read().mean() == 10
            assert after_tile.read().mean() == 20


def test_incompatible_pair_raises_clear_error(tmp_path: Path):
    before = tmp_path / "before_source.tif"
    after = tmp_path / "after_source.tif"
    _write_image(before, Affine.translation(363000, 2068000) * Affine.scale(10, -10), 10)
    _write_image(after, Affine.translation(500000, 2100000) * Affine.scale(10, -10), 20)

    with pytest.raises(BeforeAfterError, match="cannot be aligned|preprocessing failed"):
        process_before_after(
            _request(),
            tmp_path / "pair",
            before_source=before,
            after_source=after,
            target_crs="EPSG:32643",
        )


def test_request_rejects_overlapping_before_and_after_ranges():
    with pytest.raises(ValueError, match="after date range"):
        BeforeAfterRequest(
            aoi=AOI(bbox=(363000, 2067900, 363100, 2068000), crs="EPSG:32643"),
            sensor=SatelliteSensor.SENTINEL_2,
            before_start_date=date(2026, 1, 1),
            before_end_date=date(2026, 2, 1),
            after_start_date=date(2026, 2, 1),
            bands=["B04"],
        )