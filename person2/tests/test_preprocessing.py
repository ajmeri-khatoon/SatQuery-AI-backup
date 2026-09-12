# ruff: noqa: E402

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # type: ignore[import-untyped]

from person2.preprocessing import (
    BeforeAfterInput,
    PreprocessingError,
    SingleImageInput,
    SpatialCompatibilityError,
    inspect_raster,
    normalize_image,
    prepare,
    tile_image,
)


def write_raster(path: Path, data: Any, x_origin: float = 100) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(x_origin, 20, 10, 10),
        nodata=-9999,
    ) as dataset:
        dataset.write(data)
        dataset.update_tags(source="synthetic")


def test_inspect_raster_returns_deterministic_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.tif"
    write_raster(path, np.arange(24, dtype=np.int16).reshape(2, 3, 4))

    inspection = inspect_raster(path)

    assert inspection.as_dict() == {
        "width": 4,
        "height": 3,
        "band_count": 2,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": (10.0, 0.0, 100.0, 0.0, -10.0, 20.0),
        "bounds": (100.0, -10.0, 140.0, 20.0),
        "resolution": (10.0, 10.0),
        "nodata": -9999.0,
        "metadata": {"AREA_OR_POINT": "Area", "source": "synthetic"},
    }


def test_normalize_image_is_finite_and_handles_nodata() -> None:
    data = np.array([[[1, 2], [3, -9999]]], dtype=np.float32)

    normalized = normalize_image(data, nodata=-9999)

    assert normalized.dtype == np.float32
    assert np.array_equal(normalized, np.array([[[0, 0.5], [1, 0]]], dtype=np.float32))
    assert np.isfinite(normalized).all()


def test_tile_image_includes_edges_with_overlap() -> None:
    data = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

    tiles = tile_image(data, tile_size=3, overlap=1)

    assert [tile.window for tile in tiles] == [
        (0, 0, 3, 3),
        (1, 0, 4, 3),
        (0, 1, 3, 4),
        (1, 1, 4, 4),
    ]


def test_prepare_supports_single_and_before_after(tmp_path: Path) -> None:
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    data = np.arange(16, dtype=np.int16).reshape(1, 4, 4)
    write_raster(before, data)
    write_raster(after, data + 1)

    single = prepare(SingleImageInput(before), tile_size=2)
    pair = prepare(BeforeAfterInput(before, after), tile_size=2)

    assert single.kind == "single"
    assert len(single.tiles["image"]) == 4
    assert pair.kind == "before_after"
    assert set(pair.images) == {"before", "after"}
    assert pair.as_dict()["images"]["before"]["shape"] == (1, 4, 4)


def test_prepare_rejects_spatially_incompatible_pair(tmp_path: Path) -> None:
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    data = np.zeros((1, 2, 2), dtype=np.int16)
    write_raster(first, data)
    write_raster(second, data, x_origin=110)

    with pytest.raises(SpatialCompatibilityError, match="mismatched fields: transform, bounds"):
        prepare(BeforeAfterInput(first, second))


def test_non_tiff_input_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not a raster")

    with pytest.raises(PreprocessingError, match=".tif or .tiff"):
        inspect_raster(path)
