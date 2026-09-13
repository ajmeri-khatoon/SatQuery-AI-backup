from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from app.tiling import create_tiles


def _write_source(path: Path) -> tuple[np.ndarray, Affine]:
    transform = Affine.translation(500000, 2000000) * Affine.scale(10, -10)
    values = np.stack([
        np.arange(70, dtype=np.uint16).reshape(7, 10),
        np.full((7, 10), 900, dtype=np.uint16),
    ])
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=7, count=2,
        dtype="uint16", crs="EPSG:32643", transform=transform, nodata=0,
    ) as dataset:
        dataset.write(values)
        dataset.descriptions = ("VV", "VH")
    return values, transform


def test_tiles_preserve_transforms_bounds_bands_and_edge_sizes(tmp_path: Path):
    source = tmp_path / "source.tif"
    values, source_transform = _write_source(source)

    metadata = create_tiles(source, tmp_path / "tiles", tile_width=4, tile_height=3, overlap=1)

    assert len(metadata) == 16
    assert [item.tile_id for item in metadata[:3]] == ["tile_0001", "tile_0002", "tile_0003"]
    for item in metadata:
        with rasterio.open(item.output_path) as tile:
            assert tile.crs.to_epsg() == 32643
            assert tile.transform == Affine(*item.transform)
            assert tile.bounds == item.bounds
            assert tile.descriptions == ("VV", "VH")
            assert tile.width == item.width
            assert tile.height == item.height
    first = metadata[0]
    with rasterio.open(first.output_path) as tile:
        assert tile.transform == source_transform
        assert np.array_equal(tile.read(), values[:, :3, :4])
    last = metadata[-1]
    assert (last.width, last.height) == (1, 1)
    assert last.padded is False


def test_padding_keeps_fixed_dimensions_and_records_fill_value(tmp_path: Path):
    source = tmp_path / "source.tif"
    _write_source(source)

    metadata = create_tiles(
        source,
        tmp_path / "padded",
        tile_width=4,
        tile_height=3,
        overlap=1,
        padding=True,
        padding_value=777,
        metadata_path=tmp_path / "tiles.json",
    )

    assert all((item.width, item.height) == (4, 3) for item in metadata)
    assert any(item.padded for item in metadata)
    assert metadata[-1].padding_value == 777
    assert (tmp_path / "tiles.json").exists()
    with rasterio.open(metadata[-1].output_path) as tile:
        assert 777 in tile.read()


def test_tiling_does_not_resize_source_pixels(tmp_path: Path):
    source = tmp_path / "source.tif"
    values, _ = _write_source(source)

    metadata = create_tiles(source, tmp_path / "tiles", tile_width=20, tile_height=20)

    with rasterio.open(metadata[0].output_path) as tile:
        assert (tile.width, tile.height) == (10, 7)
        assert np.array_equal(tile.read(), values)