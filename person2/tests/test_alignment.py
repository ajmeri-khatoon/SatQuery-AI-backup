from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from app.preprocessing.alignment import AlignmentError, align_images


def _write(path: Path, *, crs: str, transform: Affine, value: int, size: int = 10):
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=1,
        dtype="uint16", crs=crs, transform=transform, nodata=0,
    ) as dataset:
        dataset.write(np.full((1, size, size), value, dtype=np.uint16))
        dataset.set_band_description(1, "change-band")


def test_alignment_supports_different_crs_resolution_and_common_grid(tmp_path: Path):
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    output_a = tmp_path / "aligned_a.tif"
    output_b = tmp_path / "aligned_b.tif"
    _write(image_a, crs="EPSG:4326", transform=Affine.translation(73.7, 18.7) * Affine.scale(0.01, -0.01), value=10)
    _write(image_b, crs="EPSG:32643", transform=Affine.translation(363000, 2068000) * Affine.scale(100, -100), value=20)

    report = align_images(image_a, image_b, output_a, output_b, target_crs="EPSG:32643", resolution=100)

    assert report.source_crs_a == "EPSG:4326"
    assert report.source_crs_b == "EPSG:32643"
    assert report.source_transform_a != report.source_transform_b
    assert len(report.target_transform) == 9
    assert report.target_dimensions[0] > 0 and report.target_dimensions[1] > 0
    with rasterio.open(output_a) as aligned_a, rasterio.open(output_b) as aligned_b:
        assert aligned_a.crs == aligned_b.crs
        assert aligned_a.transform == aligned_b.transform
        assert aligned_a.width == aligned_b.width
        assert aligned_a.height == aligned_b.height
        assert aligned_a.bounds == aligned_b.bounds
        assert aligned_a.read().mean() == 10
        assert aligned_b.read().mean() == 20


def test_alignment_rejects_non_overlapping_images(tmp_path: Path):
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    transform = Affine.translation(0, 10) * Affine.scale(1, -1)
    _write(image_a, crs="EPSG:4326", transform=transform, value=1)
    _write(image_b, crs="EPSG:4326", transform=Affine.translation(100, 110) * Affine.scale(1, -1), value=2)

    with pytest.raises(AlignmentError, match="common valid area is empty"):
        align_images(image_a, image_b, tmp_path / "out_a.tif", tmp_path / "out_b.tif")


def test_alignment_rejects_missing_crs(tmp_path: Path):
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    _write(image_b, crs="EPSG:4326", transform=Affine.translation(0, 10) * Affine.scale(1, -1), value=2)
    with rasterio.open(image_a, "w", driver="GTiff", width=10, height=10, count=1, dtype="uint16") as dataset:
        dataset.write(np.ones((1, 10, 10), dtype=np.uint16))

    with pytest.raises(AlignmentError, match="has no CRS"):
        align_images(image_a, image_b, tmp_path / "out_a.tif", tmp_path / "out_b.tif")