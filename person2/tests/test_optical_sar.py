from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from app.models import AOI, OpticalSARRequest, Polarization
from app.pairing import OpticalSARPairingError, process_optical_sar


TRANSFORM = Affine.translation(363000, 2068000) * Affine.scale(10, -10)


def _write_optical(path: Path, transform: Affine = TRANSFORM) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=10, count=1,
        dtype="uint16", crs="EPSG:32643", transform=transform, nodata=0,
    ) as dataset:
        dataset.write(np.full((1, 10, 10), 100, dtype=np.uint16))
        dataset.set_band_description(1, "B04")


def _write_sar(path: Path, transform: Affine = TRANSFORM) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=10, count=2,
        dtype="float32", crs="EPSG:32643", transform=transform, nodata=-9999,
    ) as dataset:
        dataset.write(np.stack([
            np.full((10, 10), 12, dtype=np.float32),
            np.full((10, 10), 24, dtype=np.float32),
        ]))
        dataset.descriptions = ("VV", "VH")


def _request(max_days: int = 3) -> OpticalSARRequest:
    return OpticalSARRequest(
        aoi=AOI(bbox=(363000, 2067900, 363100, 2068000), crs="EPSG:32643"),
        optical_start_date=date(2026, 1, 1),
        optical_end_date=date(2026, 1, 2),
        sar_start_date=date(2026, 1, 3),
        sar_end_date=date(2026, 1, 4),
        max_date_difference_days=max_days,
        optical_bands=["B04"],
        sar_polarizations=[Polarization.VV, Polarization.VH],
        resolution_m=10,
        target_crs="EPSG:32643",
    )


def test_valid_optical_sar_pair_generates_common_grid_report(tmp_path: Path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"
    _write_optical(optical)
    _write_sar(sar)

    result = process_optical_sar(
        _request(), tmp_path / "pair", optical_source=optical, sar_source=sar
    )

    report = result.report
    assert report.optical_sensor == "sentinel-2-optical"
    assert report.sar_sensor == "sentinel-1-sar"
    assert report.date_difference_days == 2
    assert report.aoi_coverage == {"sentinel-2": True, "sentinel-1": True}
    assert report.optical_bands == ["B04"]
    assert report.sar_polarizations == ["VV", "VH"]
    assert report.common_crs == "EPSG:32643"
    assert report.common_resolution == (10.0, 10.0)
    assert Path(report.optical_aligned_path).exists()
    assert Path(report.sar_aligned_path).exists()
    assert result.report_path.exists()
    with rasterio.open(report.optical_aligned_path) as optical_aligned:
        with rasterio.open(report.sar_aligned_path) as sar_aligned:
            assert optical_aligned.transform == sar_aligned.transform
            assert optical_aligned.bounds == sar_aligned.bounds
            assert optical_aligned.width == sar_aligned.width
            assert optical_aligned.height == sar_aligned.height


def test_pair_rejects_dates_too_far_apart(tmp_path: Path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"
    _write_optical(optical)
    _write_sar(sar)
    request = _request(max_days=1)

    with pytest.raises(OpticalSARPairingError, match="differ by 2 days"):
        process_optical_sar(request, tmp_path / "pair", optical_source=optical, sar_source=sar)


def test_pair_rejects_incompatible_aoi_coverage(tmp_path: Path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"
    _write_optical(optical)
    _write_sar(sar, Affine.translation(500000, 2100000) * Affine.scale(10, -10))

    with pytest.raises(OpticalSARPairingError, match="not fully covered"):
        process_optical_sar(_request(), tmp_path / "pair", optical_source=optical, sar_source=sar)


def test_pair_rejects_missing_imagery(tmp_path: Path):
    with pytest.raises(OpticalSARPairingError, match="Unable to validate AOI coverage"):
        process_optical_sar(
            _request(),
            tmp_path / "pair",
            optical_source=tmp_path / "missing_optical.tif",
            sar_source=tmp_path / "missing_sar.tif",
        )