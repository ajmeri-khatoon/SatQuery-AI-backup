from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from app.acquisition.sentinel1 import Sentinel1Acquirer
from app.models import AOI, Polarization, Sentinel1Request
from app.models.metadata import ProcessingConfig
from app.preprocessing.sentinel1 import prepare_sentinel1


def test_sentinel1_preprocessing_preserves_sar_values_and_metadata(tmp_path: Path):
    source_path = tmp_path / "sar_raw.tif"
    output_path = tmp_path / "sar_processed.tif"
    values = np.stack([
        np.full((10, 10), 12, dtype=np.float32),
        np.full((10, 10), 24, dtype=np.float32),
    ])
    with rasterio.open(
        source_path, "w", driver="GTiff", width=10, height=10, count=2,
        dtype="float32", crs="EPSG:4326",
        transform=Affine.translation(73.7, 18.7) * Affine.scale(0.01, -0.01),
        nodata=-9999,
    ) as source:
        source.write(values)
        source.descriptions = ("VV", "VH")

    metadata = prepare_sentinel1(
        source_path, output_path,
        aoi=AOI(bbox=(73.72, 18.62, 73.76, 18.68)),
        acquisition_date=date(2026, 1, 2),
        polarizations=[Polarization.VV, Polarization.VH],
        processing=ProcessingConfig(target_crs="EPSG:32643", resolution_m=100, nodata=-9999),
    )

    assert metadata.sensor == "sentinel-1-sar"
    assert metadata.bands == ["VV", "VH"]
    assert metadata.extra["value_domain"].startswith("SAR backscatter")
    with rasterio.open(output_path) as output:
        result = output.read()
        assert output.crs.to_epsg() == 32643
        assert output.descriptions == ("VV", "VH")
        assert output.nodata == -9999
        assert np.isclose(result[0][result[0] != -9999].mean(), 12, atol=0.01)
        assert np.isclose(result[1][result[1] != -9999].mean(), 24, atol=0.01)


def test_sentinel1_can_select_single_polarization(tmp_path: Path):
    source_path = tmp_path / "vv.tif"
    output_path = tmp_path / "vv_processed.tif"
    with rasterio.open(
        source_path, "w", driver="GTiff", width=4, height=4, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=Affine.translation(73.7, 18.7) * Affine.scale(0.01, -0.01),
    ) as source:
        source.write(np.ones((1, 4, 4), dtype=np.float32) * 3)

    metadata = prepare_sentinel1(
        source_path, output_path,
        aoi=AOI(bbox=(73.7, 18.67, 73.74, 18.7)),
        acquisition_date=date(2026, 1, 2),
        polarizations=[Polarization.VV],
        processing=ProcessingConfig(target_crs="EPSG:4326", resolution_m=0.01),
    )

    assert metadata.bands == ["VV"]


def test_sentinel1_acquisition_applies_polarization_filters_without_network():
    class Value:
        def __init__(self, value): self.value = value
        def getInfo(self): return self.value

    class Image:
        def id(self): return Value("COPERNICUS/S1_GRD/TEST")
        def date(self): return self
        def format(self, _pattern): return Value("2026-01-02")
        def get(self, name): return Value("ASCENDING" if name == "orbitProperties_pass" else "IW")

    class Collection:
        def __init__(self): self.filters = []
        def filterBounds(self, value): self.filters.append(("bounds", value)); return self
        def filterDate(self, start, end): self.filters.append(("date", start, end)); return self
        def filter(self, value): self.filters.append(value); return self
        def size(self): return Value(1)
        def first(self): return Image()

    class FakeEE:
        class Filter:
            @staticmethod
            def eq(name, value): return ("eq", name, value)
            @staticmethod
            def listContains(name, value): return ("contains", name, value)
        def __init__(self): self.collection = Collection()
        def Initialize(self, project): pass
        def Geometry(self, value): return value
        def ImageCollection(self, _name): return self.collection
        def Image(self, image): return image

    request = Sentinel1Request(
        aoi=AOI(bbox=(73.8, 18.5, 73.81, 18.51)),
        start_date=date(2026, 1, 1), end_date=date(2026, 1, 3),
        polarizations=[Polarization.VV, Polarization.VH],
    )
    scene = Sentinel1Acquirer(ee_module=FakeEE()).search(request)

    assert scene.polarizations == [Polarization.VV, Polarization.VH]
    assert scene.orbit_pass == "ASCENDING"