from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile

from app.models import AOI, ProcessingConfig, SatelliteRequest, SatelliteSensor
from app.acquisition.sentinel2 import Sentinel2Acquirer, _normalize_earth_engine_bands
from app.preprocessing.sentinel2 import prepare_sentinel2


def test_sentinel2_preprocessing_reprojects_clips_and_writes_metadata(tmp_path: Path):
    source_path = tmp_path / "raw.tif"
    output_path = tmp_path / "processed.tif"
    transform = Affine.translation(73.7, 18.7) * Affine.scale(0.01, -0.01)
    values = np.arange(100, dtype=np.uint16).reshape(1, 10, 10)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as source:
        source.write(values)
        source.set_band_description(1, "B04")

    metadata = prepare_sentinel2(
        source_path,
        output_path,
        aoi=AOI(bbox=(73.72, 18.62, 73.76, 18.68)),
        acquisition_date=date(2026, 1, 2),
        processing=ProcessingConfig(target_crs="EPSG:32643", resolution_m=100, nodata=0),
        bands=["B04"],
    )

    assert output_path.exists()
    assert metadata.sensor == "sentinel-2"
    assert metadata.crs == "EPSG:32643"
    assert metadata.bands == ["B04"]
    with rasterio.open(output_path) as output:
        assert output.crs.to_epsg() == 32643
        assert output.nodata == 0
        assert output.width > 0 and output.height > 0
        assert output.descriptions == ("B04",)


def test_sentinel2_acquisition_search_uses_filters_without_network():
    class Value:
        def __init__(self, value):
            self.value = value

        def getInfo(self):
            return self.value

    class Image:
        def id(self):
            return Value("COPERNICUS/S2_SR_HARMONIZED/TEST")

        def date(self):
            return self

        def format(self, _pattern):
            return Value("2026-01-02")

        def get(self, _name):
            return Value(4.5)

    class Collection:
        def __init__(self):
            self.calls = []

        def filterBounds(self, value):
            self.calls.append(("filterBounds", value))
            return self

        def filterDate(self, start, end):
            self.calls.append(("filterDate", start, end))
            return self

        def filter(self, value):
            self.calls.append(("filter", value))
            return self

        def sort(self, value):
            self.calls.append(("sort", value))
            return self

        def size(self):
            return Value(1)

        def first(self):
            return Image()

    class FakeEE:
        class Filter:
            @staticmethod
            def lte(name, value):
                return (name, value)

        def __init__(self):
            self.collection = Collection()

        def Initialize(self, project):
            self.project = project

        def Geometry(self, value):
            return value

        def ImageCollection(self, _name):
            return self.collection

        def Image(self, image):
            return image

    request = SatelliteRequest(
        aoi=AOI(bbox=(73.7, 18.6, 73.8, 18.7)),
        sensor=SatelliteSensor.SENTINEL_2,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 3),
        bands=["B04", "B08"],
        max_cloud_cover=10,
    )
    scene = Sentinel2Acquirer(ee_module=FakeEE()).search(request)

    assert scene.image_id.endswith("TEST")
    assert scene.acquisition_date == date(2026, 1, 2)
    assert scene.cloud_percentage == 4.5


def test_sentinel2_band_names_are_normalized_for_earth_engine():
    assert _normalize_earth_engine_bands(["B02", "B03", "B04", "B08", "B8A"]) == [
        "B2", "B3", "B4", "B8", "B8A"
    ]


def test_sentinel2_acquisition_writes_requested_band_descriptions(tmp_path: Path):
    raster_bytes = MemoryFile()
    with raster_bytes.open(
        driver="GTiff", width=2, height=2, count=4, dtype="uint16",
        crs="EPSG:4326", transform=Affine.scale(0.01, -0.01),
    ) as dataset:
        dataset.write(np.ones((4, 2, 2), dtype=np.uint16))
    payload = raster_bytes.read()

    class Value:
        def __init__(self, value):
            self.value = value

        def getInfo(self):
            return self.value

    class Image:
        def id(self):
            return Value("COPERNICUS/S2_SR_HARMONIZED/TEST")

        def date(self):
            return self

        def format(self, _pattern):
            return Value("2026-01-02")

        def get(self, _name):
            return Value(4.5)

        def select(self, bands):
            assert bands == ["B4", "B3", "B2", "B8"]
            return self

        def getDownloadURL(self, _parameters):
            return "https://example.invalid/download"

    class Collection:
        def filterBounds(self, _value): return self
        def filterDate(self, _start, _end): return self
        def sort(self, _value): return self
        def size(self): return Value(1)
        def first(self): return Image()

    class FakeEE:
        def Initialize(self, project): pass
        def Geometry(self, value): return value
        def ImageCollection(self, _name): return Collection()
        def Image(self, _image_id): return Image()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self): return payload

    request = SatelliteRequest(
        aoi=AOI(bbox=(0, -0.02, 0.02, 0), crs="EPSG:4326"),
        sensor=SatelliteSensor.SENTINEL_2,
        start_date=date(2026, 1, 1),
        bands=["B04", "B03", "B02", "B08"],
    )
    output = tmp_path / "sentinel2.tif"
    Sentinel2Acquirer(
        ee_module=FakeEE(),
        urlopen_function=lambda _url: Response(),
    ).download(request, output)

    with rasterio.open(output) as downloaded:
        assert downloaded.descriptions == ("B04", "B03", "B02", "B08")


def test_sentinel2_preprocessing_rejects_unnamed_external_bands(tmp_path: Path):
    source = tmp_path / "unnamed.tif"
    with rasterio.open(
        source, "w", driver="GTiff", width=2, height=2, count=4,
        dtype="uint16", crs="EPSG:4326", transform=Affine.scale(0.01, -0.01),
    ) as dataset:
        dataset.write(np.ones((4, 2, 2), dtype=np.uint16))

    with pytest.raises(Exception, match="Requested band 'B04' is unavailable"):
        prepare_sentinel2(
            source, tmp_path / "processed.tif",
            aoi=AOI(bbox=(0, -0.02, 0.02, 0), crs="EPSG:4326"),
            acquisition_date=date(2026, 1, 2),
            processing=ProcessingConfig(target_crs="EPSG:32643", resolution_m=100),
            bands=["B04"],
        )