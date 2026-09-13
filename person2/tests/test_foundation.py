from datetime import date

import pytest
from pydantic import ValidationError

from app.config import AppConfig
from app.models import AOI, ImageMetadata, ProcessingConfig, SatelliteRequest, SatelliteSensor


def test_aoi_accepts_bbox_and_exposes_shapely_geometry():
    aoi = AOI(bbox=(72.0, 18.0, 72.5, 18.5))
    assert aoi.shapely_geometry.bounds == (72.0, 18.0, 72.5, 18.5)


def test_aoi_rejects_invalid_bbox():
    with pytest.raises(ValidationError):
        AOI(bbox=(72.5, 18.0, 72.0, 18.5))


def test_satellite_request_rejects_reversed_date_range():
    with pytest.raises(ValidationError):
        SatelliteRequest(
            aoi=AOI(latitude=18.5, longitude=73.8),
            sensor=SatelliteSensor.SENTINEL_2,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 1),
        )


def test_metadata_and_processing_config_are_structured():
    processing = ProcessingConfig(target_crs="EPSG:32643")
    metadata = ImageMetadata(
        sensor="sentinel-2",
        satellite="Sentinel-2A",
        acquisition_date=date(2026, 2, 1),
        processing_date="2026-02-02T10:00:00Z",
        crs=processing.target_crs,
        resolution_m=10,
        width=10,
        height=10,
        bounds=(0, 0, 1, 1),
        transform=(10, 0, 0, 0, -10, 0),
        bands=["B04"],
        dtype="uint16",
        aoi={"type": "Point", "coordinates": [73.8, 18.5]},
        source="test",
    )
    assert metadata.crs == "EPSG:32643"


def test_config_resolves_default_data_root():
    config = AppConfig()
    assert config.resolved_data_root.name == "data"