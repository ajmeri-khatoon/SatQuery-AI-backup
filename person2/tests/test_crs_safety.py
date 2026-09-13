import pytest
from pydantic import ValidationError

from app.models import ProcessingConfig


def test_metre_resolution_rejects_geographic_target_crs():
    with pytest.raises(ValidationError, match="projected target_crs"):
        ProcessingConfig(target_crs="EPSG:4326", resolution_m=10)


def test_degree_scale_processing_remains_explicitly_available():
    config = ProcessingConfig(target_crs="EPSG:4326", resolution_m=0.01)
    assert config.target_crs == "EPSG:4326"