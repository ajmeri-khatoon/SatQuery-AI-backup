"""Satellite acquisition request contract."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .aoi import AOI


class SatelliteSensor(StrEnum):
    SENTINEL_1 = "sentinel-1"
    SENTINEL_2 = "sentinel-2"


class SatelliteRequest(BaseModel):
    """Validated request passed to a future acquisition adapter."""

    model_config = ConfigDict(extra="forbid")

    aoi: AOI
    sensor: SatelliteSensor
    start_date: date
    end_date: date | None = None
    bands: list[str] = Field(default_factory=list)
    resolution_m: float = Field(default=10, gt=0)
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    source: str = "earth-engine"

    @model_validator(mode="after")
    def validate_dates(self) -> "SatelliteRequest":
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self