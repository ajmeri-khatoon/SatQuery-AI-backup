"""Contracts for before/after satellite imagery workflows."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .aoi import AOI
from .metadata import ImageMetadata
from .request import SatelliteSensor


class BeforeAfterRequest(BaseModel):
    """Validated request for two temporally ordered images."""

    model_config = ConfigDict(extra="forbid")

    aoi: AOI
    sensor: SatelliteSensor
    before_start_date: date
    before_end_date: date | None = None
    after_start_date: date
    after_end_date: date | None = None
    bands: list[str] = Field(default_factory=list)
    resolution_m: float = Field(default=10, gt=0)
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    source: str = "earth-engine"

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "BeforeAfterRequest":
        if self.before_end_date and self.before_end_date < self.before_start_date:
            raise ValueError("before_end_date must be on or after before_start_date")
        if self.after_end_date and self.after_end_date < self.after_start_date:
            raise ValueError("after_end_date must be on or after after_start_date")
        before_end = self.before_end_date or self.before_start_date
        if self.after_start_date <= before_end:
            raise ValueError("after date range must start after the before date range")
        return self


class PairMetadata(BaseModel):
    """Complete pair contract consumed by downstream change detection."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    sensor: str
    before: ImageMetadata
    after: ImageMetadata
    before_aligned_path: str
    after_aligned_path: str
    common_crs: str
    common_resolution: tuple[float, float]
    common_bounds: tuple[float, float, float, float]
    tile_width: int = Field(gt=0)
    tile_height: int = Field(gt=0)
    overlap: int = Field(ge=0)
    before_tiles: list[dict]
    after_tiles: list[dict]
    preprocessing_steps: list[str]