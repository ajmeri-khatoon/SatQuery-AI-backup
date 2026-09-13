"""Contracts for Sentinel-2 optical and Sentinel-1 SAR pairing."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .aoi import AOI
from .sentinel1 import Polarization


class OpticalSARRequest(BaseModel):
    """Validated date-window and sensor configuration for an optical/SAR pair."""

    model_config = ConfigDict(extra="forbid")

    aoi: AOI
    optical_start_date: date
    optical_end_date: date | None = None
    sar_start_date: date
    sar_end_date: date | None = None
    max_date_difference_days: int = Field(default=3, ge=0)
    optical_bands: list[str] = Field(default_factory=lambda: ["B04", "B03", "B02", "B08"])
    sar_polarizations: list[Polarization] = Field(
        default_factory=lambda: [Polarization.VV, Polarization.VH], min_length=1
    )
    resolution_m: float = Field(default=10, gt=0)
    target_crs: str | None = None
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    source: str = "earth-engine"

    @model_validator(mode="after")
    def validate_ranges(self) -> "OpticalSARRequest":
        if self.optical_end_date and self.optical_end_date < self.optical_start_date:
            raise ValueError("optical_end_date must be on or after optical_start_date")
        if self.sar_end_date and self.sar_end_date < self.sar_start_date:
            raise ValueError("sar_end_date must be on or after sar_start_date")
        self.sar_polarizations = list(dict.fromkeys(self.sar_polarizations))
        return self


class OpticalSARPairReport(BaseModel):
    """Pairing and common-grid report for downstream change detection."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    optical_sensor: str = "sentinel-2-optical"
    sar_sensor: str = "sentinel-1-sar"
    optical_scene_id: str
    sar_scene_id: str
    optical_acquisition_date: date
    sar_acquisition_date: date
    date_difference_days: int
    max_date_difference_days: int
    aoi: dict
    aoi_coverage: dict[str, bool]
    optical_bands: list[str]
    sar_polarizations: list[str]
    optical_source_path: str
    sar_source_path: str
    optical_processed_path: str
    sar_processed_path: str
    optical_aligned_path: str
    sar_aligned_path: str
    common_crs: str
    common_resolution: tuple[float, float]
    common_bounds: tuple[float, float, float, float]
    common_dimensions: tuple[int, int]
    common_transform: tuple[float, ...]
    preprocessing_steps: list[str]