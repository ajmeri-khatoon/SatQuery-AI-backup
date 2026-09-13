"""Sentinel-1-specific request contracts."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .aoi import AOI


class Polarization(StrEnum):
    """Sentinel-1 GRD polarization channels."""

    VV = "VV"
    VH = "VH"


class Sentinel1Request(BaseModel):
    """Validated Sentinel-1 acquisition request."""

    model_config = ConfigDict(extra="forbid")

    aoi: AOI
    start_date: date
    end_date: date | None = None
    polarizations: list[Polarization] = Field(
        default_factory=lambda: [Polarization.VV, Polarization.VH], min_length=1
    )
    resolution_m: float = Field(default=10, gt=0)
    source: str = "earth-engine"

    @model_validator(mode="after")
    def validate_dates_and_polarizations(self) -> "Sentinel1Request":
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        self.polarizations = list(dict.fromkeys(self.polarizations))
        return self