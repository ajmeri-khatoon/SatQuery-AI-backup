"""High-level satellite pipeline request and result contracts."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .aoi import AOI
from .request import SatelliteSensor
from .sentinel1 import Polarization


class PipelineWorkflow(StrEnum):
    SINGLE_SENTINEL_2 = "single-sentinel-2"
    SINGLE_SENTINEL_1 = "single-sentinel-1"
    BEFORE_AFTER_SENTINEL_2 = "before-after-sentinel-2"
    BEFORE_AFTER_SENTINEL_1 = "before-after-sentinel-1"
    OPTICAL_SAR = "optical-sar"
    BEFORE_AFTER_OPTICAL_SAR = "before-after-optical-sar"


class SatellitePipelineRequest(BaseModel):
    """One validated request shape for the high-level pipeline facade."""

    model_config = ConfigDict(extra="forbid")

    workflow: PipelineWorkflow
    aoi: AOI
    start_date: date | None = None
    end_date: date | None = None
    before_start_date: date | None = None
    before_end_date: date | None = None
    after_start_date: date | None = None
    after_end_date: date | None = None
    optical_start_date: date | None = None
    optical_end_date: date | None = None
    sar_start_date: date | None = None
    sar_end_date: date | None = None
    bands: list[str] = Field(default_factory=list)
    optical_bands: list[str] = Field(default_factory=list)
    polarizations: list[Polarization] = Field(default_factory=list)
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    max_date_difference_days: int = Field(default=3, ge=0)
    resolution_m: float = Field(default=10, gt=0)
    resampling: str = "nearest"
    nodata: float | int | None = None
    target_crs: str | None = None
    source: str = "earth-engine"
    source_path: Path | None = None
    before_source: Path | None = None
    after_source: Path | None = None
    optical_source: Path | None = None
    sar_source: Path | None = None
    output_dir: Path | None = None
    tile: bool = False
    tile_width: int = Field(default=256, gt=0)
    tile_height: int = Field(default=256, gt=0)
    overlap: int = Field(default=0, ge=0)
    padding: bool = False

    @model_validator(mode="after")
    def validate_workflow_fields(self) -> "SatellitePipelineRequest":
        if self.overlap >= min(self.tile_width, self.tile_height):
            raise ValueError("overlap must be smaller than both tile dimensions")
        if self.workflow in {
            PipelineWorkflow.SINGLE_SENTINEL_1,
            PipelineWorkflow.SINGLE_SENTINEL_2,
        }:
            if self.start_date is None:
                raise ValueError("start_date is required for a single-image workflow")
            if self.end_date and self.end_date < self.start_date:
                raise ValueError("end_date must be on or after start_date")
        elif self.workflow in {
            PipelineWorkflow.BEFORE_AFTER_SENTINEL_1,
            PipelineWorkflow.BEFORE_AFTER_SENTINEL_2,
        }:
            required = (self.before_start_date, self.after_start_date)
            if any(value is None for value in required):
                raise ValueError("before_start_date and after_start_date are required")
            if self.before_end_date and self.before_end_date < self.before_start_date:
                raise ValueError("before_end_date must be on or after before_start_date")
            if self.after_end_date and self.after_end_date < self.after_start_date:
                raise ValueError("after_end_date must be on or after after_start_date")
            if self.after_start_date <= (self.before_end_date or self.before_start_date):
                raise ValueError("after date range must start after before date range")
        elif self.workflow == PipelineWorkflow.OPTICAL_SAR:
            if self.optical_start_date is None or self.sar_start_date is None:
                raise ValueError("optical_start_date and sar_start_date are required")
        elif self.workflow == PipelineWorkflow.BEFORE_AFTER_OPTICAL_SAR:
            # The existing optical/SAR module supports one optical/SAR pair, not two temporal pairs.
            pass
        return self


class PipelineResult(BaseModel):
    """Structured high-level pipeline result, including non-hidden errors."""

    model_config = ConfigDict(extra="forbid")

    status: str
    workflow: str
    output_paths: dict[str, str] = Field(default_factory=dict)
    metadata_paths: list[str] = Field(default_factory=list)
    tile_paths: dict[str, list[str]] = Field(default_factory=dict)
    processing_information: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)