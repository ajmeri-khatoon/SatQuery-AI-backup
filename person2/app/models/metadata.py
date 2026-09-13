"""Structured metadata contracts for processed imagery."""

from __future__ import annotations

from datetime import date, datetime

from pyproj import CRS
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProcessingConfig(BaseModel):
    """Explicit preprocessing choices recorded for reproducibility."""

    model_config = ConfigDict(extra="forbid")

    target_crs: str = "EPSG:4326"
    resolution_m: float = Field(default=10, gt=0)
    resampling: str = "nearest"
    nodata: float | int | None = None
    compression: str = "deflate"
    tiled: bool = True

    @model_validator(mode="after")
    def validate_resolution_crs(self) -> "ProcessingConfig":
        try:
            is_geographic = CRS.from_user_input(self.target_crs).is_geographic
        except Exception as exc:
            raise ValueError(f"Invalid target_crs: {self.target_crs}") from exc
        if is_geographic and self.resolution_m >= 1:
            raise ValueError(
                "A projected target_crs is required when resolution_m is metre-scale; "
                "do not interpret metres as degrees"
            )
        return self


class ImageMetadata(BaseModel):
    """Metadata contract shared with downstream AI and change detection."""

    model_config = ConfigDict(extra="forbid")

    sensor: str
    satellite: str
    acquisition_date: date
    processing_date: datetime
    crs: str
    epsg: int | None = Field(default=None, ge=0)
    resolution_m: float = Field(gt=0)
    resolution: tuple[float, float] | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bounds: tuple[float, float, float, float]
    transform: tuple[float, ...]
    bands: list[str]
    polarization: list[str] = Field(default_factory=list)
    dtype: str
    nodata: float | int | None = None
    aoi: dict
    source: str
    tile_id: str | None = None
    parent_image: str | None = None
    preprocessing_steps: list[str] = Field(default_factory=list)
    extra: dict[str, str | float | int | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_complete_metadata(self) -> "ImageMetadata":
        if len(self.transform) not in (6, 9):
            raise ValueError("transform must contain 6 or 9 affine coefficients")
        if len(self.bounds) != 4 or self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]:
            raise ValueError("bounds must be ordered as left, bottom, right, top")
        if self.resolution is None:
            self.resolution = (self.resolution_m, self.resolution_m)
        if len(self.resolution) != 2 or min(self.resolution) <= 0:
            raise ValueError("resolution must contain two positive values")
        if not self.bands:
            raise ValueError("bands must contain at least one band")
        if not self.preprocessing_steps:
            self.preprocessing_steps = ["metadata_extracted"]
        return self


class TileMetadata(BaseModel):
    """Complete structured metadata for one generated tile."""

    model_config = ConfigDict(extra="forbid")

    tile_id: str
    source_image: str
    output_path: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bounds: tuple[float, float, float, float]
    crs: str
    epsg: int | None = Field(default=None, ge=0)
    resolution: tuple[float, float]
    transform: tuple[float, ...]
    bands: list[str]
    polarization: list[str] = Field(default_factory=list)
    dtype: str
    nodata: float | int | None = None
    aoi: dict = Field(default_factory=dict)
    parent_image: str
    preprocessing_steps: list[str] = Field(default_factory=list)
    padded: bool = False
    padding_value: float | int | None = None
    processing_date: datetime

    @model_validator(mode="after")
    def validate_complete_metadata(self) -> "TileMetadata":
        if len(self.transform) not in (6, 9):
            raise ValueError("transform must contain 6 or 9 affine coefficients")
        if self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]:
            raise ValueError("bounds must be ordered as left, bottom, right, top")
        if len(self.resolution) != 2 or min(self.resolution) <= 0:
            raise ValueError("resolution must contain two positive values")
        if not self.bands:
            raise ValueError("bands must contain at least one band")
        if not self.preprocessing_steps:
            self.preprocessing_steps = ["metadata_extracted"]
        if not self.parent_image:
            raise ValueError("parent_image is required")
        return self