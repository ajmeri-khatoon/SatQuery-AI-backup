"""Area-of-interest validation and normalization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shapely.geometry import box, shape


class AOI(BaseModel):
    """An AOI supplied as GeoJSON geometry, bounds, or a point coordinate."""

    model_config = ConfigDict(extra="forbid")

    geometry: dict[str, Any] | None = None
    bbox: tuple[float, float, float, float] | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def validate_definition(self) -> "AOI":
        supplied = int(self.geometry is not None) + int(self.bbox is not None) + int(
            self.latitude is not None or self.longitude is not None
        )
        if supplied != 1:
            raise ValueError("Provide exactly one of geometry, bbox, or latitude/longitude")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if self.bbox is not None:
            min_x, min_y, max_x, max_y = self.bbox
            if min_x >= max_x or min_y >= max_y:
                raise ValueError("bbox must be ordered as min_x, min_y, max_x, max_y")
        if self.geometry is not None:
            try:
                candidate = shape(self.geometry)
            except (TypeError, ValueError) as exc:
                raise ValueError("geometry must be valid GeoJSON") from exc
            if candidate.is_empty or not candidate.is_valid:
                raise ValueError("geometry must be non-empty and valid")
        return self

    @property
    def shapely_geometry(self):
        """Return the AOI as a Shapely geometry in the declared CRS."""

        if self.geometry is not None:
            return shape(self.geometry)
        if self.bbox is not None:
            return box(*self.bbox)
        return shape({
            "type": "Point",
            "coordinates": [self.longitude, self.latitude],
        })