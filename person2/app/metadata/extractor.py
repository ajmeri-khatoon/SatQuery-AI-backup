"""Single source of truth for processed raster and tile metadata."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import rasterio

from ..models.metadata import ImageMetadata, TileMetadata


class MetadataExtractionError(RuntimeError):
    """Raised when raster metadata cannot be extracted or validated."""


def _epsg(dataset: rasterio.DatasetReader) -> int | None:
    return dataset.crs.to_epsg() if dataset.crs is not None else None


def extract_image_metadata(
    path: str | Path,
    *,
    sensor: str,
    satellite: str,
    acquisition_date: date,
    source: str,
    aoi: dict,
    preprocessing_steps: Sequence[str],
    polarization: Sequence[str] = (),
    parent_image: str | None = None,
    extra: dict[str, str | float | int | None] | None = None,
) -> ImageMetadata:
    """Extract and validate complete metadata from a processed GeoTIFF."""

    raster_path = Path(path)
    try:
        with rasterio.open(raster_path) as dataset:
            if dataset.crs is None:
                raise MetadataExtractionError(f"Raster '{raster_path}' has no CRS")
            bounds = dataset.bounds
            metadata = ImageMetadata(
                sensor=sensor,
                satellite=satellite,
                acquisition_date=acquisition_date,
                processing_date=datetime.now(timezone.utc),
                crs=dataset.crs.to_string(),
                epsg=_epsg(dataset),
                resolution_m=float(dataset.res[0]),
                resolution=(float(dataset.res[0]), float(dataset.res[1])),
                width=dataset.width,
                height=dataset.height,
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                transform=tuple(dataset.transform),
                bands=[description or f"band_{index}" for index, description in enumerate(dataset.descriptions, 1)],
                polarization=list(polarization),
                dtype=dataset.dtypes[0],
                nodata=dataset.nodata,
                aoi=aoi,
                source=source,
                parent_image=parent_image,
                preprocessing_steps=list(preprocessing_steps),
                extra=extra or {},
            )
            return metadata
    except MetadataExtractionError:
        raise
    except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
        raise MetadataExtractionError(f"Unable to extract metadata from '{raster_path}': {exc}") from exc


def extract_tile_metadata(
    path: str | Path,
    *,
    tile_id: str,
    source_image: str | Path,
    output_path: str | Path,
    row: int,
    column: int,
    aoi: dict | None = None,
    parent_image: str | Path | None = None,
    preprocessing_steps: Sequence[str] = ("tile_window_read", "geotiff_written"),
    polarization: Sequence[str] = (),
    padded: bool = False,
    padding_value: float | int | None = None,
) -> TileMetadata:
    """Extract and validate complete metadata from one generated tile."""

    tile_path = Path(path)
    try:
        with rasterio.open(tile_path) as dataset:
            if dataset.crs is None:
                raise MetadataExtractionError(f"Tile '{tile_path}' has no CRS")
            bounds = dataset.bounds
            return TileMetadata(
                tile_id=tile_id,
                source_image=str(source_image),
                output_path=str(output_path),
                row=row,
                column=column,
                width=dataset.width,
                height=dataset.height,
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                crs=dataset.crs.to_string(),
                epsg=_epsg(dataset),
                resolution=(float(dataset.res[0]), float(dataset.res[1])),
                transform=tuple(dataset.transform),
                bands=[description or f"band_{index}" for index, description in enumerate(dataset.descriptions, 1)],
                polarization=list(polarization),
                dtype=dataset.dtypes[0],
                nodata=dataset.nodata,
                aoi=aoi or {},
                parent_image=str(parent_image or source_image),
                preprocessing_steps=list(preprocessing_steps),
                padded=padded,
                padding_value=padding_value,
                processing_date=datetime.now(timezone.utc),
            )
    except MetadataExtractionError:
        raise
    except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
        raise MetadataExtractionError(f"Unable to extract tile metadata from '{tile_path}': {exc}") from exc