"""Window-based GeoTIFF tiling without image resizing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

from ..metadata.extractor import extract_tile_metadata
from ..models.metadata import TileMetadata

LOGGER = logging.getLogger(__name__)


class TilingError(RuntimeError):
    """Raised when a raster cannot be tiled safely."""


def _validate_options(tile_width: int, tile_height: int, overlap: int) -> None:
    if tile_width < 1 or tile_height < 1:
        raise TilingError("tile_width and tile_height must be positive integers")
    if overlap < 0:
        raise TilingError("overlap must be zero or greater")
    if overlap >= min(tile_width, tile_height):
        raise TilingError("overlap must be smaller than both tile dimensions")


def _starts(length: int, tile_length: int, step: int) -> list[int]:
    """Return deterministic row/column starts that cover the full source."""

    return list(range(0, length, step))


def _block_size(length: int) -> int:
    """Choose a valid GeoTIFF block size for a sufficiently large tile."""

    return max(16, min(256, (length // 16) * 16))


def create_tiles(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    tile_width: int = 256,
    tile_height: int = 256,
    overlap: int = 0,
    padding: bool = False,
    padding_value: float | int | None = None,
    compression: str = "deflate",
    tiled: bool = True,
    overwrite: bool = False,
    metadata_path: str | Path | None = None,
    aoi: dict | None = None,
    polarization: Sequence[str] = (),
) -> list[TileMetadata]:
    """Create deterministic GeoTIFF tiles from a source raster.

    Tiling reads source windows directly and never resizes or resamples pixels.
    Without padding, edge tiles are smaller than the configured dimensions. With
    padding, edge tiles have the configured dimensions and pixels outside the
    source are filled with ``padding_value`` (or source nodata, or zero).
    Overlap is measured in source pixels and applies in both dimensions.
    """

    source_path = Path(input_path)
    destination_dir = Path(output_dir)
    _validate_options(tile_width, tile_height, overlap)
    try:
        dataset = rasterio.open(source_path)
    except (rasterio.errors.RasterioIOError, OSError) as exc:
        raise TilingError(f"Unable to open source raster '{source_path}': {exc}") from exc

    metadata: list[TileMetadata] = []
    step_x = tile_width - overlap
    step_y = tile_height - overlap
    try:
        with dataset:
            if dataset.crs is None:
                raise TilingError(f"Source raster '{source_path}' has no CRS")
            if dataset.width < 1 or dataset.height < 1 or dataset.count < 1:
                raise TilingError(f"Source raster '{source_path}' has invalid dimensions")
            fill_value = padding_value
            if fill_value is None:
                fill_value = dataset.nodata if dataset.nodata is not None else 0
            band_names = [
                description or f"band_{index}"
                for index, description in enumerate(dataset.descriptions, 1)
            ]
            row_starts = _starts(dataset.height, tile_height, step_y)
            column_starts = _starts(dataset.width, tile_width, step_x)
            tile_number = 1
            for row, row_start in enumerate(row_starts):
                for column, column_start in enumerate(column_starts):
                    actual_width = min(tile_width, dataset.width - column_start)
                    actual_height = min(tile_height, dataset.height - row_start)
                    window_width = tile_width if padding else actual_width
                    window_height = tile_height if padding else actual_height
                    source_window = Window(column_start, row_start, window_width, window_height)
                    tile_id = f"tile_{tile_number:04d}"
                    tile_path = destination_dir / f"{tile_id}.tif"
                    if tile_path.exists() and not overwrite:
                        raise TilingError(f"Refusing to overwrite existing tile '{tile_path}'")
                    data = dataset.read(
                        window=source_window,
                        boundless=padding,
                        fill_value=fill_value,
                    )
                    tile_transform = window_transform(source_window, dataset.transform)
                    tile_bounds = window_bounds(source_window, dataset.transform)
                    profile = dataset.profile.copy()
                    internal_tiling = tiled and window_width >= 16 and window_height >= 16
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)
                    profile.update(
                        driver="GTiff",
                        width=window_width,
                        height=window_height,
                        transform=tile_transform,
                        compress=compression,
                        tiled=internal_tiling,
                    )
                    if internal_tiling:
                        profile.update(
                            blockxsize=_block_size(window_width),
                            blockysize=_block_size(window_height),
                        )
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with rasterio.open(tile_path, "w", **profile) as tile:
                            tile.write(data)
                            tile.descriptions = tuple(band_names)
                    except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
                        raise TilingError(f"Unable to write tile '{tile_path}': {exc}") from exc
                    metadata.append(
                        extract_tile_metadata(
                            tile_path,
                            tile_id=tile_id,
                            source_image=source_path,
                            output_path=tile_path,
                            row=row,
                            column=column,
                            aoi=aoi,
                            parent_image=source_path,
                            polarization=polarization,
                            padded=padding and (window_width > actual_width or window_height > actual_height),
                            padding_value=fill_value if padding else None,
                        )
                    )
                    tile_number += 1
    except TilingError:
        raise
    except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
        raise TilingError(f"Unable to tile source raster '{source_path}': {exc}") from exc

    if metadata_path is not None:
        metadata_file = Path(metadata_path)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(
            json.dumps([item.model_dump(mode="json") for item in metadata], indent=2),
            encoding="utf-8",
        )
    LOGGER.info("Created %d tiles from %s", len(metadata), source_path)
    return metadata