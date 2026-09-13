"""Pixel-grid alignment for georeferenced satellite rasters."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio
from pydantic import BaseModel, ConfigDict, Field
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

LOGGER = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Raised when two rasters cannot be aligned safely."""


class AlignmentReport(BaseModel):
    """Reproducible description of the common grid used for two outputs."""

    model_config = ConfigDict(extra="forbid")

    image_a: str
    image_b: str
    output_a: str
    output_b: str
    source_crs_a: str
    source_crs_b: str
    target_crs: str
    source_resolution_a: tuple[float, float]
    source_resolution_b: tuple[float, float]
    source_transform_a: tuple[float, ...]
    source_transform_b: tuple[float, ...]
    target_resolution: tuple[float, float]
    source_dimensions_a: tuple[int, int]
    source_dimensions_b: tuple[int, int]
    target_dimensions: tuple[int, int]
    source_bounds_a: tuple[float, float, float, float]
    source_bounds_b: tuple[float, float, float, float]
    common_bounds: tuple[float, float, float, float]
    target_transform: tuple[float, ...]
    resampling: str
    nodata: float | int | None
    processing_date: datetime
    notes: list[str] = Field(default_factory=list)


def _resampling(value: str) -> Resampling:
    try:
        return Resampling[value.lower()]
    except KeyError as exc:
        valid = ", ".join(member.name for member in Resampling)
        raise AlignmentError(f"Unsupported resampling '{value}'. Choose one of: {valid}") from exc


def _bounds_intersection(bounds_a: tuple[float, float, float, float], bounds_b: tuple[float, float, float, float]):
    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])
    if left >= right or bottom >= top:
        raise AlignmentError(
            "Images do not overlap after reprojection; their common valid area is empty"
        )
    return left, bottom, right, top


def _target_grid(
    common_bounds: tuple[float, float, float, float],
    resolution: tuple[float, float],
) -> tuple[Affine, int, int, tuple[float, float, float, float]]:
    left, bottom, right, top = common_bounds
    x_resolution, y_resolution = resolution
    width = max(1, int(round((right - left) / x_resolution)))
    height = max(1, int(round((top - bottom) / y_resolution)))
    transform = from_bounds(left, bottom, right, top, width, height)
    snapped_bounds = (left, bottom, right, top)
    return transform, width, height, snapped_bounds


def _source_info(path: Path) -> dict[str, Any]:
    try:
        dataset = rasterio.open(path)
    except (rasterio.errors.RasterioIOError, OSError) as exc:
        raise AlignmentError(f"Unable to open raster '{path}': {exc}") from exc
    with dataset:
        if dataset.crs is None:
            raise AlignmentError(f"Raster '{path}' has no CRS; cannot align safely")
        if dataset.count < 1 or dataset.width < 1 or dataset.height < 1:
            raise AlignmentError(f"Raster '{path}' has invalid dimensions or no bands")
        return {
            "crs": dataset.crs,
            "bounds": (dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top),
            "resolution": dataset.res,
            "transform": tuple(dataset.transform),
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "nodata": dataset.nodata,
            "descriptions": dataset.descriptions,
        }


def align_images(
    image_a: str | Path,
    image_b: str | Path,
    output_a: str | Path,
    output_b: str | Path,
    *,
    target_crs: CRS | str | None = None,
    resolution: float | tuple[float, float] | None = None,
    resampling: str = "nearest",
    nodata: float | int | None = None,
    compression: str = "deflate",
    tiled: bool = False,
    overwrite: bool = False,
) -> AlignmentReport:
    """Align two rasters to one CRS, transform, resolution, bounds, and size.

    The target area is the spatial intersection of both input bounds after
    transforming them into the target CRS. Source CRS and resolution may
    differ. No output is produced when either raster lacks a CRS, has invalid
    dimensions, or has no common area.
    """

    path_a, path_b = Path(image_a), Path(image_b)
    destination_a, destination_b = Path(output_a), Path(output_b)
    if not overwrite and (destination_a.exists() or destination_b.exists()):
        raise AlignmentError("Refusing to overwrite an existing aligned output")
    info_a, info_b = _source_info(path_a), _source_info(path_b)
    target = CRS.from_user_input(target_crs or info_a["crs"])
    target_resolution = resolution
    if target_resolution is None:
        transform_a, width_a, height_a = calculate_default_transform(
            info_a["crs"], target, info_a["width"], info_a["height"], *info_a["bounds"]
        )
        transform_b, width_b, height_b = calculate_default_transform(
            info_b["crs"], target, info_b["width"], info_b["height"], *info_b["bounds"]
        )
        target_resolution = (
            min(abs(transform_a.a), abs(transform_b.a)),
            min(abs(transform_a.e), abs(transform_b.e)),
        )
    elif isinstance(target_resolution, (int, float)):
        if target_resolution <= 0:
            raise AlignmentError("resolution must be positive")
        target_resolution = (float(target_resolution), float(target_resolution))
    else:
        if len(target_resolution) != 2 or min(target_resolution) <= 0:
            raise AlignmentError("resolution must contain two positive values")
        target_resolution = (float(target_resolution[0]), float(target_resolution[1]))

    try:
        bounds_a = transform_bounds(info_a["crs"], target, *info_a["bounds"], densify_pts=21)
        bounds_b = transform_bounds(info_b["crs"], target, *info_b["bounds"], densify_pts=21)
    except (ValueError, rasterio.errors.CRSError) as exc:
        raise AlignmentError(f"Unable to transform source bounds to target CRS {target}: {exc}") from exc
    common_bounds = _bounds_intersection(bounds_a, bounds_b)
    target_transform, target_width, target_height, common_bounds = _target_grid(
        common_bounds, target_resolution
    )
    resampling_method = _resampling(resampling)
    selected_nodata = nodata if nodata is not None else info_a["nodata"]

    for source_path, destination_path, info in (
        (path_a, destination_a, info_a),
        (path_b, destination_b, info_b),
    ):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with rasterio.open(source_path) as source:
                profile = source.profile.copy()
                profile.update(
                    driver="GTiff",
                    width=target_width,
                    height=target_height,
                    crs=target,
                    transform=target_transform,
                    nodata=selected_nodata,
                    compress=compression,
                    tiled=tiled,
                )
                with rasterio.open(destination_path, "w", **profile) as destination:
                    for band_index in range(1, source.count + 1):
                        reproject(
                            source=rasterio.band(source, band_index),
                            destination=rasterio.band(destination, band_index),
                            src_transform=source.transform,
                            src_crs=source.crs,
                            src_nodata=source.nodata,
                            dst_transform=target_transform,
                            dst_crs=target,
                            dst_nodata=selected_nodata,
                            resampling=resampling_method,
                        )
                    destination.descriptions = source.descriptions
        except (rasterio.errors.RasterioIOError, ValueError, OSError) as exc:
            raise AlignmentError(f"Unable to align '{source_path}' to '{destination_path}': {exc}") from exc

    report = AlignmentReport(
        image_a=str(path_a), image_b=str(path_b), output_a=str(destination_a), output_b=str(destination_b),
        source_crs_a=info_a["crs"].to_string(), source_crs_b=info_b["crs"].to_string(),
        target_crs=target.to_string(), source_resolution_a=info_a["resolution"],
        source_resolution_b=info_b["resolution"],
        source_transform_a=info_a["transform"], source_transform_b=info_b["transform"],
        target_resolution=target_resolution,
        source_dimensions_a=(info_a["width"], info_a["height"]),
        source_dimensions_b=(info_b["width"], info_b["height"]),
        target_dimensions=(target_width, target_height),
        source_bounds_a=info_a["bounds"], source_bounds_b=info_b["bounds"],
        common_bounds=common_bounds, target_transform=tuple(target_transform),
        resampling=resampling_method.name, nodata=selected_nodata,
        processing_date=datetime.now(timezone.utc),
        notes=["common spatial intersection used", "both outputs share one pixel grid"],
    )
    LOGGER.info("Aligned %s and %s to %s", path_a, path_b, target)
    return report