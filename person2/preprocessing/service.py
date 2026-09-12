"""Credential-free raster inspection and preprocessing for satellite imagery.

The module deliberately accepts files only. Provider and download concerns stay
outside this boundary, so local GeoTIFF/TIFF processing does not require cloud
credentials or a particular satellite provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, cast
from uuid import UUID

from shared.contracts import ImageAsset, Specialist, SpecialistResult, SpecialistStatus

try:
    import numpy as np
    import rasterio  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised when the optional geo extra is absent
    np = None  # type: ignore[assignment]
    rasterio = None  # type: ignore[assignment]


SUPPORTED_SUFFIXES = {".tif", ".tiff"}


class PreprocessingError(ValueError):
    """Raised when a raster cannot be safely inspected or prepared."""


class SpatialCompatibilityError(PreprocessingError):
    """Raised when paired rasters do not describe the same spatial grid."""


def _require_geo_dependencies() -> None:
    if np is None or rasterio is None:
        raise PreprocessingError(
            "rasterio and numpy are required; install the optional dependency group with "
            "pip install -e .[geo]"
        )


def _as_path(source: str | Path) -> Path:
    path = Path(source)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise PreprocessingError("supported inputs must have a .tif or .tiff extension")
    if not path.is_file():
        raise PreprocessingError(f"raster file does not exist: {path}")
    return path


def _metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _metadata(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class RasterInspection:
    width: int
    height: int
    band_count: int
    dtype: str
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    nodata: float | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "dtype": self.dtype,
            "crs": self.crs,
            "transform": self.transform,
            "bounds": self.bounds,
            "resolution": self.resolution,
            "nodata": self.nodata,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RasterImage:
    data: Any
    inspection: RasterInspection


@dataclass(frozen=True)
class RasterTile:
    data: Any
    row: int
    column: int
    window: tuple[int, int, int, int]


@dataclass(frozen=True)
class SingleImageInput:
    image: str | Path


@dataclass(frozen=True)
class BeforeAfterInput:
    before: str | Path
    after: str | Path


@dataclass(frozen=True)
class OpticalSarInput:
    optical: str | Path
    sar: str | Path


@dataclass(frozen=True)
class PreparedInput:
    kind: str
    images: dict[str, RasterImage]
    tiles: dict[str, tuple[RasterTile, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "images": {
                name: {"inspection": image.inspection.as_dict(), "shape": tuple(image.data.shape)}
                for name, image in self.images.items()
            },
            "tiles": {
                name: tuple(
                    {
                        "row": tile.row,
                        "column": tile.column,
                        "window": tile.window,
                        "shape": tuple(tile.data.shape),
                    }
                    for tile in image_tiles
                )
                for name, image_tiles in self.tiles.items()
            },
        }


def inspect_raster(source: str | Path) -> RasterInspection:
    """Inspect a TIFF without inferring a sensor from incomplete metadata."""
    _require_geo_dependencies()
    path = _as_path(source)
    with rasterio.open(path) as dataset:
        transform = cast(
            tuple[float, float, float, float, float, float],
            tuple(float(value) for value in dataset.transform[:6]),
        )
        bounds = cast(
            tuple[float, float, float, float], tuple(float(value) for value in dataset.bounds)
        )
        resolution = (abs(transform[0]), abs(transform[4]))
        nodata = None if dataset.nodata is None else float(dataset.nodata)
        return RasterInspection(
            width=dataset.width,
            height=dataset.height,
            band_count=dataset.count,
            dtype=str(dataset.dtypes[0]),
            crs=None if dataset.crs is None else dataset.crs.to_string(),
            transform=transform,
            bounds=bounds,
            resolution=resolution,
            nodata=nodata,
            metadata=_metadata(dataset.tags()),
        )


def validate_raster(source: str | Path) -> RasterInspection:
    """Validate a local satellite-compatible raster and return its inspection."""
    inspection = inspect_raster(source)
    if inspection.band_count < 1:
        raise PreprocessingError("raster must contain at least one band")
    return inspection


def load_image(source: str | Path) -> RasterImage:
    """Load a raster as a channels-first NumPy array with its inspection."""
    _require_geo_dependencies()
    path = _as_path(source)
    with rasterio.open(path) as dataset:
        data = dataset.read()
    return RasterImage(data=data, inspection=inspect_raster(path))


def normalize_image(data: Any, nodata: float | None = None) -> Any:
    """Convert channels-first data to finite float32 values in [0, 1]."""
    _require_geo_dependencies()
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise PreprocessingError(
            "image data must have shape (height, width) or (bands, height, width)"
        )
    channels = array[None, ...] if array.ndim == 2 else array
    result = np.zeros(channels.shape, dtype=np.float32)
    for band_index, band in enumerate(channels):
        valid = np.isfinite(band)
        if nodata is not None:
            valid &= band != nodata
        if not np.any(valid):
            continue
        minimum = float(np.min(band[valid]))
        maximum = float(np.max(band[valid]))
        if maximum > minimum:
            result[band_index, valid] = (band[valid] - minimum) / (maximum - minimum)
    return result if array.ndim == 3 else result[0]


def _starts(length: int, tile_length: int, stride: int) -> list[int]:
    values = list(range(0, max(length - tile_length, 0) + 1, stride))
    last = max(length - tile_length, 0)
    if not values or values[-1] != last:
        values.append(last)
    return values


def tile_image(
    data: Any, tile_size: int | tuple[int, int], overlap: int | float = 0
) -> tuple[RasterTile, ...]:
    """Crop channels-first data, including edge tiles exactly once."""
    _require_geo_dependencies()
    array = np.asarray(data)
    if array.ndim != 3:
        raise PreprocessingError(
            "tiling requires channels-first data with shape (bands, height, width)"
        )
    tile_height, tile_width = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
    if tile_height <= 0 or tile_width <= 0:
        raise PreprocessingError("tile dimensions must be positive")
    overlap_height, overlap_width = (
        (overlap, overlap) if isinstance(overlap, (int, float)) else overlap
    )
    if isinstance(overlap, float):
        overlap_height = round(tile_height * overlap_height)
        overlap_width = round(tile_width * overlap_width)
    if not (0 <= overlap_height < tile_height and 0 <= overlap_width < tile_width):
        raise PreprocessingError(
            "overlap must be non-negative and smaller than each tile dimension"
        )
    height, width = array.shape[1:]
    tiles = []
    for row, top in enumerate(_starts(height, tile_height, tile_height - int(overlap_height))):
        for column, left in enumerate(_starts(width, tile_width, tile_width - int(overlap_width))):
            bottom = min(top + tile_height, height)
            right = min(left + tile_width, width)
            tiles.append(
                RasterTile(
                    array[:, top:bottom, left:right], row, column, (left, top, right, bottom)
                )
            )
    return tuple(tiles)


def _check_compatible(images: Sequence[RasterImage]) -> None:
    first = images[0].inspection
    for index, image in enumerate(images[1:], start=2):
        current = image.inspection
        mismatches = []
        for field in ("width", "height", "crs", "transform", "bounds", "resolution"):
            if getattr(first, field) != getattr(current, field):
                mismatches.append(field)
        if mismatches:
            raise SpatialCompatibilityError(
                f"paired raster {index} is spatially incompatible; mismatched fields: "
                + ", ".join(mismatches)
            )


def prepare(
    inputs: SingleImageInput | BeforeAfterInput | OpticalSarInput,
    tile_size: int | tuple[int, int] | None = None,
    overlap: int | float = 0,
) -> PreparedInput:
    """Load one supported input structure and optionally generate deterministic tiles."""
    if isinstance(inputs, SingleImageInput):
        sources = {"image": inputs.image}
        kind = "single"
    elif isinstance(inputs, BeforeAfterInput):
        sources = {"before": inputs.before, "after": inputs.after}
        kind = "before_after"
    elif isinstance(inputs, OpticalSarInput):
        sources = {"optical": inputs.optical, "sar": inputs.sar}
        kind = "optical_sar"
    else:
        raise PreprocessingError(
            "inputs must be SingleImageInput, BeforeAfterInput, or OpticalSarInput"
        )
    images = {name: load_image(source) for name, source in sources.items()}
    if len(images) > 1:
        _check_compatible(tuple(images.values()))
    tiles = {
        name: tile_image(normalize_image(image.data, image.inspection.nodata), tile_size, overlap)
        if tile_size is not None
        else tuple()
        for name, image in images.items()
    }
    return PreparedInput(kind=kind, images=images, tiles=tiles)


class PreprocessingProvider(Protocol):
    def validate(self, asset: ImageAsset, analysis_id: str, step_id: str) -> SpecialistResult: ...


class RasterPreprocessingProvider:
    """Adapter that validates an existing ImageAsset against a local storage root."""

    def __init__(self, storage_root: str | Path) -> None:
        self.storage_root = Path(storage_root).resolve()

    def validate(self, asset: ImageAsset, analysis_id: str, step_id: str) -> SpecialistResult:
        path = (self.storage_root / asset.storage_key).resolve()
        if self.storage_root not in path.parents:
            raise PreprocessingError("asset storage key escapes the configured storage root")
        inspection = validate_raster(path)
        return SpecialistResult(
            analysis_id=UUID(analysis_id),
            step_id=step_id,
            specialist=Specialist.PREPROCESSING,
            status=SpecialistStatus.COMPLETED,
            answer="Raster input validated and ready for preprocessing.",
            limitations=["Sensor identity was not inferred by preprocessing."],
            provenance={
                "component": "person2.preprocessing",
                "crs": inspection.crs or "unreferenced",
            },
        )


class UnavailablePreprocessingProvider:
    def validate(self, asset: ImageAsset, analysis_id: str, step_id: str) -> SpecialistResult:
        raise NotImplementedError(
            "Install the optional geo dependencies to use raster preprocessing"
        )
