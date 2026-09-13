"""Local Sentinel-2 optical preprocessing."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry
from pyproj import Transformer

from ..models.aoi import AOI
from ..models.metadata import ImageMetadata, ProcessingConfig
from ..metadata.extractor import extract_image_metadata

LOGGER = logging.getLogger(__name__)


class Sentinel2PreprocessingError(RuntimeError):
    """Raised when Sentinel-2 preprocessing cannot produce a valid GeoTIFF."""


def _resampling(value: str) -> Resampling:
    try:
        return Resampling[value.lower()]
    except KeyError as exc:
        valid = ", ".join(member.name for member in Resampling)
        raise Sentinel2PreprocessingError(
            f"Unsupported resampling '{value}'. Choose one of: {valid}"
        ) from exc


def _aoi_in_crs(aoi: AOI, target_crs: CRS):
    geometry = aoi.shapely_geometry
    source_crs = CRS.from_user_input(aoi.crs)
    if source_crs != target_crs:
        converter = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geometry = transform_geometry(converter.transform, geometry)
    return geometry


def _band_indexes(dataset: rasterio.DatasetReader, bands: Sequence[str] | None) -> list[int]:
    if not bands:
        return list(range(1, dataset.count + 1))
    descriptions = {description: index for index, description in enumerate(dataset.descriptions, 1)}
    indexes: list[int] = []
    for band in bands:
        if band.isdigit() and 1 <= int(band) <= dataset.count:
            indexes.append(int(band))
        elif band in descriptions:
            indexes.append(descriptions[band])
        else:
            raise Sentinel2PreprocessingError(
                f"Requested band '{band}' is unavailable; descriptions are {dataset.descriptions}"
            )
    return indexes


def prepare_sentinel2(
    input_path: str | Path,
    output_path: str | Path,
    *,
    aoi: AOI,
    acquisition_date: date,
    processing: ProcessingConfig | None = None,
    bands: Sequence[str] | None = None,
    satellite: str = "Sentinel-2",
    source: str = "google-earth-engine",
) -> ImageMetadata:
    """Reproject, resample, clip, and write Sentinel-2 optical imagery.

    Input values are not treated as SAR data or log-scaled. This function
    preserves optical values, replaces masked pixels with the configured
    nodata value, and records every operation in the returned metadata.
    """

    settings = processing or ProcessingConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    target_crs = CRS.from_user_input(settings.target_crs)
    target_nodata = settings.nodata if settings.nodata is not None else 0
    try:
        with rasterio.open(input_path) as source_dataset:
            if source_dataset.crs is None:
                raise Sentinel2PreprocessingError("Input Sentinel-2 GeoTIFF has no CRS")
            indexes = _band_indexes(source_dataset, bands)
            geometry = _aoi_in_crs(aoi, target_crs)
            destination_transform, destination_width, destination_height = calculate_default_transform(
                source_dataset.crs,
                target_crs,
                source_dataset.width,
                source_dataset.height,
                *source_dataset.bounds,
                resolution=settings.resolution_m,
            )
            destination = np.full(
                (len(indexes), destination_height, destination_width),
                target_nodata,
                dtype=source_dataset.dtypes[0],
            )
            for output_index, source_index in enumerate(indexes):
                reproject(
                    source=rasterio.band(source_dataset, source_index),
                    destination=destination[output_index],
                    src_transform=source_dataset.transform,
                    src_crs=source_dataset.crs,
                    src_nodata=source_dataset.nodata,
                    dst_transform=destination_transform,
                    dst_crs=target_crs,
                    dst_nodata=target_nodata,
                    resampling=_resampling(settings.resampling),
                )
            temporary_profile = source_dataset.profile.copy()
            temporary_profile.update(
                driver="GTiff",
                count=len(indexes),
                width=destination_width,
                height=destination_height,
                crs=target_crs,
                transform=destination_transform,
                nodata=target_nodata,
            )
            with rasterio.io.MemoryFile() as memory_file:
                with memory_file.open(**temporary_profile) as temporary:
                    temporary.write(destination)
                    clipped, clipped_transform = mask(
                        temporary,
                        [mapping(geometry)],
                        crop=True,
                        nodata=target_nodata,
                        filled=True,
                    )
            band_names = [source_dataset.descriptions[index - 1] or f"band_{index}" for index in indexes]
    except Sentinel2PreprocessingError:
        raise
    except (rasterio.errors.RasterioIOError, ValueError, OSError) as exc:
        raise Sentinel2PreprocessingError(
            f"Sentinel-2 preprocessing failed for '{input_path}': {exc}"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = {
        "driver": "GTiff",
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "count": clipped.shape[0],
        "dtype": clipped.dtype,
        "crs": target_crs,
        "transform": clipped_transform,
        "nodata": target_nodata,
        "compress": settings.compression,
        "tiled": settings.tiled,
    }
    try:
        with rasterio.open(output_path, "w", **output_profile) as output:
            output.write(clipped)
            output.descriptions = tuple(band_names)
    except (rasterio.errors.RasterioIOError, ValueError, OSError) as exc:
        raise Sentinel2PreprocessingError(
            f"Unable to write processed Sentinel-2 GeoTIFF '{output_path}': {exc}"
        ) from exc

    metadata = extract_image_metadata(
        output_path,
        sensor="sentinel-2",
        satellite=satellite,
        acquisition_date=acquisition_date,
        source=source,
        aoi=aoi.model_dump(mode="json"),
        parent_image=str(input_path),
        preprocessing_steps=[
            "reprojected", f"resampled:{settings.resampling}",
            "clipped_to_aoi", "nodata_handled", "geotiff_written",
        ],
    )
    LOGGER.info("Prepared Sentinel-2 GeoTIFF %s", output_path)
    return metadata