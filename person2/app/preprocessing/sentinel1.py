"""Local Sentinel-1 SAR preprocessing.

SAR backscatter is deliberately preserved as SAR data. This module does not
apply optical reflectance scaling, cloud masking, or optical normalization.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry

from ..models.aoi import AOI
from ..models.metadata import ImageMetadata, ProcessingConfig
from ..models.sentinel1 import Polarization
from ..metadata.extractor import extract_image_metadata

LOGGER = logging.getLogger(__name__)


class Sentinel1PreprocessingError(RuntimeError):
    """Raised when Sentinel-1 SAR preprocessing fails."""


def _resampling(value: str) -> Resampling:
    try:
        return Resampling[value.lower()]
    except KeyError as exc:
        raise Sentinel1PreprocessingError(f"Unsupported resampling method: {value}") from exc


def _aoi_in_crs(aoi: AOI, target_crs: CRS):
    geometry = aoi.shapely_geometry
    source_crs = CRS.from_user_input(aoi.crs)
    if source_crs != target_crs:
        converter = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geometry = transform_geometry(converter.transform, geometry)
    return geometry


def _polarization_indexes(
    dataset: rasterio.DatasetReader,
    polarizations: Sequence[Polarization],
) -> tuple[list[int], list[str]]:
    descriptions = {
        description.upper(): index
        for index, description in enumerate(dataset.descriptions, 1)
        if description
    }
    indexes: list[int] = []
    names: list[str] = []
    for polarization in polarizations:
        name = polarization.value
        index = descriptions.get(name)
        if index is None:
            if len(polarizations) == 1 and dataset.count == 1:
                index = 1
            else:
                raise Sentinel1PreprocessingError(
                    f"Polarization '{name}' is unavailable; GeoTIFF bands are {dataset.descriptions}"
                )
        indexes.append(index)
        names.append(name)
    return indexes, names


def prepare_sentinel1(
    input_path: str | Path,
    output_path: str | Path,
    *,
    aoi: AOI,
    acquisition_date: date,
    polarizations: Sequence[Polarization] = (Polarization.VV, Polarization.VH),
    processing: ProcessingConfig | None = None,
    satellite: str = "Sentinel-1",
    source: str = "google-earth-engine",
    orbit_pass: str | None = None,
) -> ImageMetadata:
    """Reproject, resample, clip, and write Sentinel-1 SAR backscatter.

    Values remain in the source representation (for example linear GRD
    backscatter). No optical reflectance normalization or implicit dB
    conversion is performed.
    """

    selected_polarizations = list(dict.fromkeys(polarizations))
    if not selected_polarizations:
        raise Sentinel1PreprocessingError("At least one polarization must be selected")
    settings = processing or ProcessingConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    target_crs = CRS.from_user_input(settings.target_crs)
    target_nodata = settings.nodata if settings.nodata is not None else 0
    try:
        with rasterio.open(input_path) as source_dataset:
            if source_dataset.crs is None:
                raise Sentinel1PreprocessingError("Input Sentinel-1 GeoTIFF has no CRS")
            indexes, polarization_names = _polarization_indexes(source_dataset, selected_polarizations)
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
                driver="GTiff", count=len(indexes), width=destination_width,
                height=destination_height, crs=target_crs,
                transform=destination_transform, nodata=target_nodata,
            )
            with rasterio.io.MemoryFile() as memory_file:
                with memory_file.open(**temporary_profile) as temporary:
                    temporary.write(destination)
                    clipped, clipped_transform = mask(
                        temporary, [mapping(geometry)], crop=True,
                        nodata=target_nodata, filled=True,
                    )
    except Sentinel1PreprocessingError:
        raise
    except (rasterio.errors.RasterioIOError, ValueError, OSError) as exc:
        raise Sentinel1PreprocessingError(
            f"Sentinel-1 preprocessing failed for '{input_path}': {exc}"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = {
        "driver": "GTiff", "height": clipped.shape[1], "width": clipped.shape[2],
        "count": clipped.shape[0], "dtype": clipped.dtype, "crs": target_crs,
        "transform": clipped_transform, "nodata": target_nodata,
        "compress": settings.compression, "tiled": settings.tiled,
    }
    try:
        with rasterio.open(output_path, "w", **output_profile) as output:
            output.write(clipped)
            output.descriptions = tuple(polarization_names)
    except (rasterio.errors.RasterioIOError, ValueError, OSError) as exc:
        raise Sentinel1PreprocessingError(
            f"Unable to write processed Sentinel-1 GeoTIFF '{output_path}': {exc}"
        ) from exc

    metadata = extract_image_metadata(
        output_path,
        sensor="sentinel-1-sar",
        satellite=satellite,
        acquisition_date=acquisition_date,
        source=source,
        aoi=aoi.model_dump(mode="json"),
        polarization=polarization_names,
        parent_image=str(input_path),
        preprocessing_steps=[
            "sar_backscatter_preserved", "reprojected",
            f"resampled:{settings.resampling}", "clipped_to_aoi",
            "nodata_handled", "geotiff_written",
        ],
        extra={
            "polarizations": ",".join(polarization_names),
            "orbit_pass": orbit_pass,
            "value_domain": "SAR backscatter; no optical reflectance scaling",
        },
    )
    LOGGER.info("Prepared Sentinel-1 SAR GeoTIFF %s", output_path)
    return metadata