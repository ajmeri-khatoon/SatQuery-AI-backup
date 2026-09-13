"""Google Earth Engine acquisition adapter for Sentinel-1 SAR imagery."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict
from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry

from ..config import AppConfig, get_config
from ..models.sentinel1 import Polarization, Sentinel1Request

LOGGER = logging.getLogger(__name__)


class Sentinel1AcquisitionError(RuntimeError):
    """Raised when Earth Engine cannot search or download Sentinel-1 data."""


class Sentinel1Scene(BaseModel):
    """Selected Sentinel-1 scene details retained with the raw raster."""

    model_config = ConfigDict(frozen=True)

    image_id: str
    acquisition_date: date
    polarizations: list[Polarization]
    collection: str
    orbit_pass: str | None = None
    instrument_mode: str | None = None
    source: str = "google-earth-engine"


def _aoi_geojson_wgs84(request: Sentinel1Request) -> dict[str, Any]:
    geometry = request.aoi.shapely_geometry
    if request.aoi.crs.upper() not in {"EPSG:4326", "CRS:84"}:
        converter = Transformer.from_crs(request.aoi.crs, "EPSG:4326", always_xy=True)
        geometry = transform_geometry(converter.transform, geometry)
    return mapping(geometry)


class Sentinel1Acquirer:
    """Search and download Sentinel-1 GRD data without optical normalization."""

    collection_id = "COPERNICUS/S1_GRD"

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        ee_module: Any | None = None,
        urlopen_function: Any = urlopen,
    ) -> None:
        self.config = config or get_config()
        self._ee = ee_module
        self._urlopen = urlopen_function
        self._initialized = False

    def _earth_engine(self) -> Any:
        if self._ee is None:
            try:
                import ee
            except ImportError as exc:
                raise Sentinel1AcquisitionError(
                    "earthengine-api is not installed; install requirements.txt first"
                ) from exc
            self._ee = ee
        if not self._initialized:
            try:
                self._ee.Initialize(project=self.config.google_project_id)
            except Exception as exc:
                raise Sentinel1AcquisitionError(
                    "Earth Engine initialization failed. Run 'earthengine authenticate' "
                    "and configure GOOGLE_PROJECT_ID."
                ) from exc
            self._initialized = True
        return self._ee

    def search(self, request: Sentinel1Request) -> Sentinel1Scene:
        """Select the first Sentinel-1 GRD scene matching AOI, dates, and polarization."""

        ee = self._earth_engine()
        aoi = ee.Geometry(_aoi_geojson_wgs84(request))
        end_date = request.end_date or request.start_date
        collection = (
            ee.ImageCollection(self.collection_id)
            .filterBounds(aoi)
            .filterDate(request.start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        for polarization in request.polarizations:
            collection = collection.filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", polarization.value)
            )
        try:
            if collection.size().getInfo() == 0:
                raise Sentinel1AcquisitionError(
                    "No Sentinel-1 imagery matched the AOI, date range, and polarization filters"
                )
            image = ee.Image(collection.first())
            image_id = image.id().getInfo()
            acquisition_date = date.fromisoformat(
                image.date().format("YYYY-MM-dd").getInfo()
            )
            orbit_pass = image.get("orbitProperties_pass").getInfo()
            instrument_mode = image.get("instrumentMode").getInfo()
        except Sentinel1AcquisitionError:
            raise
        except Exception as exc:
            raise Sentinel1AcquisitionError(f"Earth Engine Sentinel-1 search failed: {exc}") from exc
        LOGGER.info("Selected Sentinel-1 scene %s for %s", image_id, request.aoi.model_dump())
        return Sentinel1Scene(
            image_id=image_id,
            acquisition_date=acquisition_date,
            polarizations=request.polarizations,
            collection=self.collection_id,
            orbit_pass=orbit_pass,
            instrument_mode=instrument_mode,
        )

    def download(
        self,
        request: Sentinel1Request,
        output_path: str | Path,
        *,
        scene: Sentinel1Scene | None = None,
    ) -> Sentinel1Scene:
        """Download selected Sentinel-1 channels as a raw GeoTIFF."""

        selected = scene or self.search(request)
        ee = self._earth_engine()
        aoi = ee.Geometry(_aoi_geojson_wgs84(request))
        try:
            image = ee.Image(selected.image_id).select(
                [polarization.value for polarization in request.polarizations]
            )
            download_url = image.getDownloadURL(
                {
                    "name": Path(output_path).stem,
                    "region": aoi,
                    "scale": request.resolution_m,
                    "crs": request.aoi.crs,
                    "filePerBand": False,
                    "format": "GEO_TIFF",
                }
            )
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self._urlopen(download_url) as response, destination.open("wb") as stream:
                stream.write(response.read())
        except Exception as exc:
            raise Sentinel1AcquisitionError(
                f"Sentinel-1 download failed for scene '{selected.image_id}': {exc}"
            ) from exc
        LOGGER.info("Downloaded Sentinel-1 SAR scene to %s", output_path)
        return selected