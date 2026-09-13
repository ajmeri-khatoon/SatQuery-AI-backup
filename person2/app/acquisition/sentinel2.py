"""Google Earth Engine acquisition adapter for Sentinel-2 optical imagery."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field
import rasterio
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry
from pyproj import Transformer

from ..config import AppConfig, get_config
from ..models.request import SatelliteRequest, SatelliteSensor

LOGGER = logging.getLogger(__name__)


class Sentinel2AcquisitionError(RuntimeError):
    """Raised when Earth Engine cannot search or download Sentinel-2 data."""


class Sentinel2Scene(BaseModel):
    """Selected Earth Engine scene information retained beside the raster."""

    model_config = ConfigDict(frozen=True)

    image_id: str
    acquisition_date: date
    cloud_percentage: float | None = Field(default=None, ge=0, le=100)
    collection: str
    bands: list[str]
    source: str = "google-earth-engine"


def _aoi_geojson_wgs84(request: SatelliteRequest) -> dict[str, Any]:
    """Convert the validated AOI to GeoJSON coordinates in EPSG:4326."""

    geometry = request.aoi.shapely_geometry
    if request.aoi.crs.upper() not in {"EPSG:4326", "CRS:84"}:
        converter = Transformer.from_crs(request.aoi.crs, "EPSG:4326", always_xy=True)
        geometry = transform_geometry(converter.transform, geometry)
    return mapping(geometry)


def _normalize_earth_engine_bands(bands: list[str]) -> list[str]:
    """Convert standard padded Sentinel-2 names to Earth Engine names."""

    earth_engine_names = {f"B{number:02d}": f"B{number}" for number in range(1, 13)}
    return [earth_engine_names.get(band, band) for band in bands]


class Sentinel2Acquirer:
    """Search and download Sentinel-2 SR imagery through Earth Engine.

    The Earth Engine module is injectable to make search/download behavior
    unit-testable without credentials or network access.
    """

    collection_id = "COPERNICUS/S2_SR_HARMONIZED"

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
                raise Sentinel2AcquisitionError(
                    "earthengine-api is not installed; install requirements.txt first"
                ) from exc
            self._ee = ee
        if not self._initialized:
            try:
                self._ee.Initialize(project=self.config.google_project_id)
            except Exception as exc:  # Earth Engine exposes several auth exception types.
                raise Sentinel2AcquisitionError(
                    "Earth Engine initialization failed. Run 'earthengine authenticate' "
                    "and configure GOOGLE_PROJECT_ID."
                ) from exc
            self._initialized = True
        return self._ee

    def search(self, request: SatelliteRequest) -> Sentinel2Scene:
        """Select the least-cloudy Sentinel-2 SR scene for a request."""

        if request.sensor is not SatelliteSensor.SENTINEL_2:
            raise Sentinel2AcquisitionError("Sentinel2Acquirer only accepts sentinel-2 requests")
        ee = self._earth_engine()
        aoi = ee.Geometry(_aoi_geojson_wgs84(request))
        end_date = request.end_date or request.start_date
        collection = (
            ee.ImageCollection(self.collection_id)
            .filterBounds(aoi)
            .filterDate(request.start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
        )
        if request.max_cloud_cover is not None:
            collection = collection.filter(
                ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", request.max_cloud_cover)
            )
        collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE")
        try:
            if collection.size().getInfo() == 0:
                raise Sentinel2AcquisitionError(
                    "No Sentinel-2 imagery matched the AOI, date range, and cloud filter"
                )
            image = ee.Image(collection.first())
            image_id = image.id().getInfo()
            if not image_id.startswith(f"{self.collection_id}/"):
                image_id = f"{self.collection_id}/{image_id}"
            acquisition_date = date.fromisoformat(
                image.date().format("YYYY-MM-dd").getInfo()
            )
            cloud_value = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        except Sentinel2AcquisitionError:
            raise
        except Exception as exc:
            raise Sentinel2AcquisitionError(f"Earth Engine scene search failed: {exc}") from exc
        LOGGER.info("Selected Sentinel-2 scene %s for %s", image_id, request.aoi.model_dump())
        return Sentinel2Scene(
            image_id=image_id,
            acquisition_date=acquisition_date,
            cloud_percentage=float(cloud_value) if cloud_value is not None else None,
            collection=self.collection_id,
            bands=request.bands,
        )

    def download(
        self,
        request: SatelliteRequest,
        output_path: str | Path,
        *,
        scene: Sentinel2Scene | None = None,
    ) -> Sentinel2Scene:
        """Download the selected scene as a raw, georeferenced GeoTIFF."""

        selected = scene or self.search(request)
        ee = self._earth_engine()
        aoi = ee.Geometry(_aoi_geojson_wgs84(request))
        try:
            image = ee.Image(selected.image_id).select(
                _normalize_earth_engine_bands(request.bands) or None
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
            if request.bands:
                with rasterio.open(destination, "r+") as downloaded:
                    if downloaded.count != len(request.bands):
                        raise Sentinel2AcquisitionError(
                            "Downloaded Sentinel-2 band count does not match the requested bands"
                        )
                    downloaded.descriptions = tuple(request.bands)
        except Exception as exc:
            if isinstance(exc, Sentinel2AcquisitionError):
                raise
            raise Sentinel2AcquisitionError(
                f"Sentinel-2 download failed for scene '{selected.image_id}': {exc}"
            ) from exc
        LOGGER.info("Downloaded Sentinel-2 scene to %s", output_path)
        return selected