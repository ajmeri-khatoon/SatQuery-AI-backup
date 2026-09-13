# Person 2: Satellite ingestion and preprocessing

Person 2 is the satellite data subsystem for local GeoTIFF/TIFF inspection, Earth Engine acquisition, metadata, CRS-aware preprocessing, pairing, alignment, and deterministic GeoTIFF tiling. P4 owns optical/SAR analysis and fusion; this package only prepares and reports compatible inputs.

## Public pipeline

The high-level interface is:

```python
from app.models import SatellitePipelineRequest
from app.pipeline import process_satellite_request

result = process_satellite_request(
    SatellitePipelineRequest(
        workflow="single-sentinel-2",
        aoi={"bbox": [73.80, 18.50, 73.81, 18.51]},
        start_date="2026-01-01",
        bands=["B04", "B03", "B02", "B08"],
        target_crs="EPSG:32643",
        source_path="data/local_scene.tif",
        tile=True,
    )
)
```

`PipelineResult` reports status, processed/aligned paths, metadata paths, tile paths, processing information, warnings, and errors. The complete input and output contract is documented in [integration_contract.md](integration_contract.md).

Supported workflows:

- single Sentinel-1 or Sentinel-2;
- before/after Sentinel-1 or Sentinel-2 with common-grid alignment;
- optical/SAR pairing with acquisition-date and AOI-coverage checks; and
- explicit failure for the reserved before/after optical/SAR workflow.

## Capabilities

The `app/` subsystem contains Earth Engine Sentinel-1/Sentinel-2 search and download adapters, AOI validation and CRS conversion, metadata extraction, Sentinel-specific preprocessing, reprojection, clipping, resampling, nodata handling, before/after and optical/SAR pairing, alignment, and GeoTIFF tiling.

Local source paths are supported for offline workflows. The app pipeline reuses the preserved `person2.preprocessing.service.validate_raster` boundary before delegating to the richer sensor-specific processors, so local file inspection does not create a second validation contract.

Earth Engine access is explicit. If `earthengine-api` is absent, authentication is missing, `GOOGLE_PROJECT_ID` is unset, or no matching scene exists, the acquisition layer raises a truthful typed acquisition error and the high-level pipeline returns `status="failed"`; it never fabricates imagery or metadata.

Install the teammate subsystem requirements with:

```powershell
pip install -r person2/requirements.txt
```

Configure non-secret settings through environment variables such as `GOOGLE_PROJECT_ID`, `DATA_ROOT`, and `LOG_LEVEL`. Keep credentials outside the repository; `.env.example` documents names only.

## Preserved service boundary

`person2/preprocessing/service.py` remains the lightweight credential-free API for `inspect_raster`, `load_image`, `prepare`, normalization, and in-memory tiling. The app pipeline reuses its local validation function while the app modules provide CRS-aware output files and acquisition provenance.

## Tests and checks

From the repository root:

```powershell
python -m pytest person2
python -m ruff check person2/app/pipeline/pipeline.py person2/tests/conftest.py
```

The full Person 2 suite covers local inspection, preprocessing, CRS safety, Sentinel-1/Sentinel-2 acquisition with injected Earth Engine doubles, metadata, pairing, alignment, tiling, pipeline outputs, and failure paths. The test conftest only exposes the target `person2/app` package under its teammate contractual import name `app`; it does not alter production behavior.
