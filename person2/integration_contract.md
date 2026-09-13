# Person 2 Integration Contract

This document is the stable integration boundary for the satellite acquisition and preprocessing component. Consumers should use the public request/result models and `process_satellite_request`; internal acquisition and preprocessing modules may change independently.

## Import

Run from the `person2` directory:

```python
from app.models import PipelineResult, PipelineWorkflow, SatellitePipelineRequest
from app.pipeline import process_satellite_request
```

The primary interface is:

```python
process_satellite_request(
    request: SatellitePipelineRequest | dict
) -> PipelineResult
```

The function validates the request, acquires imagery when source paths are not supplied, preprocesses it, aligns images when the workflow requires it, optionally creates tiles, writes metadata, and returns a structured result.

No FastAPI layer is included. The Python interface is the canonical contract and avoids adding an HTTP dependency. An API adapter can call this function later without coupling callers to internal modules.

## Input Contract

### Common fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `workflow` | `PipelineWorkflow` or string | yes | Workflow listed below. |
| `aoi` | `AOI` or object | yes | GeoJSON geometry, bounding box, or latitude/longitude point. CRS defaults to `EPSG:4326`. |
| `resolution_m` | positive number | no | Requested output resolution. Default `10`. |
| `resampling` | string | no | Rasterio resampling method, for example `nearest` or `bilinear`. Default `nearest`. |
| `nodata` | number/null | no | Explicit output NoData value. If omitted, the sensor preprocessing default is used. |
| `target_crs` | string | conditional | Output CRS, for example `EPSG:32643`. A projected CRS is required for metre-scale `resolution_m`; geographic CRS values such as `EPSG:4326` are only accepted for explicitly degree-scale resolutions below `1`. |
| `source` | string | no | Source identifier. Default `earth-engine`. |
| `output_dir` | path/string | no | Output root. Defaults to configured `DATA_ROOT/processed/<workflow>`. |
| `tile` | boolean | no | Whether to generate tiles. Default `false`. |
| `tile_width` | positive integer | no | Tile width in pixels. Default `256`. |
| `tile_height` | positive integer | no | Tile height in pixels. Default `256`. |
| `overlap` | non-negative integer | no | Tile overlap in pixels. Must be smaller than both tile dimensions. |
| `padding` | boolean | no | Pad edge tiles to the configured dimensions. Default `false`. |

### AOI forms

```json
{"bbox": [73.80, 18.50, 73.81, 18.51]}
```

```json
{
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[73.80, 18.50], [73.81, 18.50], [73.81, 18.51], [73.80, 18.51], [73.80, 18.50]]]
  },
  "crs": "EPSG:4326"
}
```

```json
{"latitude": 18.505, "longitude": 73.805, "crs": "EPSG:4326"}
```

### Workflow-specific fields

| Workflow | Required date fields | Band/polarization fields | Optional local inputs |
| --- | --- | --- | --- |
| `single-sentinel-2` | `start_date`, optional `end_date` | `bands`, for example `B04`, `B03`, `B02`, `B08` | `source_path` |
| `single-sentinel-1` | `start_date`, optional `end_date` | `polarizations`, values `VV` and/or `VH` | `source_path` |
| `before-after-sentinel-2` | `before_start_date`, `after_start_date`, optional end dates | `bands` | `before_source` and `after_source` together |
| `before-after-sentinel-1` | `before_start_date`, `after_start_date`, optional end dates | `polarizations` | `before_source` and `after_source` together |
| `optical-sar` | `optical_start_date`, `sar_start_date`, optional end dates | `optical_bands`, `polarizations` | `optical_source` and `sar_source` together |
| `before-after-optical-sar` | reserved | reserved | Not implemented; returns a failed result explicitly. |

For optical/SAR pairing, `max_date_difference_days` controls the maximum absolute difference between the actual selected Sentinel-2 and Sentinel-1 acquisition dates. The timestamps do not need to be equal.

Date ranges are validated. Before/after workflows require the after period to start after the before period. A single local source path is not accepted for a two-image workflow; both paths must be supplied. For typical Sentinel-2/Sentinel-1 requests using `resolution_m=10`, provide a projected target CRS such as the appropriate UTM zone; the pipeline rejects metre values in geographic degrees rather than silently producing an incorrect grid.

## Example Requests

### Single Sentinel-2 with tiles

```python
request = SatellitePipelineRequest(
    workflow=PipelineWorkflow.SINGLE_SENTINEL_2,
    aoi={"bbox": [73.80, 18.50, 73.81, 18.51]},
    start_date="2026-01-01",
    end_date="2026-01-31",
    bands=["B04", "B03", "B02", "B08"],
    max_cloud_cover=20,
    resolution_m=10,
    target_crs="EPSG:32643",
    tile=True,
    tile_width=256,
    tile_height=256,
    overlap=32,
)
result = process_satellite_request(request)
```

Equivalent dictionary input is supported:

```python
result = process_satellite_request({
    "workflow": "single-sentinel-2",
    "aoi": {"bbox": [73.80, 18.50, 73.81, 18.51]},
    "start_date": "2026-01-01",
    "end_date": "2026-01-31",
    "bands": ["B04", "B03", "B02", "B08"],
    "resolution_m": 10,
    "target_crs": "EPSG:32643",
})
```

### Before/after Sentinel-2

```python
request = SatellitePipelineRequest(
    workflow="before-after-sentinel-2",
    aoi={"bbox": [73.80, 18.50, 73.81, 18.51]},
    before_start_date="2026-01-01",
    before_end_date="2026-01-31",
    after_start_date="2026-03-01",
    after_end_date="2026-03-31",
    bands=["B04", "B03", "B02", "B08"],
    resolution_m=10,
    target_crs="EPSG:32643",
    tile=True,
    tile_width=256,
    tile_height=256,
    overlap=32,
)
result = process_satellite_request(request)
```

### Optical + SAR

```python
request = SatellitePipelineRequest(
    workflow="optical-sar",
    aoi={"bbox": [73.80, 18.50, 73.81, 18.51]},
    optical_start_date="2026-01-01",
    optical_end_date="2026-01-31",
    sar_start_date="2026-01-01",
    sar_end_date="2026-01-31",
    max_date_difference_days=5,
    optical_bands=["B04", "B03", "B02", "B08"],
    polarizations=["VV", "VH"],
    resolution_m=10,
    target_crs="EPSG:32643",
)
result = process_satellite_request(request)
```

For tests and offline reprocessing, provide `source_path`, or both `before_source` and `after_source`, or both `optical_source` and `sar_source`. These paths must point to real georeferenced GeoTIFFs; the pipeline does not fabricate imagery.

## Output Contract

`PipelineResult` is a Pydantic model with this stable shape:

```python
class PipelineResult:
    status: str
    workflow: str
    output_paths: dict[str, str]
    metadata_paths: list[str]
    tile_paths: dict[str, list[str]]
    processing_information: dict
    warnings: list[str]
    errors: list[str]
```

### Successful response example

```json
{
  "status": "success",
  "workflow": "single-sentinel-2",
  "output_paths": {
    "processed": "data/processed/single-sentinel-2/processed/sentinel2.tif"
  },
  "metadata_paths": [
    "data/processed/single-sentinel-2/metadata/image_metadata.json",
    "data/processed/single-sentinel-2/tiles/tiles.json"
  ],
  "tile_paths": {
    "image": [
      "data/processed/single-sentinel-2/tiles/tile_0001.tif",
      "data/processed/single-sentinel-2/tiles/tile_0002.tif"
    ]
  },
  "processing_information": {
    "sensor": "sentinel-2",
    "acquisition_date": "2026-01-15",
    "preprocessing": [
      "reprojected",
      "resampled:nearest",
      "clipped_to_aoi",
      "nodata_handled",
      "geotiff_written"
    ],
    "tile_count": 2
  },
  "warnings": [],
  "errors": []
}
```

### Before/after response paths

Before/after workflows return:

```text
output_paths["before_aligned"]
output_paths["after_aligned"]
metadata_paths[0]                  # pair_metadata.json
 tile_paths["before"]
 tile_paths["after"]
```

The pair metadata JSON contains before and after metadata, common CRS, common bounds, common resolution, aligned paths, tile dimensions, overlap, and preprocessing steps. Matching tile IDs in the two tile lists are suitable for pixel-level comparison.

### Optical/SAR response paths

Optical/SAR workflows return:

```text
output_paths["optical_aligned"]
output_paths["sar_aligned"]
metadata_paths[0]                  # optical_sar_pair_report.json
```

The report records Sentinel-1 and Sentinel-2 acquisition dates, date difference, AOI coverage flags, selected bands/polarizations, output paths, common CRS, bounds, resolution, dimensions, transform, and preprocessing steps.

## Metadata Contract

Every processed image and tile has JSON-serializable Pydantic metadata generated by the centralized `app.metadata.extractor` module.

Image metadata includes:

```text
sensor, satellite, acquisition_date, processing_date, source
crs, epsg, resolution_m, resolution, width, height
bounds, transform, bands, polarization, dtype, nodata
aoi, tile_id, parent_image, preprocessing_steps, extra
```

Tile metadata additionally includes `tile_id`, `source_image`, `output_path`, `row`, and `column`. EPSG is populated when the CRS can provide one. CRS, bounds, transform, dimensions, resolution, band descriptions, dtype, and nodata are read from the actual output GeoTIFF.

Serialization:

```python
json_text = metadata.model_dump_json(indent=2)
metadata_dict = metadata.model_dump(mode="json")
```

## Responsibilities By Consumer

### Person 1: orchestrator

- Build a `SatellitePipelineRequest`.
- Call `process_satellite_request`.
- Check `result.status` and `result.errors`.
- Pass output and metadata paths to the selected downstream component.
- Do not depend on Earth Engine, Rasterio, or internal adapter classes.

### Person 3: VQA/vision

- Consume `output_paths["processed"]` for a single image.
- Use `tile_paths["image"]` when tiled inference is requested.
- Read the referenced metadata JSON before inference.
- Preserve the reported CRS and bounds when returning spatial evidence.

### Person 4: change detection and optical/SAR

- Consume `before_aligned` and `after_aligned` paths for temporal change detection.
- Consume `optical_aligned` and `sar_aligned` for optical/SAR analysis.
- For temporal tiles, compare equal tile IDs from `tile_paths["before"]` and `tile_paths["after"]`.
- Verify common CRS, resolution, bounds, dimensions, and transform from the pair report before pixel operations.
- Do not treat Sentinel-1 SAR values as optical reflectance.

## Error Semantics

The facade returns `status="failed"` with a non-empty `errors` list for request validation, authentication, unavailable imagery, missing files, invalid GeoTIFFs, AOI coverage failures, incompatible grids, and unsupported workflows. A failed result must not be passed to a downstream model as usable imagery.

Lower-level functions remain available when a caller needs direct exception handling:

- `Sentinel2Acquirer` / `prepare_sentinel2`
- `Sentinel1Acquirer` / `prepare_sentinel1`
- `align_images`
- `create_tiles`
- `process_before_after`
- `process_optical_sar`
- `extract_image_metadata` / `extract_tile_metadata`

## Environment and Authentication

Set credentials and configuration through environment variables. No credentials are accepted in the request model and none are hard-coded:

```text
GOOGLE_PROJECT_ID
EARTH_ENGINE_PROJECT_ID
SENTINEL_HUB_CLIENT_ID
SENTINEL_HUB_CLIENT_SECRET
DATA_ROOT
LOG_LEVEL
```

For Earth Engine, authenticate with `earthengine authenticate` and set `GOOGLE_PROJECT_ID` before using workflows without local source paths. Unit tests and offline integrations should provide local GeoTIFF source paths.
