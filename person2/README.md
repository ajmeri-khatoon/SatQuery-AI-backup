# Person 2: Satellite preprocessing

This package performs local, credential-free preprocessing of existing GeoTIFF/TIFF files.
It does not download imagery, call Google Earth Engine, or infer a sensor when the file metadata
does not establish one. Install the optional geospatial dependencies first:

```powershell
pip install -e ".\[geo,dev\]"
```

## Inspect and load an image

```python
from person2.preprocessing import inspect_raster, load_image

inspection = inspect_raster("data/example.tif")
print(inspection.as_dict())

image = load_image("data/example.tif")
print(image.data.shape)  # (bands, height, width)
```

Inspection returns width, height, band count, dtype, CRS, affine transform, bounds, resolution,
nodata, and deterministically ordered file metadata. A missing CRS remains `None`.

## Normalize and tile for a model

```python
from person2.preprocessing import SingleImageInput, prepare

prepared = prepare(
	SingleImageInput("data/example.tif"),
	tile_size=(256, 256),
	overlap=32,
)
for tile in prepared.tiles["image"]:
	model_input = tile.data  # float32, channels-first, values in [0, 1]
	print(tile.window, model_input.shape)
```

Normalization is per band, ignores non-finite values and nodata, maps constant valid bands to zero,
and never fabricates missing values. `normalize_image` is also available directly when an array is
already loaded.

## Before/after and optical/SAR pairs

```python
from person2.preprocessing import BeforeAfterInput, OpticalSarInput, prepare

change_input = prepare(BeforeAfterInput("data/before.tif", "data/after.tif"))
fusion_input = prepare(OpticalSarInput("data/optical.tif", "data/sar.tif"), tile_size=512)
print(change_input.kind)  # before_after
print(fusion_input.kind)  # optical_sar
```

Paired rasters must have matching width, height, CRS, transform, bounds, and resolution. A
`SpatialCompatibilityError` names every mismatched field. Pairing does not identify whether a file is
Sentinel-1, Sentinel-2, optical, or SAR; that classification must come from trusted upstream metadata.

## Tests

From the repository root, run the focused suite:

```powershell
python -m pytest person2/tests -q
python -m ruff check person2
python -m mypy person2
```