Person 2 tests are self-contained and use synthetic GeoTIFFs for local raster
behavior. External Earth Engine acquisition is mocked; no credentials or
network access are required.

Run the full suite from the `person2` directory:

	python -m pytest -q tests

The suite covers AOI/request validation, GeoTIFF metadata, Sentinel-1 and
Sentinel-2 preprocessing, reprojection, alignment, tiling, before/after pairs,
optical/SAR pairing, centralized metadata extraction, and the high-level
pipeline facade.
