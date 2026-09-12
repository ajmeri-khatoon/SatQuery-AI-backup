"""Small, credential-free sample-data workflow for Person 4 validation.

The workflow intentionally does not auto-download imagery. It records public
catalogue sources and validates locally supplied GeoTIFF pairs, avoiding large
datasets and credentials in tests or source control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from person4.optical_sar import validate_optical_sar_geotiffs


@dataclass(frozen=True)
class PublicSampleSource:
    name: str
    catalogue_url: str
    data_scope: str
    credentials: str


SENTINEL_SAMPLE_SOURCES = (
    PublicSampleSource(
        name="AWS Sentinel-2 L2A open data",
        catalogue_url="https://registry.opendata.aws/sentinel-2/",
        data_scope="small RGB-compatible optical tiles; choose one registered pair",
        credentials="public catalogue; access method and terms must be checked at retrieval",
    ),
    PublicSampleSource(
        name="AWS Sentinel-1 GRD open data",
        catalogue_url="https://registry.opendata.aws/sentinel-1/",
        data_scope="small VV/VH-compatible SAR tiles matching the optical footprint",
        credentials="public catalogue; access method and terms must be checked at retrieval",
    ),
)


def validate_downloaded_pair(optical: str | Path, sar: str | Path) -> dict[str, object]:
    """Validate a locally downloaded registered Sentinel-2/Sentinel-1 pair."""
    _, _, optical_meta, sar_meta = validate_optical_sar_geotiffs(optical, sar)
    return {
        "optical": optical_meta.__dict__,
        "sar": sar_meta.__dict__,
        "registered": True,
        "note": "BIT accepts only the optical RGB path; SAR is never sent to BIT.",
    }