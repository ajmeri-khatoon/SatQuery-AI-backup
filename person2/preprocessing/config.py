"""Configuration and filesystem layout for the satellite pipeline.

This module only defines configuration; data retrieval and preprocessing are
implemented in later pipeline phases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PIPELINE_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
TILES_ROOT = DATA_ROOT / "tiles"


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration shared by pipeline components."""

    pipeline_root: Path = PIPELINE_ROOT
    data_root: Path = DATA_ROOT
    raw_root: Path = RAW_ROOT
    processed_root: Path = PROCESSED_ROOT
    tiles_root: Path = TILES_ROOT
    earth_engine_project: str | None = os.getenv("GOOGLE_PROJECT_ID")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def from_environment(cls) -> "PipelineConfig":
        """Build configuration from environment variables without reading secrets."""

        return cls(
            earth_engine_project=(
                os.getenv("GOOGLE_PROJECT_ID")
                or os.getenv("EARTH_ENGINE_PROJECT_ID")
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


DEFAULT_CONFIG = PipelineConfig.from_environment()
