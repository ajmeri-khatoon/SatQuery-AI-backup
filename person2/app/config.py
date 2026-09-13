"""Environment-backed runtime configuration for Person 2."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppConfig(BaseModel):
    """Non-secret settings used by the pipeline foundation."""

    model_config = ConfigDict(frozen=True)

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_root: Path | None = None
    log_level: str = "INFO"
    google_project_id: str | None = None
    sentinel_hub_client_id: str | None = None
    sentinel_hub_client_secret: str | None = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @property
    def resolved_data_root(self) -> Path:
        """Return the configured data directory, defaulting to ``data``."""

        return self.data_root or self.project_root / "data"

    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Create settings from environment variables without loading secrets from code."""

        configured_root = os.getenv("DATA_ROOT")
        return cls(
            data_root=Path(configured_root) if configured_root else None,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            google_project_id=os.getenv("GOOGLE_PROJECT_ID")
            or os.getenv("EARTH_ENGINE_PROJECT_ID"),
            sentinel_hub_client_id=os.getenv("SENTINEL_HUB_CLIENT_ID"),
            sentinel_hub_client_secret=os.getenv("SENTINEL_HUB_CLIENT_SECRET"),
        )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the process-wide immutable configuration."""

    return AppConfig.from_environment()