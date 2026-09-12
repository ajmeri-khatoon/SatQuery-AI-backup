"""Validated optical/SAR analysis and feature-level fusion boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import numpy as np
import torch
from torch import nn

from person4.change_detection.provider import (
    RasterSummary,
    RasterValidationError,
    SpatialCompatibilityError,
    _read,
)
from shared.contracts import Specialist, SpecialistResult, SpecialistStatus


@dataclass(frozen=True)
class OpticalSarAnalysis:
    optical_summary: dict[str, float | int]
    sar_summary: dict[str, float | int]
    fused_feature_shape: tuple[int, ...]
    fused_spatial_shape: tuple[int, ...]
    quality: dict[str, float | bool | str]
    evidence_regions: tuple[dict[str, int | float], ...]
    confidence: float
    provenance: dict[str, str]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_specialist_result(self, analysis_id: UUID, step_id: str) -> SpecialistResult:
        return SpecialistResult(
            analysis_id=analysis_id, step_id=step_id, specialist=Specialist.OPTICAL_SAR,
            status=SpecialistStatus.COMPLETED, answer=json.dumps(self.as_dict(), sort_keys=True),
            confidence=self.confidence, confidence_method="untrained_fusion_quality_not_calibrated",
            limitations=list(self.limitations), provenance=self.provenance,
        )


def validate_optical_sar_geotiffs(
    optical: str | Path, sar: str | Path
) -> tuple[np.ndarray, np.ndarray, RasterSummary, RasterSummary]:
    """Validate paired optical and SAR rasters on one spatial grid."""
    optical_data, optical_summary = _read(optical)
    sar_data, sar_summary = _read(sar)
    fields = ("width", "height", "crs", "transform", "bounds")
    mismatches = [
        field for field in fields if getattr(optical_summary, field) != getattr(sar_summary, field)
    ]
    if mismatches:
        raise SpatialCompatibilityError(
            "optical and SAR grids are incompatible: " + ", ".join(mismatches)
        )
    if optical_summary.bands < 1 or sar_summary.bands < 1:
        raise RasterValidationError("optical and SAR inputs must contain at least one band")
    return optical_data, sar_data, optical_summary, sar_summary


def normalize_sentinel2_rgb(
    data: np.ndarray, nodata: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Select B04/B03/B02-style RGB bands and normalize reflectance robustly."""
    if data.ndim != 3 or data.shape[0] < 3:
        raise RasterValidationError("Sentinel-2 RGB preprocessing requires at least 3 bands")
    selected = data[:3].astype(np.float32, copy=False)
    valid = np.isfinite(selected)
    if nodata is not None:
        valid &= selected != nodata
    normalized = np.zeros_like(selected, dtype=np.float32)
    for index in range(3):
        band_valid = valid[index]
        if band_valid.any():
            low, high = np.percentile(selected[index][band_valid], (2, 98))
            if high > low:
                normalized[index, band_valid] = np.clip(
                    (selected[index, band_valid] - low) / (high - low), 0, 1
                )
    return normalized, valid.all(axis=0)


def normalize_sentinel1_sar(
    data: np.ndarray, nodata: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a Sentinel-1 sigma0/dB band using a robust percentile range."""
    if data.ndim != 3 or data.shape[0] < 1:
        raise RasterValidationError("Sentinel-1 preprocessing requires at least one band")
    selected = data[:1].astype(np.float32, copy=False)
    valid = np.isfinite(selected)
    if nodata is not None:
        valid &= selected != nodata
    normalized = np.zeros_like(selected, dtype=np.float32)
    values = selected[0][valid[0]]
    if values.size:
        low, high = np.percentile(values, (2, 98))
        if high > low:
            normalized[0, valid[0]] = np.clip((selected[0, valid[0]] - low) / (high - low), 0, 1)
    return normalized, valid[0]


class FeatureLevelFusion(nn.Module):
    """Learnable modality-specific encoders with global and spatial heads."""

    def __init__(
        self, optical_channels: int, sar_channels: int, feature_channels: int = 16
    ) -> None:
        super().__init__()
        if optical_channels < 1 or sar_channels < 1 or feature_channels < 1:
            raise ValueError("channel counts and feature_channels must be positive")
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(optical_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_channels * 2, feature_channels), nn.ReLU()
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, 1, 1),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        global_features, _ = self.forward_features(optical, sar)
        return global_features

    def forward_features(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if optical.ndim != 4 or sar.ndim != 4 or optical.shape[0] != sar.shape[0]:
            raise ValueError("optical and SAR tensors must be batched NCHW tensors")
        if optical.shape[-2:] != sar.shape[-2:]:
            raise ValueError("optical and SAR tensors must share spatial dimensions")
        optical_features = self.optical_encoder(optical).flatten(1)
        sar_features = self.sar_encoder(sar).flatten(1)
        global_features = self.projection(torch.cat((optical_features, sar_features), dim=1))
        spatial_optical = self.optical_encoder[0](optical)
        spatial_sar = self.sar_encoder[0](sar)
        spatial_logits = self.spatial_head(torch.cat((spatial_optical, spatial_sar), dim=1))
        return global_features, spatial_logits


def _stats(data: np.ndarray) -> dict[str, float | int]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise RasterValidationError("raster contains no finite values")
    return {
        "bands": int(data.shape[0]),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


class BaselineOpticalSarAnalyzer:
    """Descriptive baseline plus an explicit untrained feature-fusion pass."""

    def __init__(self, feature_channels: int = 16) -> None:
        self.feature_channels = feature_channels

    def analyze(self, optical: str | Path, sar: str | Path) -> OpticalSarAnalysis:
        optical_data, sar_data, optical_meta, sar_meta = validate_optical_sar_geotiffs(optical, sar)
        optical_data, optical_valid = normalize_sentinel2_rgb(optical_data, optical_meta.nodata)
        sar_data, sar_valid = normalize_sentinel1_sar(sar_data, sar_meta.nodata)
        fusion = FeatureLevelFusion(3, 1, self.feature_channels).eval()
        with torch.no_grad():
            features, spatial_logits = fusion.forward_features(
                torch.from_numpy(optical_data[None]).float(),
                torch.from_numpy(sar_data[None]).float(),
            )
        spatial_score = torch.sigmoid(spatial_logits)[0, 0].numpy()
        evidence_mask = (spatial_score >= 0.5) & optical_valid & sar_valid
        evidence_regions = _evidence_regions(evidence_mask, spatial_score)
        valid_fraction = float(np.mean(optical_valid & sar_valid))
        confidence = float(min(1.0, 0.5 * valid_fraction + 0.5 * np.mean(
            np.maximum(spatial_score, 1 - spatial_score)
        )))
        return OpticalSarAnalysis(
            optical_summary=_stats(optical_data), sar_summary=_stats(sar_data),
            fused_feature_shape=tuple(int(value) for value in features.shape),
            fused_spatial_shape=tuple(int(value) for value in spatial_logits.shape),
            quality={
                "registered": True,
                "valid_fraction": valid_fraction,
                "optical_sensor": "sentinel-2 RGB B04/B03/B02-style first-three-band contract",
                "sar_sensor": "sentinel-1 first valid band",
            },
            evidence_regions=evidence_regions,
            confidence=confidence,
            provenance={
                "provider": "person4.optical_sar_baseline",
                "fusion": "feature_level_cnn",
                "model_output": "architecture-only",
            },
            limitations=(
                "Feature encoders are untrained; fused features are architectural output only.",
                "No learned optical/SAR interpretation is claimed.",
            ),
        )


def _evidence_regions(mask: np.ndarray, score: np.ndarray) -> tuple[dict[str, int | float], ...]:
    """Return compact 4-connected spatial evidence boxes for the fusion head."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[dict[str, int | float]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                current_y, current_x = stack.pop()
                pixels.append((current_y, current_x))
                for next_y, next_x in ((current_y - 1, current_x), (current_y + 1, current_x),
                                        (current_y, current_x - 1), (current_y, current_x + 1)):
                    if (0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x]
                            and not visited[next_y, next_x]):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
            ys, xs = zip(*pixels)
            regions.append({
                "x_min": min(xs), "y_min": min(ys), "x_max": max(xs) + 1,
                "y_max": max(ys) + 1, "pixel_count": len(pixels),
                "score": float(np.mean([score[py, px] for py, px in pixels])),
            })
    return tuple(sorted(regions, key=lambda region: -int(region["pixel_count"])))


class OpticalSarProvider:
    """Functional untrained fusion provider with explicit model status."""

    def __init__(self, analyzer: BaselineOpticalSarAnalyzer | None = None) -> None:
        self.analyzer = analyzer or BaselineOpticalSarAnalyzer()

    def run(
        self, analysis_id: UUID, step_id: str, optical: str | Path | None = None,
        sar: str | Path | None = None,
    ) -> SpecialistResult:
        if optical is None or sar is None:
            return SpecialistResult(
                analysis_id=analysis_id, step_id=step_id, specialist=Specialist.OPTICAL_SAR,
                status=SpecialistStatus.UNAVAILABLE,
                limitations=["Registered optical and SAR raster paths are required."],
                provenance={"provider": "person4.optical_sar"}, error_code="missing_input",
            )
        try:
            return self.analyzer.analyze(optical, sar).to_specialist_result(analysis_id, step_id)
        except (OSError, RasterValidationError, RuntimeError, ValueError) as error:
            return SpecialistResult(
                analysis_id=analysis_id, step_id=step_id, specialist=Specialist.OPTICAL_SAR,
                status=SpecialistStatus.FAILED, limitations=[str(error)],
                provenance={"provider": "person4.optical_sar"}, error_code="optical_sar_failed",
            )


class LazyLearnedOpticalSarProvider:
    """Lazy trained-model boundary that reports unavailable without a model."""

    def __init__(self, model_factory: Callable[[], nn.Module] | None = None) -> None:
        self._model_factory = model_factory
        self._model: nn.Module | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def run(self, analysis_id: UUID, step_id: str) -> SpecialistResult:
        if self._model_factory is not None and self._model is None:
            self._model = self._model_factory()
        return SpecialistResult(
            analysis_id=analysis_id, step_id=step_id, specialist=Specialist.OPTICAL_SAR,
            status=SpecialistStatus.UNAVAILABLE,
            limitations=["No compatible trained optical/SAR model is configured."],
            provenance={
                "provider": "learned_optical_sar_lazy", "model_loaded": str(self.model_loaded)
            },
            error_code="provider_unavailable",
        )


UnavailableOpticalSarProvider = LazyLearnedOpticalSarProvider