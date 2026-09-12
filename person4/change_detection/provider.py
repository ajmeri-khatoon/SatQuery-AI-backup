"""Bi-temporal GeoTIFF validation and deterministic change detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import numpy as np
import rasterio  # type: ignore[import-untyped]
import torch
from torch.nn import functional

from person4.models import BITLevirModel, SiameseChangeModel
from shared.contracts import Specialist, SpecialistResult, SpecialistStatus


class RasterValidationError(ValueError):
    """Raised when a GeoTIFF cannot be used for analysis."""


class SpatialCompatibilityError(RasterValidationError):
    """Raised when paired rasters do not share the same spatial grid."""


@dataclass(frozen=True)
class RasterSummary:
    path: str
    width: int
    height: int
    bands: int
    crs: str
    transform: tuple[float, ...]
    bounds: tuple[float, ...]
    nodata: float | None


@dataclass(frozen=True)
class ChangeRegion:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    pixel_count: int
    change_score: float

    def as_dict(self) -> dict[str, int | float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ChangeDetectionResult:
    changed: bool
    change_mask: np.ndarray
    valid_mask: np.ndarray
    changed_regions: tuple[ChangeRegion, ...]
    change_percentage: float
    confidence: float
    before: RasterSummary
    after: RasterSummary
    provenance: dict[str, str]
    limitations: tuple[str, ...]
    confidence_method: str = "deterministic_threshold_baseline"

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "change_mask": self.change_mask.astype(np.uint8).tolist(),
            "changed_regions": [region.as_dict() for region in self.changed_regions],
            "change_percentage": self.change_percentage,
            "confidence": self.confidence,
            "before": self.before.__dict__,
            "after": self.after.__dict__,
            "provenance": self.provenance,
            "limitations": self.limitations,
        }

    def to_specialist_result(self, analysis_id: UUID, step_id: str) -> SpecialistResult:
        answer = {
            "changed": self.changed,
            "change_percentage": self.change_percentage,
            "changed_regions": [region.as_dict() for region in self.changed_regions],
        }
        return SpecialistResult(
            analysis_id=analysis_id,
            step_id=step_id,
            specialist=Specialist.CHANGE_DETECTION,
            status=SpecialistStatus.COMPLETED,
            answer=json.dumps(answer, sort_keys=True),
            confidence=self.confidence,
            confidence_method=self.confidence_method,
            limitations=list(self.limitations),
            provenance=self.provenance,
        )


def _read(path_value: str | Path) -> tuple[np.ndarray, RasterSummary]:
    path = Path(path_value)
    if path.suffix.lower() not in {".tif", ".tiff"} or not path.is_file():
        raise RasterValidationError(f"expected an existing .tif or .tiff file: {path}")
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise RasterValidationError(f"{path} must have a CRS")
        summary = RasterSummary(
            path=str(path),
            width=dataset.width,
            height=dataset.height,
            bands=dataset.count,
            crs=dataset.crs.to_string(),
            transform=tuple(float(value) for value in dataset.transform[:6]),
            bounds=tuple(float(value) for value in dataset.bounds),
            nodata=None if dataset.nodata is None else float(dataset.nodata),
        )
        data = dataset.read().astype(np.float32)
    if not np.isfinite(data).any():
        raise RasterValidationError(f"{path} contains no finite pixels")
    return data, summary


def validate_bitemporal_geotiffs(
    before: str | Path, after: str | Path
) -> tuple[np.ndarray, np.ndarray, RasterSummary, RasterSummary]:
    """Read two GeoTIFFs and enforce identical spatial grids and band counts."""
    before_data, before_summary = _read(before)
    after_data, after_summary = _read(after)
    fields = ("width", "height", "crs", "transform", "bounds")
    mismatches = [
        field
        for field in fields
        if getattr(before_summary, field) != getattr(after_summary, field)
    ]
    if mismatches:
        raise SpatialCompatibilityError(
            "rasters are spatially incompatible: " + ", ".join(mismatches)
        )
    if before_summary.bands != after_summary.bands:
        raise RasterValidationError("before and after rasters must have the same band count")
    return before_data, after_data, before_summary, after_summary


def _scale(data: np.ndarray, nodata: float | None) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    scaled = np.zeros_like(data, dtype=np.float32)
    for band_index in range(data.shape[0]):
        band_valid = valid[band_index]
        if band_valid.any():
            values = data[band_index]
            low, high = np.percentile(values[band_valid], (2, 98))
            if high > low:
                scaled[band_index, band_valid] = np.clip(
                    (values[band_valid] - low) / (high - low), 0, 1
                )
    return scaled, valid.all(axis=0)


def _regions(mask: np.ndarray, score: np.ndarray, minimum_pixels: int) -> tuple[ChangeRegion, ...]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[ChangeRegion] = []
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
            if len(pixels) >= minimum_pixels:
                ys, xs = zip(*pixels)
                regions.append(ChangeRegion(
                    min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(pixels),
                    float(np.mean([score[y, x] for y, x in pixels])),
                ))
    return tuple(
        sorted(regions, key=lambda region: (-region.pixel_count, region.y_min, region.x_min))
    )


class DeterministicBaselineChangeDetector:
    """A reproducible per-pixel spectral-difference baseline."""

    def __init__(self, threshold: float = 0.2, minimum_region_pixels: int = 1) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if minimum_region_pixels < 1:
            raise ValueError("minimum_region_pixels must be positive")
        self.threshold = threshold
        self.minimum_region_pixels = minimum_region_pixels

    def detect(self, before: str | Path, after: str | Path) -> ChangeDetectionResult:
        before_data, after_data, before_summary, after_summary = validate_bitemporal_geotiffs(
            before, after
        )
        before_scaled, before_valid = _scale(before_data, before_summary.nodata)
        after_scaled, after_valid = _scale(after_data, after_summary.nodata)
        valid = before_valid & after_valid
        score = np.mean(np.abs(after_scaled - before_scaled), axis=0)
        mask = (score >= self.threshold) & valid
        regions = _regions(mask, score, self.minimum_region_pixels)
        valid_count = int(valid.sum())
        percentage = 0.0 if valid_count == 0 else float(mask.sum() / valid_count * 100)
        confidence = 0.0 if valid_count == 0 else float(min(1.0, 0.5 + abs(percentage - 50) / 100))
        return ChangeDetectionResult(
            changed=bool(mask.any()), change_mask=mask, valid_mask=valid,
            changed_regions=regions, change_percentage=percentage, confidence=confidence,
            before=before_summary, after=after_summary,
            provenance={
                "provider": "person4.deterministic_baseline",
                "algorithm": "spectral_l1",
                "model_output": "fallback-generated",
            },
            limitations=(
                "This is a deterministic threshold baseline, not learned change detection.",
            ),
        )


@dataclass(frozen=True)
class LearnedChangeConfig:
    """Explicit local model configuration; no checkpoint is downloaded."""

    input_channels: int
    base_channels: int = 16
    threshold: float = 0.5
    device: str = "cpu"
    checkpoint_path: str | Path | None = None
    allow_untrained: bool = False

    def __post_init__(self) -> None:
        if self.input_channels < 1 or self.base_channels < 1:
            raise ValueError("input_channels and base_channels must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if not self.device.strip():
            raise ValueError("device cannot be blank")


class LearnedChangeDetector:
    """Lazy Siamese model inference for registered bi-temporal GeoTIFFs."""

    def __init__(
        self,
        config: LearnedChangeConfig,
        model: SiameseChangeModel | None = None,
    ) -> None:
        self.config = config
        self._model = model
        self._checkpoint_loaded = False

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and (self._checkpoint_loaded or self.config.allow_untrained)

    @property
    def available(self) -> bool:
        checkpoint_exists = self.config.checkpoint_path is not None and Path(
            self.config.checkpoint_path
        ).is_file()
        return self.config.allow_untrained or checkpoint_exists

    def _ensure_model(self) -> SiameseChangeModel:
        if self._model is None:
            self._model = SiameseChangeModel(
                self.config.input_channels, self.config.base_channels
            )
        if self.config.checkpoint_path is not None and not self._checkpoint_loaded:
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=self.config.device, weights_only=True
            )
            state_dict = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            self._model.load_state_dict(state_dict)
            self._checkpoint_loaded = True
        self._model.to(self.config.device)
        self._model.eval()
        return self._model

    def detect(self, before: str | Path, after: str | Path) -> ChangeDetectionResult:
        before_data, after_data, before_summary, after_summary = validate_bitemporal_geotiffs(
            before, after
        )
        if before_data.shape[0] != self.config.input_channels:
            raise RasterValidationError(
                f"learned model expects {self.config.input_channels} bands, "
                f"received {before_data.shape[0]}"
            )
        before_scaled, before_valid = _scale(before_data, before_summary.nodata)
        after_scaled, after_valid = _scale(after_data, after_summary.nodata)
        valid = before_valid & after_valid
        model = self._ensure_model()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(before_scaled[None]).to(self.config.device),
                torch.from_numpy(after_scaled[None]).to(self.config.device),
            )
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (probabilities >= self.config.threshold) & valid
        regions = _regions(mask, probabilities, minimum_pixels=1)
        valid_count = int(valid.sum())
        percentage = 0.0 if valid_count == 0 else float(mask.sum() / valid_count * 100)
        confidence = 0.0 if valid_count == 0 else float(
            np.mean(np.maximum(probabilities[valid], 1 - probabilities[valid]))
        )
        checkpoint = (
            "untrained"
            if self.config.checkpoint_path is None
            else str(self.config.checkpoint_path)
        )
        limitations = (
            ("Model weights are randomly initialized; this is an architecture execution test.",)
            if self.config.checkpoint_path is None
            else ("Checkpoint provenance is local and confidence is not calibrated.",)
        )
        return ChangeDetectionResult(
            changed=bool(mask.any()), change_mask=mask, valid_mask=valid,
            changed_regions=regions, change_percentage=percentage, confidence=confidence,
            before=before_summary, after=after_summary,
            provenance={
                "provider": "person4.siamese_change",
                "architecture": "shared_encoder_difference_decoder",
                "checkpoint": checkpoint,
                "device": self.config.device,
                "model_output": "model-generated",
            },
            limitations=limitations,
            confidence_method="learned_model_max_probability_not_calibrated",
        )


@dataclass(frozen=True)
class BITLevirConfig:
    """Configuration for the public BIT LEVIR-CD checkpoint."""

    checkpoint_path: str | Path
    threshold: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if not self.device.strip():
            raise ValueError("device cannot be blank")


class BITLevirChangeDetector:
    """BIT transformer inference for RGB LEVIR-CD-style image pairs."""

    architecture = "BIT base_transformer_pos_s4_dd8_dedim8"
    training_dataset = "LEVIR-CD"
    input_requirements = "registered RGB pair; resized to 256x256; [0, 1] then mean/std 0.5"

    def __init__(self, config: BITLevirConfig) -> None:
        self.config = config
        self._model: BITLevirModel | None = None

    @property
    def available(self) -> bool:
        return Path(self.config.checkpoint_path).is_file()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> BITLevirModel:
        if self._model is None:
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=self.config.device, weights_only=False
            )
            if not isinstance(checkpoint, dict) or "model_G_state_dict" not in checkpoint:
                raise RuntimeError("BIT checkpoint must contain model_G_state_dict")
            model = BITLevirModel()
            model.load_state_dict(checkpoint["model_G_state_dict"], strict=True)
            self._model = model.to(self.config.device).eval()
        return self._model

    def detect(self, before: str | Path, after: str | Path) -> ChangeDetectionResult:
        before_data, after_data, before_summary, after_summary = validate_bitemporal_geotiffs(
            before, after
        )
        if before_data.shape[0] != 3:
            raise RasterValidationError(
                f"BIT LEVIR checkpoint expects exactly 3 RGB bands, received {before_data.shape[0]}"
            )
        before_scaled, before_valid = _scale(before_data, before_summary.nodata)
        after_scaled, after_valid = _scale(after_data, after_summary.nodata)
        valid = before_valid & after_valid
        height, width = before_scaled.shape[-2:]
        before_tensor = torch.from_numpy(before_scaled[None]).float()
        after_tensor = torch.from_numpy(after_scaled[None]).float()
        before_tensor = functional.interpolate(
            before_tensor, size=(256, 256), mode="bicubic", align_corners=False
        )
        after_tensor = functional.interpolate(
            after_tensor, size=(256, 256), mode="bicubic", align_corners=False
        )
        before_tensor = before_tensor.mul(2).sub(1).to(self.config.device)
        after_tensor = after_tensor.mul(2).sub(1).to(self.config.device)
        with torch.no_grad():
            probability_tensor = torch.softmax(
                self._ensure_model()(before_tensor, after_tensor), dim=1
            )[0, 1].cpu()[None, None]
        probabilities: np.ndarray = functional.interpolate(
            probability_tensor, size=(height, width), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        mask = (probabilities >= self.config.threshold) & valid
        regions = _regions(mask, probabilities, minimum_pixels=1)
        valid_count = int(valid.sum())
        percentage = 0.0 if valid_count == 0 else float(mask.sum() / valid_count * 100)
        confidence = 0.0 if valid_count == 0 else float(
            np.mean(np.maximum(probabilities[valid], 1 - probabilities[valid]))
        )
        return ChangeDetectionResult(
            changed=bool(mask.any()), change_mask=mask, valid_mask=valid,
            changed_regions=regions, change_percentage=percentage, confidence=confidence,
            before=before_summary, after=after_summary,
            provenance={
                "provider": "person4.bit_levir",
                "architecture": self.architecture,
                "checkpoint": str(self.config.checkpoint_path),
                "training_dataset": self.training_dataset,
                "input_requirements": self.input_requirements,
                "device": self.config.device,
                "model_output": "model-generated",
            },
            limitations=(
                "Trained on LEVIR-CD building changes; performance on Sentinel imagery "
                "is not established.",
                "Confidence is a softmax score and is not calibrated for this input domain.",
            ),
            confidence_method="bit_softmax_max_probability_not_calibrated",
        )


class ChangeDetectionProvider:
    """Select learned inference when available, otherwise use the baseline."""

    def __init__(
        self,
        learned: LearnedChangeDetector | None = None,
        baseline: DeterministicBaselineChangeDetector | None = None,
    ) -> None:
        self.learned = learned
        self.baseline = baseline or DeterministicBaselineChangeDetector()

    def detect(self, before: str | Path, after: str | Path) -> ChangeDetectionResult:
        if self.learned is not None and self.learned.available:
            try:
                return self.learned.detect(before, after)
            except (OSError, RuntimeError, ValueError) as error:
                fallback = self.baseline.detect(before, after)
                return replace(
                    fallback,
                    limitations=fallback.limitations + (
                        f"Learned inference failed; baseline fallback used: {error}",
                    ),
                    provenance={
                        **fallback.provenance,
                        "fallback_reason": "learned_inference_failed",
                        "model_output": "fallback-generated",
                    },
                )
        result = self.baseline.detect(before, after)
        if self.learned is not None:
            return replace(
                result,
                limitations=result.limitations + (
                    "Configured learned model is unavailable; baseline used.",
                ),
                provenance={
                    **result.provenance,
                    "fallback_reason": "learned_model_unavailable",
                    "model_output": "fallback-generated",
                },
            )
        return result

    def run(
        self,
        analysis_id: UUID,
        step_id: str,
        before: str | Path | None = None,
        after: str | Path | None = None,
    ) -> SpecialistResult:
        if before is None or after is None:
            return SpecialistResult(
                analysis_id=analysis_id, step_id=step_id,
                specialist=Specialist.CHANGE_DETECTION,
                status=SpecialistStatus.UNAVAILABLE,
                limitations=["Before and after raster paths are required for inference."],
                provenance={"provider": "person4.change_detection"},
                error_code="missing_input",
            )
        try:
            return self.detect(before, after).to_specialist_result(analysis_id, step_id)
        except (OSError, RasterValidationError, RuntimeError, ValueError) as error:
            return SpecialistResult(
                analysis_id=analysis_id, step_id=step_id,
                specialist=Specialist.CHANGE_DETECTION,
                status=SpecialistStatus.FAILED,
                limitations=[str(error)],
                provenance={"provider": "person4.change_detection"},
                error_code="change_detection_failed",
            )


class LazyLearnedChangeProvider:
    """Open-CD-compatible provider boundary; unavailable without a trained model."""

    def __init__(self, model_factory: Callable[[], torch.nn.Module] | None = None) -> None:
        self._model_factory = model_factory
        self._model: torch.nn.Module | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def run(self, analysis_id: UUID, step_id: str) -> SpecialistResult:
        if self._model_factory is not None and self._model is None:
            self._model = self._model_factory()
        return SpecialistResult(
            analysis_id=analysis_id, step_id=step_id, specialist=Specialist.CHANGE_DETECTION,
            status=SpecialistStatus.UNAVAILABLE,
            limitations=[
                "Open-CD/MMCV is unavailable and no compatible trained model is configured."
            ],
            provenance={"provider": "open_cd_lazy", "model_loaded": str(self.model_loaded)},
            error_code="provider_unavailable",
        )


UnavailableChangeProvider = LazyLearnedChangeProvider