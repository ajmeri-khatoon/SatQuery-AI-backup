"""Typed, lazy-loaded inference boundary for remote-sensing vision tasks.

No model or ML framework is imported at module import time. The Transformers
provider is usable with local model files by default; downloading a public model
requires explicit opt-in in :class:`VisionModelConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Protocol
from uuid import UUID

from shared.contracts import Specialist, SpecialistResult, SpecialistStatus


class VisionTask(StrEnum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"


class VisionModelKind(StrEnum):
    GENERIC_MODEL = "GENERIC_MODEL"
    REMOTE_SENSING_ADAPTED_MODEL = "REMOTE_SENSING_ADAPTED_MODEL"


class VisionStatus(StrEnum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class VisionModelConfig:
    """Explicit model configuration; no weights are loaded during construction."""

    model_identifier: str
    remote_sensing_adapted: bool = False
    adaptation_name: str | None = None
    revision: str | None = None
    device: str = "cpu"
    allow_download: bool = False
    trust_remote_code: bool = False
    adapter_path: str | Path | None = None
    adaptation_dataset: str | None = None
    adaptation_method: str | None = None
    model_kind: VisionModelKind = VisionModelKind.GENERIC_MODEL
    raster_bands: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        if not self.model_identifier.strip():
            raise ValueError("model_identifier cannot be blank")
        if not self.device.strip():
            raise ValueError("device cannot be blank")
        if self.remote_sensing_adapted and not self.adaptation_name:
            raise ValueError(
                "adaptation_name is required when remote_sensing_adapted is true"
            )
        if self.remote_sensing_adapted and self.adapter_path is None:
            raise ValueError("adapter_path is required for a remote-sensing adapted model")
        if (
            self.remote_sensing_adapted
            and self.model_kind is not VisionModelKind.REMOTE_SENSING_ADAPTED_MODEL
        ):
            raise ValueError("adapted models must use REMOTE_SENSING_ADAPTED_MODEL")
        if self.model_kind is VisionModelKind.REMOTE_SENSING_ADAPTED_MODEL and (
            not self.remote_sensing_adapted or self.adapter_path is None
        ):
            raise ValueError("adapted model kind requires an adapter and adapted flag")
        if not self.raster_bands or any(band < 1 for band in self.raster_bands):
            raise ValueError("raster_bands must contain positive 1-based indexes")


@dataclass(frozen=True)
class VisionRequest:
    """A validated image-and-task request accepted by every vision provider."""

    image_reference: str | Path
    task: VisionTask
    prompt: str | None = None
    analysis_id: UUID | None = None
    step_id: str = "vision"

    def __post_init__(self) -> None:
        image_path = Path(self.image_reference)
        if not image_path.suffix:
            raise ValueError("image_reference must include an image file extension")
        if self.task in {VisionTask.VQA, VisionTask.GROUNDING} and not self.prompt:
            raise ValueError(f"prompt is required for {self.task.value}")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt cannot be blank")
        if not self.step_id.strip():
            raise ValueError("step_id cannot be blank")

    @property
    def image_path(self) -> Path:
        return Path(self.image_reference)


@dataclass(frozen=True)
class VisionRegion:
    """An optional normalized image region in xyxy order."""

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float | None = None

    def __post_init__(self) -> None:
        coordinates = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(0 <= value <= 1 for value in coordinates):
            raise ValueError("region coordinates must be normalized to [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("region coordinates must be ordered and non-empty")
        if not self.label.strip():
            raise ValueError("region label cannot be blank")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("region score must be in [0, 1]")

    def as_dict(self) -> dict[str, str | float | None]:
        return {
            "label": self.label,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "score": self.score,
        }


@dataclass(frozen=True)
class VisionProvenance:
    provider: str
    model_identifier: str
    remote_sensing_adapted: bool
    adaptation_name: str | None = None
    model_revision: str | None = None
    adaptation_dataset: str | None = None
    adaptation_method: str | None = None
    model_kind: VisionModelKind = VisionModelKind.GENERIC_MODEL

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "provider": self.provider,
            "model_identifier": self.model_identifier,
            "remote_sensing_adapted": self.remote_sensing_adapted,
            "adaptation_name": self.adaptation_name,
            "model_revision": self.model_revision,
            "adaptation_dataset": self.adaptation_dataset,
            "adaptation_method": self.adaptation_method,
            "model_kind": self.model_kind.value,
        }


@dataclass(frozen=True)
class VisionResult:
    task: VisionTask
    status: VisionStatus
    answer: str | None
    confidence: float | None
    evidence: tuple[VisionRegion, ...]
    provenance: VisionProvenance
    limitations: tuple[str, ...] = ()
    error_code: str | None = None
    analysis_id: UUID | None = None
    step_id: str = "vision"
    query: str | None = None
    image_dimensions: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.status is VisionStatus.COMPLETED and not self.answer:
            raise ValueError("completed results require an answer")
        if self.status is not VisionStatus.COMPLETED and self.answer is not None:
            raise ValueError("unavailable or failed results cannot contain an answer")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.status is VisionStatus.UNAVAILABLE and not self.error_code:
            raise ValueError("unavailable results require an error_code")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "status": self.status.value,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence": tuple(region.as_dict() for region in self.evidence),
            "provenance": self.provenance.as_dict(),
            "limitations": self.limitations,
            "error_code": self.error_code,
            "analysis_id": None if self.analysis_id is None else str(self.analysis_id),
            "step_id": self.step_id,
            "query": self.query,
            "image_dimensions": self.image_dimensions,
        }

    def to_specialist_result(self) -> SpecialistResult | None:
        """Adapt a result to the existing shared orchestration contract."""
        if self.analysis_id is None:
            return None
        status = {
            VisionStatus.COMPLETED: SpecialistStatus.COMPLETED,
            VisionStatus.UNAVAILABLE: SpecialistStatus.UNAVAILABLE,
            VisionStatus.FAILED: SpecialistStatus.FAILED,
        }[self.status]
        return SpecialistResult(
            analysis_id=self.analysis_id,
            step_id=self.step_id,
            specialist=Specialist.VISION,
            status=status,
            answer=self.answer,
                limitations=list(self.limitations),
                evidence_regions=[region.as_dict() for region in self.evidence],
            provenance={
                key: str(value)
                for key, value in self.provenance.as_dict().items()
                if value is not None
            },
            error_code=self.error_code,
        )


class VisionProvider(Protocol):
    def run(self, request: VisionRequest) -> VisionResult: ...


class UnavailableVisionProvider:
    """Provider used when no executable vision backend has been configured."""

    def __init__(self, reason: str = "No vision model is configured.") -> None:
        self.reason = reason

    def run(self, request: VisionRequest) -> VisionResult:
        return _unavailable_result(
            request,
            model_identifier="unconfigured",
            reason=self.reason,
            error_code="provider_unavailable",
        )


class HuggingFaceVisionProvider:
    """Lazy Transformers provider for compatible image-to-text models.

    This generic provider can produce text for all three task prompts. It does
    not invent regions from text, so grounding results explicitly report that
    evidence is unavailable unless a future model adapter supplies regions.
    """

    def __init__(self, config: VisionModelConfig) -> None:
        self.config = config
        self._processor: Any = None
        self._model: Any = None

    @property
    def model_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def _load_model(self) -> None:
        if self.model_loaded:
            return
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForVision2Seq,
                AutoProcessor,
            )
        except ImportError as error:
            raise RuntimeError("transformers is not installed") from error
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_identifier,
                revision=self.config.revision,
                local_files_only=not self.config.allow_download,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_identifier,
                revision=self.config.revision,
                local_files_only=not self.config.allow_download,
                trust_remote_code=self.config.trust_remote_code,
            )
            if self.config.adapter_path is not None:
                from peft import PeftModel  # type: ignore[import-not-found]

                adapter_path = Path(self.config.adapter_path)
                if not self.config.allow_download and not adapter_path.is_dir():
                    raise FileNotFoundError(f"local adapter is unavailable: {adapter_path}")
                provenance_path = adapter_path / "provenance.json"
                if not provenance_path.is_file():
                    raise ValueError("adapter provenance.json is missing")
                import json

                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                if not provenance.get("remote_sensing_adapted"):
                    raise ValueError(
                        "adapter provenance does not declare remote-sensing adaptation"
                    )
                if not provenance.get("dataset") or not provenance.get("adaptation_method"):
                    raise ValueError(
                        "adapter provenance must include dataset and adaptation_method"
                    )
                self._model = PeftModel.from_pretrained(self._model, str(self.config.adapter_path))
            self._model.to(self.config.device)
            self._model.eval()
        except Exception:
            self._processor = None
            self._model = None
            raise

    def run(self, request: VisionRequest) -> VisionResult:
        if not request.image_path.is_file():
            return _unavailable_result(
                request,
                model_identifier=self.config.model_identifier,
                reason=f"image_reference does not point to a file: {request.image_path}",
                error_code="image_unavailable",
                config=self.config,
            )
        try:
            self._load_model()
            import torch  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
            return _unavailable_result(
                request,
                model_identifier=self.config.model_identifier,
                reason=str(error),
                error_code="model_unavailable",
                config=self.config,
            )

        try:
            image, image_dimensions, preprocessing_limitations = _load_input_image(
                request.image_path, self.config, Image
            )
            with image:
                inputs = self._processor(
                    images=image,
                    text=_prompt_for(request),
                    return_tensors="pt",
                )
            inputs = {
                key: value.to(self.config.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=256)
            answer = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            if not answer:
                return _failed_result(request, self.config, "model_empty_response")
            limitations = [
                "Confidence was not provided by the configured model; "
                "any confidence is uncalibrated."
            ]
            limitations.extend(preprocessing_limitations)
            if not self.config.remote_sensing_adapted:
                limitations.append(
                    "The configured model is not marked as remote-sensing adapted."
                )
            regions: tuple[VisionRegion, ...] = ()
            if request.task is VisionTask.GROUNDING:
                extracted = list(_extract_grounding_regions(answer))
                if not extracted:
                    return _unavailable_result(
                        request,
                        model_identifier=self.config.model_identifier,
                        reason=(
                            "The configured model cannot perform grounding "
                            "(no valid bounding boxes returned)."
                        ),
                        error_code="capability_unsupported",
                        config=self.config,
                    )
                regions = tuple(extracted)
                limitations.append("Grounding bounding boxes were extracted from raw text output.")

            return VisionResult(
                task=request.task,
                status=VisionStatus.COMPLETED,
                answer=answer,
                confidence=None,
                evidence=regions,
                provenance=_provenance(self.config),
                limitations=tuple(limitations),
                analysis_id=request.analysis_id,
                step_id=request.step_id,
                query=request.prompt,
                image_dimensions=image_dimensions,
            )
        except Exception as error:
            return _failed_result(request, self.config, "inference_failed", str(error))

    def run_vqa(self, request: VisionRequest) -> VisionResult:
        if request.task is not VisionTask.VQA:
            raise ValueError("run_vqa requires a VQA request")
        return self.run(request)

    def run_caption(self, request: VisionRequest) -> VisionResult:
        if request.task is not VisionTask.CAPTIONING:
            raise ValueError("run_caption requires a captioning request")
        return self.run(request)

    def run_grounding(self, request: VisionRequest) -> VisionResult:
        if request.task is not VisionTask.GROUNDING:
            raise ValueError("run_grounding requires a grounding request")
        return self.run(request)


def _prompt_for(request: VisionRequest) -> str:
    if request.prompt:
        return request.prompt
    return "Describe the visible content of this image factually."


def _extract_grounding_regions(text: str) -> Iterator[VisionRegion]:
    import re
    pattern = re.compile(
        r"(?:([A-Za-z0-9_-]+)\s*)?(?:\[|<box>|<loc_)\s*(0\.\d+)\s*,\s*(0\.\d+)\s*,\s*(0\.\d+)\s*,\s*(0\.\d+)\s*(?:\]|</box>|>|loc_>)"
    )
    for match in pattern.finditer(text):
        label = (match.group(1) or "object").strip()
        if not label:
            label = "object"
        try:
            x_min, y_min, x_max, y_max = (
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
                float(match.group(5)),
            )
            if x_min < x_max and y_min < y_max:
                yield VisionRegion(
                    label=label[:50], x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
                )
        except ValueError:
            pass


def _load_input_image(
    path: Path, config: VisionModelConfig, image_module: Any
) -> tuple[Any, tuple[int, int], tuple[str, ...]]:
    """Load ordinary images or explicitly selected GeoTIFF bands for RGB models."""
    if path.suffix.lower() not in {".tif", ".tiff"}:
        image = image_module.open(path).convert("RGB")
        return image, image.size, ()
    try:
        import numpy as np
        import rasterio  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("rasterio and numpy are required for GeoTIFF inference") from error
    with rasterio.open(path) as dataset:
        if max(config.raster_bands) > dataset.count:
            raise ValueError(
                f"selected raster bands {config.raster_bands} exceed GeoTIFF "
                f"band count {dataset.count}"
            )
        data = dataset.read(list(config.raster_bands)).astype("float32")
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("selected GeoTIFF bands contain no finite pixels")
    output = np.zeros_like(data, dtype=np.uint8)
    for index, band in enumerate(data):
        valid = finite[index]
        minimum, maximum = float(band[valid].min()), float(band[valid].max())
        if maximum > minimum:
            output[index] = np.clip((band - minimum) * 255 / (maximum - minimum), 0, 255)
    image = image_module.fromarray(np.moveaxis(output, 0, -1), mode="RGB")
    return (
        image,
        (image.width, image.height),
        (
            "GeoTIFF converted to RGB for the configured vision model.",
            f"Selected 1-based bands: {config.raster_bands}.",
            "Per-band min/max uint8 scaling was applied; original raster values "
            "were not passed to the model.",
        ),
    )


def _provenance(config: VisionModelConfig, provider: str = "huggingface") -> VisionProvenance:
    return VisionProvenance(
        provider=provider,
        model_identifier=config.model_identifier,
        remote_sensing_adapted=config.remote_sensing_adapted,
        adaptation_name=config.adaptation_name,
        model_revision=config.revision,
        adaptation_dataset=config.adaptation_dataset,
        adaptation_method=config.adaptation_method,
        model_kind=config.model_kind,
    )


def _unavailable_result(
    request: VisionRequest,
    model_identifier: str,
    reason: str,
    error_code: str,
    config: VisionModelConfig | None = None,
) -> VisionResult:
    provenance = _provenance(config) if config else VisionProvenance(
        provider="unconfigured",
        model_identifier=model_identifier,
        remote_sensing_adapted=False,
    )
    return VisionResult(
        task=request.task,
        status=VisionStatus.UNAVAILABLE,
        answer=None,
        confidence=None,
        evidence=(),
        provenance=provenance,
        limitations=(reason,),
        error_code=error_code,
        analysis_id=request.analysis_id,
        step_id=request.step_id,
    )


def _failed_result(
    request: VisionRequest,
    config: VisionModelConfig,
    error_code: str,
    reason: str = "Model inference failed.",
) -> VisionResult:
    return VisionResult(
        task=request.task,
        status=VisionStatus.FAILED,
        answer=None,
        confidence=None,
        evidence=(),
        provenance=_provenance(config),
        limitations=(reason,),
        error_code=error_code,
        analysis_id=request.analysis_id,
        step_id=request.step_id,
    )
