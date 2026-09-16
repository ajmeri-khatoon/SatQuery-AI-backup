"""Stable, validated data exchanged across SATQUERY components."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from pathlib import PurePath
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImageRole(StrEnum):
    SINGLE = "single"
    BEFORE = "before"
    AFTER = "after"
    OPTICAL = "optical"
    SAR = "sar"


class SensorType(StrEnum):
    UNKNOWN = "unknown"
    SENTINEL_1 = "sentinel_1"
    SENTINEL_2 = "sentinel_2"
    OTHER_OPTICAL = "other_optical"
    OTHER_SAR = "other_sar"


class AssetFormat(StrEnum):
    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"


class AssetStatus(StrEnum):
    UPLOADED = "uploaded"
    VALIDATED = "validated"
    PREPROCESSING_FAILED = "preprocessing_failed"


class RequestedCapability(StrEnum):
    AUTO = "auto"
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class TaskType(StrEnum):
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class Specialist(StrEnum):
    PREPROCESSING = "preprocessing"
    VISION = "vision"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR = "optical_sar"
    FUSION = "fusion"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class SpecialistStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class EvidenceKind(StrEnum):
    CHANGE_MASK = "change_mask"
    RASTER_OVERLAY = "raster_overlay"
    VECTOR_REGION = "vector_region"
    BOUNDING_BOX = "bounding_box"
    TEXTUAL_RATIONALE = "textual_rationale"


class TraceOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    REJECTED = "rejected"


class RasterMetadata(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    acquired_at: datetime | None = None
    nodata: float | None = None
    is_georeferenced: bool

    @field_validator("crs")
    @classmethod
    def require_nonempty_crs(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("crs cannot be blank")
        return value

    @field_validator("nodata")
    @classmethod
    def require_finite_nodata(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("nodata must be finite")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> RasterMetadata:
        if self.bounds is not None:
            min_x, min_y, max_x, max_y = self.bounds
            if (
                not all(isfinite(value) for value in self.bounds)
                or min_x >= max_x
                or min_y >= max_y
            ):
                raise ValueError("bounds must be finite and ordered min_x, min_y, max_x, max_y")
        return self


class ImageAsset(ContractModel):
    id: UUID = Field(default_factory=uuid4)
    original_filename: str = Field(min_length=1, max_length=255)
    storage_key: str = Field(min_length=1, max_length=512)
    content_type: str = Field(pattern=r"^image/(tiff|geotiff|png|jpeg)$")
    format: AssetFormat
    role: ImageRole
    sensor: SensorType = SensorType.UNKNOWN
    status: AssetStatus = AssetStatus.UPLOADED
    metadata: RasterMetadata | None = None
    uploaded_at: datetime = Field(default_factory=utc_now)

    @field_validator("original_filename")
    @classmethod
    def require_basename(cls, value: str) -> str:
        path = PurePath(value)
        if path.name != value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("original_filename must be a basename")
        return value

    @field_validator("storage_key")
    @classmethod
    def require_opaque_key(cls, value: str) -> str:
        if PurePath(value).is_absolute() or ".." in PurePath(value).parts:
            raise ValueError("storage_key must not be a filesystem path")
        return value


class AnalysisRequest(ContractModel):
    id: UUID = Field(default_factory=uuid4)
    question: str = Field(min_length=1, max_length=4000)
    asset_ids: list[UUID] = Field(min_length=1, max_length=4)
    requested_capability: RequestedCapability = RequestedCapability.AUTO
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("question")
    @classmethod
    def require_question_content(cls, value: str) -> str:
        if not value:
            raise ValueError("question cannot be blank")
        return value

    @field_validator("asset_ids")
    @classmethod
    def require_unique_assets(cls, value: list[UUID]) -> list[UUID]:
        if len(set(value)) != len(value):
            raise ValueError("asset_ids must be unique")
        return value


class PlanStep(ContractModel):
    id: str = Field(min_length=1, max_length=100)
    specialist: Specialist
    operation: str = Field(min_length=1, max_length=200)
    input_asset_ids: list[UUID] = Field(min_length=1, max_length=4)
    depends_on: list[str] = Field(default_factory=list)
    status: PlanStepStatus = PlanStepStatus.PENDING


class TaskPlan(ContractModel):
    analysis_id: UUID
    contract_version: str = Field(min_length=1)
    task: TaskType
    specialists: list[Specialist] = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)
    validation: str = Field(pattern="^(valid|rejected)$")
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_steps(self) -> TaskPlan:
        if len(set(self.specialists)) != len(self.specialists):
            raise ValueError("specialists must be unique")
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError("step IDs must be unique")
            if any(dependency not in seen for dependency in step.depends_on):
                raise ValueError("step dependencies must refer to earlier steps")
            seen.add(step.id)
        return self


class SpecialistResult(ContractModel):
    analysis_id: UUID
    step_id: str = Field(min_length=1)
    specialist: Specialist
    status: SpecialistStatus
    answer: str | None = Field(default=None, max_length=10000)
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_method: str | None = Field(default=None, max_length=200)
    evidence_ids: list[UUID] = Field(default_factory=list)
    evidence_regions: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)
    error_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$")

    @model_validator(mode="after")
    def validate_result(self) -> SpecialistResult:
        if self.confidence is not None and not self.confidence_method:
            raise ValueError("confidence_method is required with confidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")
        if self.status is SpecialistStatus.UNAVAILABLE and self.answer is not None:
            raise ValueError("unavailable specialists cannot provide an answer")
        return self


class EvidenceArtifact(ContractModel):
    id: UUID = Field(default_factory=uuid4)
    analysis_id: UUID
    source_asset_id: UUID
    kind: EvidenceKind
    storage_key: str | None = Field(default=None, max_length=512)
    region: dict[str, Any] | None = None
    label: str | None = Field(default=None, max_length=200)
    score: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_visual_or_geospatial_evidence(self) -> EvidenceArtifact:
        if self.storage_key is None and self.region is None:
            raise ValueError("evidence needs a storage_key or region")
        return self


class TraceEvent(ContractModel):
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=utc_now)
    step_id: str = Field(min_length=1)
    event_type: str = Field(pattern="^(validated|planned|started|completed|failed|unavailable)$")
    component: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ExecutionTrace(ContractModel):
    analysis_id: UUID
    trace_id: UUID = Field(default_factory=uuid4)
    started_at: datetime
    finished_at: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    outcome: TraceOutcome

    @model_validator(mode="after")
    def validate_timing(self) -> ExecutionTrace:
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if any(
            current.timestamp < previous.timestamp
            for previous, current in zip(self.events, self.events[1:])
        ):
            raise ValueError("trace events must be chronological")
        return self
