"""Provider-agnostic planning, execution, and synthesis for Person 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Protocol
from uuid import UUID

from shared.contracts import (
    AnalysisRequest,
    AssetStatus,
    ExecutionTrace,
    ImageAsset,
    ImageRole,
    PlanStep,
    SensorType,
    Specialist,
    SpecialistResult,
    SpecialistStatus,
    TaskPlan,
    TraceEvent,
    TraceOutcome,
)

from .planner import PlanningError, QueryInterpretation, TaskPlanner


class OrchestrationError(ValueError):
    """Raised for invalid orchestration setup or provider output."""


class OrchestrationStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class SpecialistInvocation:
    """Stable adapter input that keeps model-specific details outside Person 1."""

    request: AnalysisRequest
    assets: tuple[ImageAsset, ...]
    plan: TaskPlan
    step: PlanStep
    previous_results: tuple[SpecialistResult, ...]
    interpretation: QueryInterpretation | None = None


class SpecialistProvider(Protocol):
    """Common contract for all specialist adapters."""

    def run(self, invocation: SpecialistInvocation) -> SpecialistResult: ...


class PreprocessingProvider(SpecialistProvider, Protocol):
    """Adapter boundary owned by Person 2."""


class VisionProvider(SpecialistProvider, Protocol):
    """Adapter boundary owned by Person 3."""


class ChangeDetectionProvider(SpecialistProvider, Protocol):
    """Adapter boundary owned by Person 4 change detection."""


class OpticalSarProvider(SpecialistProvider, Protocol):
    """Adapter boundary owned by Person 4 optical/SAR analysis."""


class FusionProvider(SpecialistProvider, Protocol):
    """Adapter boundary for Person 4 modality fusion."""


@dataclass(frozen=True)
class ProviderRegistry:
    preprocessing: PreprocessingProvider | None = None
    vision: VisionProvider | None = None
    change_detection: ChangeDetectionProvider | None = None
    optical_sar: OpticalSarProvider | None = None
    fusion: FusionProvider | None = None

    def for_specialist(self, specialist: Specialist) -> SpecialistProvider | None:
        return {
            Specialist.PREPROCESSING: self.preprocessing,
            Specialist.VISION: self.vision,
            Specialist.CHANGE_DETECTION: self.change_detection,
            Specialist.OPTICAL_SAR: self.optical_sar,
            Specialist.FUSION: self.fusion,
        }[specialist]


@dataclass(frozen=True)
class OrchestrationResult:
    """Structured final output consumed by later API and integration layers."""

    status: OrchestrationStatus
    plan: TaskPlan | None
    answer: str | None
    confidence: float | None
    evidence_ids: tuple[UUID, ...]
    limitations: tuple[str, ...]
    provenance: dict[str, str]
    specialist_results: tuple[SpecialistResult, ...]
    trace: ExecutionTrace
    interpretation: QueryInterpretation | None = None
    confidence_rationale: str = "Confidence was not available from completed specialists."
    evidence_regions: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "plan": None if self.plan is None else self.plan.model_dump(mode="json"),
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence_ids": tuple(str(value) for value in self.evidence_ids),
            "limitations": self.limitations,
            "provenance": self.provenance,
            "specialist_results": tuple(
                result.model_dump(mode="json") for result in self.specialist_results
            ),
            "interpretation": None if self.interpretation is None else self.interpretation.__dict__,
            "confidence_rationale": self.confidence_rationale,
            "evidence_regions": self.evidence_regions,
            "trace": self.trace.model_dump(mode="json"),
        }


class Orchestrator:
    """Execute a planner's steps without knowing how any model works."""

    def __init__(
        self,
        planner: TaskPlanner | None = None,
        providers: ProviderRegistry | None = None,
        asset_resolver: Callable[[ImageAsset], str | Path | None] | None = None,
    ):
        self.planner = planner or TaskPlanner()
        self.providers = providers or ProviderRegistry()
        self.asset_resolver = asset_resolver

    def run(self, request: AnalysisRequest, assets: list[ImageAsset]) -> OrchestrationResult:
        started_at = datetime.now(timezone.utc)
        events: list[TraceEvent] = []
        interpretation: QueryInterpretation | None = None
        try:
            validation_limitations = self._validate_inputs(request, assets)
            interpretation = self.planner.interpret(request)
            plan = self.planner.build(request, assets)
        except (PlanningError, OrchestrationError) as error:
            events.append(
                self._event(
                    request.id,
                    "validation",
                    "failed",
                    str(error),
                    {
                        "error": "input_rejected",
                        "request": self._json(request.model_dump(mode="json")),
                    },
                )
            )
            trace = self._trace(request.id, started_at, events, TraceOutcome.REJECTED)
            return OrchestrationResult(
                status=OrchestrationStatus.REJECTED,
                plan=None,
                answer=None,
                confidence=None,
                evidence_ids=(),
                limitations=(str(error),),
                provenance={"component": "person1.orchestrator"},
                specialist_results=(),
                trace=trace,
                interpretation=interpretation,
            )

        events.append(
            self._event(
                request.id,
                "plan",
                "validated",
                "request and assets validated",
                {
                    "request": self._json(request.model_dump(mode="json")),
                    "limitations": self._json(validation_limitations),
                },
            )
        )
        events.append(self._event(request.id, "plan", "validated", "task plan validated"))
        events.append(
            self._event(
                request.id,
                "plan",
                "planned",
                f"selected task {plan.task.value}",
                {
                    "specialists": ",".join(specialist.value for specialist in plan.specialists),
                    "interpretation": self._json(interpretation.__dict__),
                },
            )
        )
        asset_map = {asset.id: asset for asset in assets}
        results: list[SpecialistResult] = []
        completed_steps: set[str] = set()
        for step in plan.steps:
            blocked = [
                dependency for dependency in step.depends_on if dependency not in completed_steps
            ]
            if blocked:
                events.append(
                    self._event(
                        request.id,
                        step.id,
                        "unavailable",
                        f"step blocked by unavailable dependency: {', '.join(blocked)}",
                        {"error": "dependency_unavailable"},
                    )
                )
                results.append(
                    self._synthetic_result(
                        request, step, SpecialistStatus.UNAVAILABLE, "dependency_unavailable"
                    )
                )
                continue
            provider = self.providers.for_specialist(step.specialist)
            if provider is None:
                events.append(
                    self._event(
                        request.id,
                        step.id,
                        "unavailable",
                        f"no provider configured for {step.specialist.value}",
                        {
                            "error": "provider_unavailable",
                            "inputs": self._json([str(item) for item in step.input_asset_ids]),
                        },
                    )
                )
                results.append(
                    self._synthetic_result(
                        request, step, SpecialistStatus.UNAVAILABLE, "provider_unavailable"
                    )
                )
                continue
            events.append(
                self._event(
                    request.id,
                    step.id,
                    "started",
                    f"started {step.operation}",
                    {
                        "operation": step.operation,
                        "inputs": self._json([str(item) for item in step.input_asset_ids]),
                        "configuration": self._json({"specialist": step.specialist.value}),
                    },
                )
            )
            invocation = SpecialistInvocation(
                request=request,
                assets=tuple(asset_map[asset_id] for asset_id in step.input_asset_ids),
                plan=plan,
                step=step,
                previous_results=tuple(results),
                interpretation=interpretation,
            )
            try:
                result = provider.run(invocation)
                self._validate_result(result, request.id, step)
            except Exception as error:
                events.append(
                    self._event(
                        request.id,
                        step.id,
                        "failed",
                        f"provider failed: {error}",
                        {"error": "provider_failed"},
                    )
                )
                results.append(
                    self._synthetic_result(
                        request, step, SpecialistStatus.FAILED, "provider_failed", str(error)
                    )
                )
                continue
            results.append(result)
            if result.status is SpecialistStatus.COMPLETED:
                completed_steps.add(step.id)
                events.append(
                    self._event(
                        request.id,
                        step.id,
                        "completed",
                        "specialist completed",
                        {"output": self._json(result.model_dump(mode="json"))},
                    )
                )
            elif result.status is SpecialistStatus.UNAVAILABLE:
                events.append(
                    self._event(
                        request.id,
                        step.id,
                        "unavailable",
                        "specialist unavailable",
                        {"output": self._json(result.model_dump(mode="json"))},
                    )
                )
            else:
                events.append(self._event(request.id, step.id, "failed", "specialist failed"))

        synthesis = self._synthesize(plan, results, validation_limitations)
        outcome = {
            OrchestrationStatus.COMPLETED: TraceOutcome.COMPLETED,
            OrchestrationStatus.PARTIAL: TraceOutcome.PARTIAL,
            OrchestrationStatus.UNAVAILABLE: TraceOutcome.PARTIAL,
            OrchestrationStatus.FAILED: TraceOutcome.FAILED,
        }[synthesis[0]]
        trace = self._trace(request.id, started_at, events, outcome)
        return OrchestrationResult(
            status=synthesis[0],
            plan=plan,
            answer=synthesis[1],
            confidence=synthesis[2],
            evidence_ids=synthesis[3],
            limitations=tuple(plan.limitations) + synthesis[4],
            provenance=synthesis[5],
            specialist_results=tuple(results),
            trace=trace,
            interpretation=interpretation,
            confidence_rationale=synthesis[6],
            evidence_regions=synthesis[7],
        )

    def _validate_inputs(
        self, request: AnalysisRequest, assets: list[ImageAsset]
    ) -> tuple[str, ...]:
        if not assets:
            raise OrchestrationError("at least one image asset is required")
        asset_ids = [asset.id for asset in assets]
        if len(set(asset_ids)) != len(asset_ids):
            raise OrchestrationError("assets must be unique")
        if set(request.asset_ids) != set(asset_ids):
            raise OrchestrationError("assets must exactly match request.asset_ids")
        invalid = [
            asset.original_filename
            for asset in assets
            if asset.status.value == "preprocessing_failed"
        ]
        if invalid:
            raise OrchestrationError("assets failed preprocessing: " + ", ".join(invalid))
        limitations: list[str] = []
        for asset in assets:
            suffix = Path(asset.original_filename).suffix.lower()
            if suffix not in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}:
                raise OrchestrationError(f"unsupported image format for {asset.original_filename}")
            if asset.role is ImageRole.SAR and asset.sensor in {
                SensorType.SENTINEL_2, SensorType.OTHER_OPTICAL
            }:
                raise OrchestrationError(
                    f"SAR asset has optical sensor metadata: {asset.original_filename}"
                )
            if asset.role is ImageRole.OPTICAL and asset.sensor is SensorType.SENTINEL_1:
                raise OrchestrationError(
                    f"optical asset has SAR sensor metadata: {asset.original_filename}"
                )
            if self.asset_resolver is not None:
                resolved = self.asset_resolver(asset)
                if resolved is None or not Path(resolved).is_file():
                    raise OrchestrationError(f"image does not exist for {asset.original_filename}")
            elif asset.status is AssetStatus.UPLOADED:
                    limitations.append(
                        f"existence not checked for {asset.original_filename}; "
                        "no asset resolver configured"
                    )
        before = next((item for item in assets if item.role is ImageRole.BEFORE), None)
        after = next((item for item in assets if item.role is ImageRole.AFTER), None)
        if before and after and before.metadata and after.metadata:
            if before.metadata.acquired_at and after.metadata.acquired_at:
                if before.metadata.acquired_at >= after.metadata.acquired_at:
                    raise OrchestrationError("before acquisition must precede after acquisition")
            else:
                limitations.append(
                    "temporal ordering could not be verified because acquisition dates are missing"
                )
        optical = next((item for item in assets if item.role is ImageRole.OPTICAL), None)
        sar = next((item for item in assets if item.role is ImageRole.SAR), None)
        if optical and sar and optical.metadata and sar.metadata:
            if optical.metadata.crs != sar.metadata.crs:
                raise OrchestrationError("optical and SAR assets must use the same CRS")
            if optical.metadata.bounds != sar.metadata.bounds:
                raise OrchestrationError("optical and SAR assets must have matching spatial bounds")
        return tuple(limitations)

    @staticmethod
    def _validate_result(result: SpecialistResult, analysis_id: UUID, step: PlanStep) -> None:
        if result.analysis_id != analysis_id:
            raise OrchestrationError(f"{step.id} returned a different analysis ID")
        if result.step_id != step.id:
            raise OrchestrationError(f"{step.id} returned an unexpected step ID")
        if result.specialist is not step.specialist:
            raise OrchestrationError(f"{step.id} returned an unexpected specialist")

    @staticmethod
    def _synthesize(
        plan: TaskPlan,
        results: list[SpecialistResult],
        validation_limitations: tuple[str, ...] = (),
    ) -> tuple[
        OrchestrationStatus,
        str | None,
        float | None,
        tuple[UUID, ...],
        tuple[str, ...],
        dict[str, str],
        str,
        tuple[dict[str, object], ...],
    ]:
        completed = [result for result in results if result.status is SpecialistStatus.COMPLETED]
        unavailable = [
            result for result in results if result.status is SpecialistStatus.UNAVAILABLE
        ]
        failed = [result for result in results if result.status is SpecialistStatus.FAILED]
        limitations = tuple(
            limitation
            for result in results
            for limitation in result.limitations
        )
        evidence_ids = tuple(
            dict.fromkeys(evidence_id for result in results for evidence_id in result.evidence_ids)
        )
        evidence_regions_list: list[dict[str, object]] = []
        for result in results:
            for region in result.evidence_regions:
                if region not in evidence_regions_list:
                    evidence_regions_list.append(region)
        evidence_regions = tuple(evidence_regions_list)
        analytical = [
            result for result in completed if result.specialist is not Specialist.PREPROCESSING
        ]
        confidence_values = [
            result.confidence for result in analytical if result.confidence is not None
        ]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
        rationale = "No completed analytical specialist reported confidence."
        if confidence is not None:
            rationale = (
                "Heuristic mean of completed analytical specialist confidence values; "
                "not calibrated by Person 1."
            )
            if (
                len(confidence_values) > 1
                and max(confidence_values) - min(confidence_values) > 0.25
            ):
                confidence = max(0.0, confidence - 0.1)
                rationale += " Specialist disagreement reduced confidence by 0.10."
            if unavailable or failed:
                rationale += " Unavailable or failed steps make the result partial."
        provenance = {
            f"{result.step_id}.provider": provider
            for result in results
            for provider in [result.provenance.get("provider", result.specialist.value)]
        }
        if not analytical:
            if failed:
                status = OrchestrationStatus.FAILED
            elif unavailable:
                status = (
                    OrchestrationStatus.PARTIAL
                    if completed
                    else OrchestrationStatus.UNAVAILABLE
                )
            else:
                status = OrchestrationStatus.COMPLETED
            return (
                status,
                (
                    next((result.answer for result in completed if result.answer), None)
                    if not failed and not unavailable
                    else None
                ),
                confidence,
                evidence_ids,
                tuple(validation_limitations) + limitations,
                provenance,
                rationale,
                evidence_regions,
            )
        answer_parts = [
            f"{result.specialist.value}: {result.answer}"
            for result in analytical
            if result.answer
        ]
        status = (
            OrchestrationStatus.COMPLETED
            if not unavailable and not failed
            else OrchestrationStatus.PARTIAL
        )
        return (
            status,
            "\n".join(answer_parts) or None,
            confidence,
            evidence_ids,
            tuple(validation_limitations) + limitations,
            provenance,
            rationale,
            evidence_regions,
        )

    @staticmethod
    def _synthetic_result(
        request: AnalysisRequest,
        step: PlanStep,
        status: SpecialistStatus,
        error_code: str,
        reason: str | None = None,
    ) -> SpecialistResult:
        return SpecialistResult(
            analysis_id=request.id,
            step_id=step.id,
            specialist=step.specialist,
            status=status,
            limitations=[reason or f"{step.specialist.value} did not execute."],
            provenance={"component": "person1.orchestrator", "operation": step.operation},
            error_code=error_code,
        )

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, default=str, sort_keys=True)

    @staticmethod
    def _event(
        analysis_id: UUID,
        step_id: str,
        event_type: str,
        message: str,
        details: dict[str, str | int | float | bool | None] | None = None,
    ) -> TraceEvent:
        return TraceEvent(
            step_id=step_id,
            event_type=event_type,
            component="person1.orchestrator",
            message=message,
            details=details or {},
        )

    @staticmethod
    def _trace(
        analysis_id: UUID,
        started_at: datetime,
        events: list[TraceEvent],
        outcome: TraceOutcome,
    ) -> ExecutionTrace:
        return ExecutionTrace(
            analysis_id=analysis_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            events=events,
            outcome=outcome,
        )
