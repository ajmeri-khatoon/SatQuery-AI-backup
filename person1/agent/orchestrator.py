"""Provider-agnostic planning, execution, and synthesis for Person 1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from shared.contracts import (
    AnalysisRequest,
    ExecutionTrace,
    ImageAsset,
    PlanStep,
    Specialist,
    SpecialistResult,
    SpecialistStatus,
    TaskPlan,
    TraceEvent,
    TraceOutcome,
)

from .planner import PlanningError, TaskPlanner


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
            "trace": self.trace.model_dump(mode="json"),
        }


class Orchestrator:
    """Execute a planner's steps without knowing how any model works."""

    def __init__(
        self, planner: TaskPlanner | None = None, providers: ProviderRegistry | None = None
    ):
        self.planner = planner or TaskPlanner()
        self.providers = providers or ProviderRegistry()

    def run(self, request: AnalysisRequest, assets: list[ImageAsset]) -> OrchestrationResult:
        started_at = datetime.now(timezone.utc)
        events: list[TraceEvent] = []
        try:
            self._validate_inputs(request, assets)
            plan = self.planner.build(request, assets)
        except (PlanningError, OrchestrationError) as error:
            events.append(
                self._event(
                    request.id,
                    "validation",
                    "failed",
                    str(error),
                    {"error": "input_rejected"},
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
            )

        events.append(self._event(request.id, "plan", "validated", "task plan validated"))
        events.append(
            self._event(
                request.id,
                "plan",
                "planned",
                f"selected task {plan.task.value}",
                {"specialists": ",".join(specialist.value for specialist in plan.specialists)},
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
                continue
            provider = self.providers.for_specialist(step.specialist)
            if provider is None:
                events.append(
                    self._event(
                        request.id,
                        step.id,
                        "unavailable",
                        f"no provider configured for {step.specialist.value}",
                        {"error": "provider_unavailable"},
                    )
                )
                continue
            events.append(self._event(request.id, step.id, "started", f"started {step.operation}"))
            invocation = SpecialistInvocation(
                request=request,
                assets=tuple(asset_map[asset_id] for asset_id in step.input_asset_ids),
                plan=plan,
                step=step,
                previous_results=tuple(results),
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
                continue
            results.append(result)
            if result.status is SpecialistStatus.COMPLETED:
                completed_steps.add(step.id)
                events.append(self._event(request.id, step.id, "completed", "specialist completed"))
            elif result.status is SpecialistStatus.UNAVAILABLE:
                events.append(
                    self._event(request.id, step.id, "unavailable", "specialist unavailable")
                )
            else:
                events.append(self._event(request.id, step.id, "failed", "specialist failed"))

        synthesis = self._synthesize(plan, results)
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
        )

    @staticmethod
    def _validate_inputs(request: AnalysisRequest, assets: list[ImageAsset]) -> None:
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
        plan: TaskPlan, results: list[SpecialistResult]
    ) -> tuple[
        OrchestrationStatus,
        str | None,
        float | None,
        tuple[UUID, ...],
        tuple[str, ...],
        dict[str, str],
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
            evidence_id for result in results for evidence_id in result.evidence_ids
        )
        analytical = [
            result for result in completed if result.specialist is not Specialist.PREPROCESSING
        ]
        confidence_values = [
            result.confidence for result in analytical if result.confidence is not None
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        )
        provenance = {
            f"{result.step_id}.provider": provider
            for result in results
            for provider in [result.provenance.get("provider", result.specialist.value)]
        }
        if not completed:
            status = OrchestrationStatus.UNAVAILABLE if unavailable else OrchestrationStatus.FAILED
            return status, None, None, evidence_ids, limitations, provenance
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
            limitations,
            provenance,
        )

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
