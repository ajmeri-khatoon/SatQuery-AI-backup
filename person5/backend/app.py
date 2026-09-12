"""FastAPI application factory for the local SATQUERY foundation."""

from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import Field

from person1.agent import PlanningError, TaskPlanner
from person3.inference import UnavailableVisionProvider
from person4.change_detection import UnavailableChangeProvider
from person4.optical_sar import UnavailableOpticalSarProvider
from shared.contracts import (
    AnalysisRequest,
    ContractModel,
    ExecutionTrace,
    ImageAsset,
    Specialist,
    SpecialistResult,
    SpecialistStatus,
    TaskPlan,
    TraceEvent,
    TraceOutcome,
)


class PlanPayload(ContractModel):
    request: AnalysisRequest
    assets: Annotated[list[ImageAsset], Field(min_length=1, max_length=4)]


class ExecutionResponse(ContractModel):
    plan: TaskPlan
    results: list[SpecialistResult]
    trace: ExecutionTrace


def create_app() -> FastAPI:
    app = FastAPI(title="SATQUERY AI", version="0.1.0")
    planner = TaskPlanner()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "stage": "foundation", "providers": "unconfigured"}

    @app.post("/analysis/plan")
    def plan_analysis(payload: PlanPayload):
        try:
            return planner.build(payload.request, payload.assets)
        except PlanningError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/analysis/execute", response_model=ExecutionResponse)
    def execute_analysis(payload: PlanPayload) -> ExecutionResponse:
        try:
            plan = planner.build(payload.request, payload.assets)
        except PlanningError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        now = datetime.now(timezone.utc)
        results: list[SpecialistResult] = []
        events = [
            TraceEvent(
                timestamp=now,
                step_id="plan",
                event_type="planned",
                component="orchestrator",
                message="Task plan created.",
            )
        ]
        for step in plan.steps:
            result = _unavailable_result(plan.analysis_id, step.id, step.specialist)
            results.append(result)
            events.append(
                TraceEvent(
                    timestamp=datetime.now(timezone.utc),
                    step_id=step.id,
                    event_type="unavailable",
                    component=step.specialist.value,
                    message="Provider is not configured; no analysis was performed.",
                )
            )
        return ExecutionResponse(
            plan=plan,
            results=results,
            trace=ExecutionTrace(
                analysis_id=plan.analysis_id,
                started_at=now,
                finished_at=datetime.now(timezone.utc),
                events=events,
                outcome=TraceOutcome.PARTIAL,
            ),
        )

    return app


def _unavailable_result(
    analysis_id: UUID, step_id: str, specialist: Specialist
) -> SpecialistResult:
    if specialist is Specialist.VISION:
        return UnavailableVisionProvider().run(analysis_id, step_id)
    if specialist is Specialist.CHANGE_DETECTION:
        return UnavailableChangeProvider().run(analysis_id, step_id)
    if specialist in {Specialist.OPTICAL_SAR, Specialist.FUSION}:
        return UnavailableOpticalSarProvider().run(analysis_id, step_id, specialist)
    return SpecialistResult(
        analysis_id=analysis_id,
        step_id=step_id,
        specialist=specialist,
        status=SpecialistStatus.UNAVAILABLE,
        limitations=["Preprocessing is not configured."],
        provenance={"provider": "unconfigured"},
        error_code="provider_unavailable",
    )
