from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from person1.agent import Orchestrator
from shared.contracts import (
    AnalysisRequest,
    AssetFormat,
    ImageAsset,
    ImageRole,
    RequestedCapability,
    SensorType,
    TraceEvent,
)

from ..integration.adapters import IntegrationProviderFactory
from ..models import Analysis, Execution, Image, Result


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_contract(
    images: list[Image], question: str, requested_capability: str
) -> tuple[AnalysisRequest, list[ImageAsset]]:
    capability = RequestedCapability(requested_capability)
    if not images or len(images) > 4:
        raise ValueError("one to four images are required")
    if (
        capability
        in {
            RequestedCapability.CHANGE_DETECTION,
            RequestedCapability.OPTICAL_SAR_FUSION,
        }
        and len(images) != 2
    ):
        raise ValueError("this analysis mode requires exactly two images")
    if capability not in {
        RequestedCapability.AUTO,
        RequestedCapability.VQA,
        RequestedCapability.CAPTION,
        RequestedCapability.GROUNDING,
        RequestedCapability.CHANGE_DETECTION,
        RequestedCapability.OPTICAL_SAR_FUSION,
    }:
        raise ValueError("unsupported analysis capability")
    roles = (
        [ImageRole.BEFORE, ImageRole.AFTER]
        if capability is RequestedCapability.CHANGE_DETECTION
        else [ImageRole.OPTICAL, ImageRole.SAR]
        if capability is RequestedCapability.OPTICAL_SAR_FUSION
        else [ImageRole.SINGLE] * len(images)
    )
    assets: list[ImageAsset] = []
    for image, role in zip(images, roles):
        suffix = Path(image.filename).suffix.lower()
        if suffix not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            raise ValueError("unsupported image format")
        if capability in {
            RequestedCapability.CHANGE_DETECTION,
            RequestedCapability.OPTICAL_SAR_FUSION,
        } and suffix not in {".tif", ".tiff"}:
            raise ValueError("change and optical/SAR analysis require GeoTIFF or TIFF inputs")
        asset_format, content_type = {
            ".tif": (AssetFormat.GEOTIFF, "image/geotiff"),
            ".tiff": (AssetFormat.TIFF, "image/tiff"),
            ".png": (AssetFormat.PNG, "image/png"),
            ".jpg": (AssetFormat.JPEG, "image/jpeg"),
            ".jpeg": (AssetFormat.JPEG, "image/jpeg"),
        }[suffix]
        asset_format, content_type = {
            ".tif": (AssetFormat.GEOTIFF, "image/geotiff"),
            ".tiff": (AssetFormat.TIFF, "image/tiff"),
            ".png": (AssetFormat.PNG, "image/png"),
            ".jpg": (AssetFormat.JPEG, "image/jpeg"),
            ".jpeg": (AssetFormat.JPEG, "image/jpeg"),
        }[suffix]
        assets.append(
            ImageAsset(
                id=uuid4(),
                original_filename=image.filename,
                storage_key=image.file_path.replace("\\", "/"),
                content_type=content_type,
                format=asset_format,
                role=role,
                sensor=SensorType.UNKNOWN,
            )
        )
    request = AnalysisRequest(
        question=question,
        asset_ids=[asset.id for asset in assets],
        requested_capability=capability,
    )
    return request, assets


def create_contract_analysis(
    analysis: Analysis,
    images: list[Image],
    question: str,
    requested_capability: str,
    db: Session,
    storage_root: str | Path,
) -> Analysis:
    request, assets = build_contract(images, question, requested_capability)
    plan = Orchestrator().planner.build(request, assets)
    analysis.image_ids = [image.id for image in images]
    analysis.request_data = {
        "request": request.model_dump(mode="json"),
        "assets": [asset.model_dump(mode="json") for asset in assets],
    }
    analysis.plan_data = plan.model_dump(mode="json")
    analysis.question = question
    db.add_all(
        Execution(
            analysis_id=analysis.id,
            step=step.id,
            specialist=step.specialist.value,
            status="pending",
        )
        for step in plan.steps
    )
    db.flush()
    return analysis


def execute_analysis(analysis: Analysis, db: Session, storage_root: str | Path) -> Analysis:
    if not analysis.request_data or not analysis.plan_data:
        raise ValueError("analysis contract is missing")
    request = AnalysisRequest.model_validate(analysis.request_data["request"])
    assets = [ImageAsset.model_validate(asset) for asset in analysis.request_data["assets"]]
    analysis.status = "running"
    db.commit()
    factory = IntegrationProviderFactory(storage_root)
    orchestrator = Orchestrator(
        providers=factory.registry(),
        asset_resolver=lambda asset: str(
            (storage_root / asset.storage_key).resolve()
        ),
    )
    missing_assets = [
        asset.original_filename
        for asset in assets
        if not (Path(storage_root) / asset.storage_key).is_file()
    ]
    result = orchestrator.run(request, assets)
    if missing_assets:
        limitation = (
            "Uploaded asset storage is unavailable: "
            + ", ".join(missing_assets)
            + ". Configure SATQUERY_STORAGE_ROOT on durable storage; local web-service "
            "files cannot be relied on after a restart."
        )
        trace = result.trace.model_copy(
            update={
                "events": [
                    *result.trace.events,
                    TraceEvent(
                        step_id="asset_storage",
                        event_type="unavailable",
                        component="person5.storage",
                        message=limitation,
                        details={"error": "persistent_storage_required"},
                    ),
                ]
            }
        )
        result = replace(
            result,
            limitations=(*result.limitations, limitation),
            provenance={**result.provenance, "storage": "asset_missing"},
            trace=trace,
        )
    analysis.status = result.status.value
    analysis.plan_data = None if result.plan is None else result.plan.model_dump(mode="json")
    analysis.result_data = result.as_dict()
    analysis.trace_data = result.trace.model_dump(mode="json")
    db.execute(delete(Result).where(Result.analysis_id == analysis.id))
    for specialist_result in result.specialist_results:
        db.add(
            Result(
                analysis_id=analysis.id,
                result_type=specialist_result.specialist.value,
                data=specialist_result.model_dump(mode="json"),
            )
        )
    steps = {
        execution.step: execution
        for execution in db.scalars(
            select(Execution).where(Execution.analysis_id == analysis.id)
        ).all()
    }
    for execution in steps.values():
        execution.status = "unavailable"
        execution.completed_at = _now()
    for specialist_result in result.specialist_results:
        execution = steps.get(specialist_result.step_id)
        if execution is not None:
            execution.status = specialist_result.status.value
            execution.result_data = specialist_result.model_dump(mode="json")
            if specialist_result.limitations:
                execution.error_message = "; ".join(specialist_result.limitations)[:2000]
    for event in result.trace.events:
        execution = steps.get(event.step_id)
        if execution is not None and event.event_type == "started":
            execution.started_at = event.timestamp
    analysis.updated_at = _now()
    db.commit()
    db.refresh(analysis)
    return analysis
