from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from person1.agent import (
    OrchestrationStatus,
    Orchestrator,
    ProviderRegistry,
    QueryInterpreter,
    SpecialistInvocation,
)
from shared.contracts import (
    AnalysisRequest,
    AssetFormat,
    ImageAsset,
    ImageRole,
    RasterMetadata,
    RequestedCapability,
    SensorType,
    SpecialistResult,
    SpecialistStatus,
)


def make_asset(
    role: ImageRole, *, sensor: SensorType = SensorType.UNKNOWN, metadata=None
) -> ImageAsset:
    return ImageAsset(
        original_filename=f"{role.value}.tif",
        storage_key=f"{role.value}.tif",
        content_type="image/tiff",
        format=AssetFormat.GEOTIFF,
        role=role,
        sensor=sensor,
        metadata=metadata,
    )


def make_request(question: str, assets: list[ImageAsset], capability=RequestedCapability.AUTO):
    return AnalysisRequest(
        question=question,
        asset_ids=[item.id for item in assets],
        requested_capability=capability,
    )


class Provider:
    def __init__(self, confidence: float | None = 0.8, answer: str = "answer") -> None:
        self.confidence = confidence
        self.answer = answer
        self.calls: list[str] = []

    def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
        self.calls.append(invocation.step.id)
        return SpecialistResult(
            analysis_id=invocation.request.id,
            step_id=invocation.step.id,
            specialist=invocation.step.specialist,
            status=SpecialistStatus.COMPLETED,
            answer=self.answer,
            confidence=self.confidence,
            confidence_method="provider_calibrated" if self.confidence is not None else None,
            evidence_regions=[{"label": "building", "x_min": 0.1, "y_min": 0.2}],
            provenance={"provider": "test-provider", "model": "test-model"},
        )


class FailingProvider(Provider):
    def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
        raise RuntimeError("specialist exploded")


def registry(**overrides):
    return ProviderRegistry(
        preprocessing=overrides.get("preprocessing", Provider(answer="validated")),
        vision=overrides.get("vision", Provider(answer="vision")),
        change_detection=overrides.get("change", Provider(answer="changed")),
        optical_sar=overrides.get("optical_sar", Provider(answer="modal analysis")),
        fusion=overrides.get("fusion", Provider(answer="fused")),
    )


def test_interpreter_extracts_structured_change_intent() -> None:
    interpretation = QueryInterpreter().interpret(
        "How much construction occurred between before and after?"
    )

    assert interpretation.task.value == "change_detection"
    assert interpretation.temporal_intent == "bi-temporal"
    assert interpretation.requested_quantitative_output is True
    assert interpretation.target is not None


def test_metadata_query_uses_only_person2_boundary() -> None:
    image = make_asset(ImageRole.SINGLE, sensor=SensorType.SENTINEL_2)
    result = Orchestrator(providers=registry()).run(
        make_request("What satellite and acquisition date are in the metadata?", [image]), [image]
    )

    assert result.status is OrchestrationStatus.COMPLETED
    assert result.plan is not None
    assert [step.specialist.value for step in result.plan.steps] == ["preprocessing"]
    assert result.plan.steps[0].operation == "answer_metadata"


def test_quantitative_change_adds_conditional_vision_interpretation() -> None:
    before = make_asset(ImageRole.BEFORE)
    after = make_asset(ImageRole.AFTER)
    result = Orchestrator(providers=registry()).run(
        make_request("How much construction occurred?", [before, after]), [before, after]
    )

    assert result.plan is not None
    assert [step.id for step in result.plan.steps] == [
        "preprocess",
        "change_detection",
        "vision_interpretation",
    ]
    assert result.answer is not None and "vision:" in result.answer


def test_asset_resolver_rejects_missing_image(tmp_path: Path) -> None:
    image = make_asset(ImageRole.SINGLE)
    result = Orchestrator(
        providers=registry(), asset_resolver=lambda asset: tmp_path / asset.storage_key
    ).run(make_request("What is visible?", [image]), [image])

    assert result.status is OrchestrationStatus.REJECTED
    assert "does not exist" in result.limitations[0]


def test_temporal_order_and_spatial_compatibility_are_validated() -> None:
    old = RasterMetadata(
        width=10, height=10, band_count=3, crs="EPSG:4326", bounds=(0, 0, 1, 1),
        acquired_at=datetime(2025, 1, 2, tzinfo=timezone.utc), is_georeferenced=True,
    )
    new = old.model_copy(update={"acquired_at": datetime(2025, 1, 1, tzinfo=timezone.utc)})
    before = make_asset(ImageRole.BEFORE, metadata=old)
    after = make_asset(ImageRole.AFTER, metadata=new)
    temporal = Orchestrator(providers=registry()).run(
        make_request("What changed?", [before, after]), [before, after]
    )
    assert temporal.status is OrchestrationStatus.REJECTED
    assert "before acquisition" in temporal.limitations[0]

    optical = make_asset(
        ImageRole.OPTICAL,
        sensor=SensorType.SENTINEL_2,
        metadata=old,
    )
    sar = make_asset(
        ImageRole.SAR,
        sensor=SensorType.SENTINEL_1,
        metadata=old.model_copy(update={"crs": "EPSG:3857"}),
    )
    spatial = Orchestrator(providers=registry()).run(
        make_request("Compare optical and SAR", [optical, sar]), [optical, sar]
    )
    assert spatial.status is OrchestrationStatus.REJECTED
    assert "same CRS" in spatial.limitations[0]


def test_specialist_failure_is_preserved_and_dependents_are_not_called() -> None:
    image = make_asset(ImageRole.SINGLE)
    vision = FailingProvider()
    result = Orchestrator(
        providers=ProviderRegistry(preprocessing=Provider(), vision=vision)
    ).run(make_request("What is visible?", [image]), [image])

    assert result.status is OrchestrationStatus.FAILED
    assert result.answer is None
    assert result.specialist_results[-1].error_code == "provider_failed"
    assert "specialist exploded" in result.specialist_results[-1].limitations[0]


def test_confidence_disagreement_is_explicit_and_evidence_propagates() -> None:
    image = make_asset(ImageRole.SINGLE)
    result = Orchestrator(
        providers=ProviderRegistry(
            preprocessing=Provider(confidence=None), vision=Provider(confidence=0.9)
        )
    ).run(make_request("Where are the buildings?", [image]), [image])

    assert result.confidence == 0.9
    assert "Heuristic" in result.confidence_rationale
    assert result.evidence_regions == ({"label": "building", "x_min": 0.1, "y_min": 0.2},)
    assert result.provenance["vision.provider"] == "test-provider"


def test_trace_contains_request_inputs_configuration_and_output() -> None:
    image = make_asset(ImageRole.SINGLE)
    result = Orchestrator(providers=registry()).run(
        make_request("Describe this image", [image]), [image]
    )

    details = [event.details for event in result.trace.events]
    assert any("request" in item for item in details)
    assert any("inputs" in item and "configuration" in item for item in details)
    assert any("output" in item for item in details)
    assert result.plan is not None
    assert result.trace.analysis_id == UUID(str(result.plan.analysis_id))


def test_ambiguous_query_is_rejected_without_specialist_calls() -> None:
    image = make_asset(ImageRole.SINGLE)
    provider = Provider()
    result = Orchestrator(providers=ProviderRegistry(preprocessing=provider)).run(
        make_request("Analyze this image", [image]), [image]
    )

    assert result.status is OrchestrationStatus.REJECTED
    assert provider.calls == []
    assert "ambiguous" in result.limitations[0]