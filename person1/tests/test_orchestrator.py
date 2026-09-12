from uuid import UUID, uuid4

from person1.agent import (
    OrchestrationStatus,
    Orchestrator,
    ProviderRegistry,
    SpecialistInvocation,
)
from shared.contracts import (
    AnalysisRequest,
    AssetFormat,
    ImageAsset,
    ImageRole,
    RequestedCapability,
    SpecialistResult,
    SpecialistStatus,
    TraceOutcome,
)


def asset(role: ImageRole) -> ImageAsset:
    return ImageAsset(
        original_filename=f"{role.value}.tif",
        storage_key=f"{role.value}.tif",
        content_type="image/tiff",
        format=AssetFormat.GEOTIFF,
        role=role,
    )


def request(
    question: str,
    assets: list[ImageAsset],
    capability: RequestedCapability = RequestedCapability.AUTO,
) -> AnalysisRequest:
    return AnalysisRequest(
        question=question,
        asset_ids=[item.id for item in assets],
        requested_capability=capability,
    )


class MockProvider:
    def __init__(self, answer: str, confidence: float | None = None, unavailable: bool = False):
        self.answer = answer
        self.confidence = confidence
        self.unavailable = unavailable
        self.calls: list[str] = []

    def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
        self.calls.append(invocation.step.id)
        status = SpecialistStatus.UNAVAILABLE if self.unavailable else SpecialistStatus.COMPLETED
        return SpecialistResult(
            analysis_id=invocation.request.id,
            step_id=invocation.step.id,
            specialist=invocation.step.specialist,
            status=status,
            answer=None if self.unavailable else self.answer,
            confidence=None if self.unavailable else self.confidence,
            confidence_method="mock-calibrated" if self.confidence is not None else None,
            limitations=["MOCK specialist output; not real AI inference."],
            provenance={"provider": "mock", "mode": "test"},
            error_code="provider_unavailable" if self.unavailable else None,
        )


class EvidenceProvider(MockProvider):
    def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
        result = super().run(invocation)
        if result.status is SpecialistStatus.COMPLETED:
            return result.model_copy(update={"evidence_ids": [UUID(int=7)]})
        return result


def providers(**overrides: MockProvider) -> ProviderRegistry:
    return ProviderRegistry(
        preprocessing=overrides.get("preprocessing", MockProvider("prepared")),
        vision=overrides.get("vision", MockProvider("vision answer", 0.8)),
        change_detection=overrides.get("change", MockProvider("change answer", 0.7)),
        optical_sar=overrides.get("optical_sar", MockProvider("modalities analyzed", 0.6)),
        fusion=overrides.get("fusion", MockProvider("fusion answer", 0.9)),
    )


def test_vqa_routes_to_preprocessing_and_vision() -> None:
    image = asset(ImageRole.SINGLE)
    result = Orchestrator(providers=providers()).run(request("What is visible?", [image]), [image])

    assert result.status is OrchestrationStatus.COMPLETED
    assert result.plan is not None and result.plan.task.value == "vqa"
    assert [item.step_id for item in result.specialist_results] == ["preprocess", "vision"]


def test_captioning_routes_from_auto_question() -> None:
    image = asset(ImageRole.SINGLE)
    result = Orchestrator(providers=providers()).run(
        request("Describe this image", [image]), [image]
    )

    assert result.plan is not None and result.plan.task.value == "caption"
    assert result.specialist_results[-1].step_id == "vision"


def test_grounding_routes_from_auto_question() -> None:
    image = asset(ImageRole.SINGLE)
    result = Orchestrator(providers=providers()).run(
        request("Where are the roads?", [image]), [image]
    )

    assert result.plan is not None and result.plan.task.value == "grounding"


def test_change_detection_routes_before_after_pair() -> None:
    before, after = asset(ImageRole.BEFORE), asset(ImageRole.AFTER)
    result = Orchestrator(providers=providers()).run(
        request("What changed?", [before, after], RequestedCapability.CHANGE_DETECTION),
        [before, after],
    )

    assert result.plan is not None and result.plan.task.value == "change_detection"
    assert [item.step_id for item in result.specialist_results] == [
        "preprocess",
        "change_detection",
    ]


def test_optical_sar_routes_all_specialists_in_order() -> None:
    optical, sar = asset(ImageRole.OPTICAL), asset(ImageRole.SAR)
    result = Orchestrator(providers=providers()).run(
        request("Compare the modalities", [optical, sar], RequestedCapability.OPTICAL_SAR_FUSION),
        [optical, sar],
    )

    assert result.plan is not None and result.plan.task.value == "optical_sar_fusion"
    assert [item.step_id for item in result.specialist_results] == [
        "preprocess",
        "optical_sar",
        "fusion",
    ]


def test_unsupported_query_is_rejected_before_execution() -> None:
    image = asset(ImageRole.SINGLE)
    provider = MockProvider("must not run")
    result = Orchestrator(providers=ProviderRegistry(preprocessing=provider)).run(
        request("What is the weather forecast?", [image]), [image]
    )

    assert result.status is OrchestrationStatus.REJECTED
    assert result.plan is None
    assert provider.calls == []
    assert result.trace.outcome is TraceOutcome.REJECTED


def test_missing_input_is_rejected() -> None:
    result = Orchestrator().run(
        AnalysisRequest(question="What is visible?", asset_ids=[uuid4()]), []
    )

    assert result.status is OrchestrationStatus.REJECTED
    assert "at least one image asset" in result.limitations[0]


def test_specialist_unavailable_propagates_and_blocks_dependents() -> None:
    image = asset(ImageRole.SINGLE)
    result = Orchestrator(
        providers=ProviderRegistry(
            preprocessing=MockProvider("", unavailable=True),
            vision=MockProvider("must not run"),
        )
    ).run(request("What is visible?", [image]), [image])

    assert result.status is OrchestrationStatus.UNAVAILABLE
    assert result.answer is None
    assert result.specialist_results[0].status is SpecialistStatus.UNAVAILABLE
    assert any(event.event_type == "unavailable" for event in result.trace.events)


def test_synthesis_propagates_answer_confidence_evidence_and_limitations() -> None:
    image = asset(ImageRole.SINGLE)
    evidence = EvidenceProvider("grounded answer", 0.75)
    result = Orchestrator(
        providers=ProviderRegistry(
            preprocessing=MockProvider("prepared"), vision=evidence
        )
    ).run(request("What is visible?", [image]), [image])

    assert result.answer == "vision: grounded answer"
    assert result.confidence == 0.75
    assert result.evidence_ids == (UUID(int=7),)
    assert "MOCK" in result.limitations[-1]
    assert result.provenance["vision.provider"] == "mock"


def test_trace_is_structured_and_chronological() -> None:
    image = asset(ImageRole.SINGLE)
    result = Orchestrator(providers=providers()).run(request("What is visible?", [image]), [image])

    assert result.trace.outcome is TraceOutcome.COMPLETED
    assert [event.step_id for event in result.trace.events[:2]] == ["plan", "plan"]
    assert any(event.event_type == "started" for event in result.trace.events)
    assert result.trace.finished_at is not None
    assert all(
        current.timestamp >= previous.timestamp
        for previous, current in zip(result.trace.events, result.trace.events[1:])
    )