from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from shared.contracts import (
    AnalysisRequest,
    EvidenceArtifact,
    EvidenceKind,
    ExecutionTrace,
    ImageAsset,
    ImageRole,
    RasterMetadata,
    Specialist,
    SpecialistResult,
    SpecialistStatus,
    TraceEvent,
    TraceOutcome,
)
from shared.contracts.schema import export_schema


def asset(role: ImageRole = ImageRole.SINGLE) -> ImageAsset:
    return ImageAsset(
        original_filename="scene.tif",
        storage_key="assets/opaque-id.tif",
        content_type="image/tiff",
        format="tiff",
        role=role,
    )


def test_asset_rejects_filename_traversal() -> None:
    with pytest.raises(ValidationError):
        ImageAsset(
            original_filename="../scene.tif",
            storage_key="assets/opaque-id.tif",
            content_type="image/tiff",
            format="tiff",
            role="single",
        )


def test_request_rejects_blank_question_and_duplicate_assets() -> None:
    image_id = uuid4()
    with pytest.raises(ValidationError):
        AnalysisRequest(question="   ", asset_ids=[image_id])
    with pytest.raises(ValidationError):
        AnalysisRequest(question="What is visible?", asset_ids=[image_id, image_id])


def test_raster_metadata_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        RasterMetadata(width=1, height=1, band_count=1, is_georeferenced=True, bounds=(2, 0, 1, 3))


def test_unavailable_result_cannot_claim_an_answer() -> None:
    with pytest.raises(ValidationError):
        SpecialistResult(
            analysis_id=uuid4(),
            step_id="vision",
            specialist=Specialist.VISION,
            status=SpecialistStatus.UNAVAILABLE,
            answer="A fabricated answer",
        )


def test_evidence_requires_visual_or_geospatial_reference() -> None:
    with pytest.raises(ValidationError):
        EvidenceArtifact(
            analysis_id=uuid4(), source_asset_id=uuid4(), kind=EvidenceKind.CHANGE_MASK
        )


def test_trace_is_chronological() -> None:
    started = datetime.now(timezone.utc)
    later = started + timedelta(seconds=1)
    with pytest.raises(ValidationError):
        ExecutionTrace(
            analysis_id=uuid4(),
            started_at=started,
            outcome=TraceOutcome.FAILED,
            events=[
                TraceEvent(
                    timestamp=later,
                    step_id="a",
                    event_type="started",
                    component="test",
                    message="a",
                ),
                TraceEvent(
                    timestamp=started,
                    step_id="a",
                    event_type="failed",
                    component="test",
                    message="b",
                ),
            ],
        )


def test_schema_export_contains_public_models(tmp_path) -> None:
    destination = tmp_path / "contracts.schema.json"
    export_schema(destination)
    schema = destination.read_text(encoding="utf-8")
    assert "AnalysisRequest" in schema
    assert "SpecialistResult" in schema
