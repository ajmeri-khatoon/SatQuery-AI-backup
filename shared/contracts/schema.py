"""JSON Schema export for generated frontend types and API tooling."""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    AnalysisRequest,
    EvidenceArtifact,
    ExecutionTrace,
    ImageAsset,
    SpecialistResult,
    TaskPlan,
)


def export_schema(destination: Path) -> None:
    """Write the aggregate public contract schema to an explicit destination."""
    payload = {
        "contract_version": "0.1",
        "models": {
            "AnalysisRequest": AnalysisRequest.model_json_schema(),
            "EvidenceArtifact": EvidenceArtifact.model_json_schema(),
            "ExecutionTrace": ExecutionTrace.model_json_schema(),
            "ImageAsset": ImageAsset.model_json_schema(),
            "SpecialistResult": SpecialistResult.model_json_schema(),
            "TaskPlan": TaskPlan.model_json_schema(),
        },
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
