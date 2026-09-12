"""Change-detection provider boundary without an Open-CD model or dataset."""

from uuid import UUID

from shared.contracts import Specialist, SpecialistResult, SpecialistStatus


class UnavailableChangeProvider:
    def run(self, analysis_id: UUID, step_id: str) -> SpecialistResult:
        return SpecialistResult(
            analysis_id=analysis_id,
            step_id=step_id,
            specialist=Specialist.CHANGE_DETECTION,
            status=SpecialistStatus.UNAVAILABLE,
            limitations=["No bi-temporal change model is configured."],
            provenance={"provider": "unconfigured"},
            error_code="provider_unavailable",
        )
