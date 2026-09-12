"""Optical/SAR fusion provider boundary without a fusion model."""

from uuid import UUID

from shared.contracts import Specialist, SpecialistResult, SpecialistStatus


class UnavailableOpticalSarProvider:
    def run(
        self, analysis_id: UUID, step_id: str, specialist: Specialist = Specialist.OPTICAL_SAR
    ) -> SpecialistResult:
        return SpecialistResult(
            analysis_id=analysis_id,
            step_id=step_id,
            specialist=specialist,
            status=SpecialistStatus.UNAVAILABLE,
            limitations=["No optical/SAR fusion model is configured."],
            provenance={"provider": "unconfigured"},
            error_code="provider_unavailable",
        )
