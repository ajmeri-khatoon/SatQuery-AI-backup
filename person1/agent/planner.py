"""Deterministic, inspectable task planning before model orchestration exists."""

from collections import Counter
from uuid import UUID

from shared.contracts import (
    CONTRACT_VERSION,
    AnalysisRequest,
    ImageAsset,
    ImageRole,
    PlanStep,
    RequestedCapability,
    Specialist,
    TaskPlan,
    TaskType,
)


class PlanningError(ValueError):
    """Raised when request assets cannot support the requested task."""


class TaskPlanner:
    """Classifies a request from declared roles; it never invents image understanding."""

    def build(self, request: AnalysisRequest, assets: list[ImageAsset]) -> TaskPlan:
        assets_by_id = {asset.id: asset for asset in assets}
        if set(request.asset_ids) != set(assets_by_id):
            raise PlanningError("assets must exactly match request.asset_ids")
        if len(assets_by_id) != len(assets):
            raise PlanningError("assets must be unique")

        task = self._select_task(request, assets)
        specialists = self._specialists(task)
        return TaskPlan(
            analysis_id=request.id,
            contract_version=CONTRACT_VERSION,
            task=task,
            specialists=specialists,
            steps=self._steps(task, request.asset_ids),
            validation="valid",
            limitations=["Specialist providers are not configured in the foundation stage."],
        )

    def _select_task(self, request: AnalysisRequest, assets: list[ImageAsset]) -> TaskType:
        roles = Counter(asset.role for asset in assets)
        capability = request.requested_capability
        if capability is RequestedCapability.CHANGE_DETECTION:
            self._require_roles(
                roles, {ImageRole.BEFORE: 1, ImageRole.AFTER: 1}, "change detection"
            )
            return TaskType.CHANGE_DETECTION
        if capability is RequestedCapability.OPTICAL_SAR_FUSION:
            self._require_roles(
                roles, {ImageRole.OPTICAL: 1, ImageRole.SAR: 1}, "optical/SAR fusion"
            )
            return TaskType.OPTICAL_SAR_FUSION
        if capability is RequestedCapability.CAPTION:
            self._require_roles(roles, {ImageRole.SINGLE: 1}, "single-image analysis")
            return TaskType.CAPTION
        if capability is RequestedCapability.GROUNDING:
            self._require_roles(roles, {ImageRole.SINGLE: 1}, "single-image analysis")
            return TaskType.GROUNDING
        if capability is RequestedCapability.VQA:
            self._require_roles(roles, {ImageRole.SINGLE: 1}, "single-image analysis")
            return TaskType.VQA

        question = request.question.lower()
        unsupported_terms = ("weather", "forecast", "stock price", "translate", "recipe")
        if any(term in question for term in unsupported_terms):
            raise PlanningError("query is outside supported satellite-image capabilities")
        if roles[ImageRole.BEFORE] or roles[ImageRole.AFTER]:
            self._require_roles(
                roles, {ImageRole.BEFORE: 1, ImageRole.AFTER: 1}, "change detection"
            )
            return TaskType.CHANGE_DETECTION
        if roles[ImageRole.OPTICAL] or roles[ImageRole.SAR]:
            self._require_roles(
                roles, {ImageRole.OPTICAL: 1, ImageRole.SAR: 1}, "optical/SAR fusion"
            )
            return TaskType.OPTICAL_SAR_FUSION
        if any(word in question for word in ("caption", "describe")):
            return TaskType.CAPTION
        if any(word in question for word in ("where", "locate", "find", "highlight")):
            self._require_roles(roles, {ImageRole.SINGLE: 1}, "single-image analysis")
            return TaskType.GROUNDING
        self._require_roles(roles, {ImageRole.SINGLE: 1}, "single-image analysis")
        return TaskType.VQA

    @staticmethod
    def _require_roles(
        roles: Counter[ImageRole], expected: dict[ImageRole, int], label: str
    ) -> None:
        if any(roles[role] != count for role, count in expected.items()) or sum(
            roles.values()
        ) != sum(expected.values()):
            raise PlanningError(
                f"{label} requires exactly: "
                + ", ".join(f"{count} {role.value}" for role, count in expected.items())
            )

    @staticmethod
    def _specialists(task: TaskType) -> list[Specialist]:
        if task is TaskType.CHANGE_DETECTION:
            return [Specialist.PREPROCESSING, Specialist.CHANGE_DETECTION]
        if task is TaskType.OPTICAL_SAR_FUSION:
            return [Specialist.PREPROCESSING, Specialist.OPTICAL_SAR, Specialist.FUSION]
        return [Specialist.PREPROCESSING, Specialist.VISION]

    def _steps(self, task: TaskType, asset_ids: list[UUID]) -> list[PlanStep]:
        steps = [
            PlanStep(
                id="preprocess",
                specialist=Specialist.PREPROCESSING,
                operation="validate_assets",
                input_asset_ids=asset_ids,
            )
        ]
        if task is TaskType.CHANGE_DETECTION:
            steps.append(
                PlanStep(
                    id="change_detection",
                    specialist=Specialist.CHANGE_DETECTION,
                    operation="detect_change",
                    input_asset_ids=asset_ids,
                    depends_on=["preprocess"],
                )
            )
        elif task is TaskType.OPTICAL_SAR_FUSION:
            steps.extend(
                [
                    PlanStep(
                        id="optical_sar",
                        specialist=Specialist.OPTICAL_SAR,
                        operation="analyze_optical_sar",
                        input_asset_ids=asset_ids,
                        depends_on=["preprocess"],
                    ),
                    PlanStep(
                        id="fusion",
                        specialist=Specialist.FUSION,
                        operation="fuse_modalities",
                        input_asset_ids=asset_ids,
                        depends_on=["optical_sar"],
                    ),
                ]
            )
        else:
            steps.append(
                PlanStep(
                    id="vision",
                    specialist=Specialist.VISION,
                    operation=task.value,
                    input_asset_ids=asset_ids,
                    depends_on=["preprocess"],
                )
            )
        return steps
