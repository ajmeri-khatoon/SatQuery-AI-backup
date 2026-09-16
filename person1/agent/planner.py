"""Deterministic query interpretation and specialist planning."""

import re
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class QueryInterpretation:
    """Structured intent extracted before any specialist is called."""

    task: TaskType
    target: str | None
    temporal_intent: str
    modality_requirements: tuple[str, ...]
    requested_evidence: tuple[str, ...]
    requested_quantitative_output: bool
    confidence_required: bool
    metadata_query: bool = False


class QueryInterpreter:
    """Small deterministic intent parser with explicit ambiguity handling."""

    _unsupported = re.compile(r"\b(weather|forecast|stock price|translate|recipe|medical)\b")
    _metadata = re.compile(
        r"\b(metadata|sensor|satellite|location|where was|when was|date|crs|coordinate)\b"
    )
    _caption = re.compile(r"\b(describe|caption|summari[sz]e|what is in)\b")
    _grounding = re.compile(
        r"\b(where|locate|find|highlight|which area|which building|which road|where are)\b"
    )
    _change = re.compile(
        r"\b(change|changed|before and after|construction|demolish|destroy|loss|gain)\b"
    )
    _modal = re.compile(r"\b(compare|difference|cross[- ]modal|optical|sar|radar)\b")
    _quantitative = re.compile(
        r"\b(how many|how much|count|area|percentage|percent|number of|amount)\b"
    )
    _evidence = re.compile(r"\b(evidence|show|region|box|bounding|mask|overlay|location)\b")

    def interpret(self, question: str) -> QueryInterpretation:
        normalized = " ".join(question.lower().split())
        if self._unsupported.search(normalized):
            raise PlanningError("query is outside supported satellite-image capabilities")
        if len(normalized.split()) < 2:
            raise PlanningError("query is ambiguous; provide a remote-sensing task or question")
        scores = {
            TaskType.CAPTION: len(self._caption.findall(normalized)),
            TaskType.GROUNDING: len(self._grounding.findall(normalized)),
            TaskType.CHANGE_DETECTION: len(self._change.findall(normalized)),
            TaskType.OPTICAL_SAR_FUSION: len(self._modal.findall(normalized)),
            TaskType.VQA: 1,
        }
        if scores[TaskType.CHANGE_DETECTION] and scores[TaskType.OPTICAL_SAR_FUSION]:
            raise PlanningError(
                "query combines temporal change and modality comparison ambiguously"
            )
        task = max(scores, key=lambda item: scores[item])
        metadata_query = bool(self._metadata.search(normalized))
        if metadata_query and not any(
            scores[item]
            for item in (TaskType.CAPTION, TaskType.GROUNDING, TaskType.CHANGE_DETECTION)
        ):
            task = TaskType.VQA
        if task is TaskType.VQA and scores[TaskType.VQA] == 1 and normalized in {
            "analyze this image",
            "analyze the image",
            "help me",
        }:
            raise PlanningError("query is ambiguous; specify VQA, captioning, grounding, or change")
        target = self._extract_target(normalized)
        temporal = "bi-temporal" if task is TaskType.CHANGE_DETECTION else "single acquisition"
        modalities = tuple(
            value for value, present in (
                ("optical", "optical" in normalized),
                ("sar", bool(re.search(r"\b(sar|radar)\b", normalized))),
            ) if present
        )
        if task is TaskType.OPTICAL_SAR_FUSION:
            modalities = ("optical", "sar")
        evidence = ("requested_visual_evidence",) if self._evidence.search(normalized) else ()
        return QueryInterpretation(
            task=task,
            target=target,
            temporal_intent=temporal,
            modality_requirements=modalities,
            requested_evidence=evidence,
            requested_quantitative_output=bool(self._quantitative.search(normalized)),
            confidence_required="confidence" in normalized or "certain" in normalized,
            metadata_query=metadata_query,
        )

    @staticmethod
    def _extract_target(question: str) -> str | None:
        match = re.search(
            r"\b(?:about|of|for|where are|locate|find|count|how much|how many)\s+([^?]+)",
            question,
        )
        if match:
            target = match.group(1).strip(" .")
            return target[:200] or None
        return None


class TaskPlanner:
    """Build a deterministic plan from query intent and declared asset roles."""

    def __init__(self, interpreter: QueryInterpreter | None = None) -> None:
        self.interpreter = interpreter or QueryInterpreter()

    def interpret(self, request: AnalysisRequest) -> QueryInterpretation:
        if request.requested_capability is not RequestedCapability.AUTO:
            task = {
                RequestedCapability.VQA: TaskType.VQA,
                RequestedCapability.CAPTION: TaskType.CAPTION,
                RequestedCapability.GROUNDING: TaskType.GROUNDING,
                RequestedCapability.CHANGE_DETECTION: TaskType.CHANGE_DETECTION,
                RequestedCapability.OPTICAL_SAR_FUSION: TaskType.OPTICAL_SAR_FUSION,
            }[request.requested_capability]
            parsed = self.interpreter.interpret(request.question)
            return QueryInterpretation(
                task=task,
                target=parsed.target,
                temporal_intent=parsed.temporal_intent,
                modality_requirements=parsed.modality_requirements,
                requested_evidence=parsed.requested_evidence,
                requested_quantitative_output=parsed.requested_quantitative_output,
                confidence_required=parsed.confidence_required,
                metadata_query=parsed.metadata_query,
            )
        return self.interpreter.interpret(request.question)

    def build(self, request: AnalysisRequest, assets: list[ImageAsset]) -> TaskPlan:
        assets_by_id = {asset.id: asset for asset in assets}
        if set(request.asset_ids) != set(assets_by_id):
            raise PlanningError("assets must exactly match request.asset_ids")
        if len(assets_by_id) != len(assets):
            raise PlanningError("assets must be unique")

        interpretation = self.interpret(request)
        task = self._select_task(request, assets, interpretation)
        ordinary_image = any(asset.format.value in {"png", "jpeg"} for asset in assets)
        specialists = self._specialists(task, interpretation, ordinary_image)
        return TaskPlan(
            analysis_id=request.id,
            contract_version=CONTRACT_VERSION,
            task=task,
            specialists=specialists,
            steps=self._steps(task, request.asset_ids, interpretation, ordinary_image),
            validation="valid",
            limitations=[],
        )

    def _select_task(
        self,
        request: AnalysisRequest,
        assets: list[ImageAsset],
        interpretation: QueryInterpretation | None = None,
    ) -> TaskType:
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
        if interpretation is not None:
            task = interpretation.task
            if task is TaskType.CHANGE_DETECTION:
                self._require_roles(
                    roles, {ImageRole.BEFORE: 1, ImageRole.AFTER: 1}, "change detection"
                )
                return task
            if task is TaskType.OPTICAL_SAR_FUSION:
                self._require_roles(
                    roles, {ImageRole.OPTICAL: 1, ImageRole.SAR: 1}, "optical/SAR fusion"
                )
                return task
            self._require_roles(roles, {ImageRole.SINGLE: 1}, "single-image analysis")
            return task
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
    def _specialists(
        task: TaskType,
        interpretation: QueryInterpretation | None = None,
        ordinary_image: bool = False,
    ) -> list[Specialist]:
        if ordinary_image and task in {TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING}:
            return [Specialist.VISION]
        if interpretation is not None and interpretation.metadata_query:
            return [Specialist.PREPROCESSING]
        if task is TaskType.CHANGE_DETECTION:
            specialists = [Specialist.PREPROCESSING, Specialist.CHANGE_DETECTION]
            if interpretation and interpretation.requested_quantitative_output:
                specialists.append(Specialist.VISION)
            return specialists
        if task is TaskType.OPTICAL_SAR_FUSION:
            return [Specialist.PREPROCESSING, Specialist.OPTICAL_SAR, Specialist.FUSION]
        return [Specialist.PREPROCESSING, Specialist.VISION]

    def _steps(
        self,
        task: TaskType,
        asset_ids: list[UUID],
        interpretation: QueryInterpretation | None = None,
        ordinary_image: bool = False,
    ) -> list[PlanStep]:
        if ordinary_image and task in {TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING}:
            return [
                PlanStep(
                    id="vision",
                    specialist=Specialist.VISION,
                    operation=task.value,
                    input_asset_ids=asset_ids,
                )
            ]
        steps = [
            PlanStep(
                id="preprocess",
                specialist=Specialist.PREPROCESSING,
                operation=(
                    "answer_metadata"
                    if interpretation and interpretation.metadata_query
                    else "validate_assets"
                ),
                input_asset_ids=asset_ids,
            )
        ]
        if interpretation and interpretation.metadata_query:
            return steps
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
            if interpretation and interpretation.requested_quantitative_output:
                steps.append(
                    PlanStep(
                        id="vision_interpretation",
                        specialist=Specialist.VISION,
                        operation="interpret_change_result",
                        input_asset_ids=asset_ids,
                        depends_on=["change_detection"],
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
