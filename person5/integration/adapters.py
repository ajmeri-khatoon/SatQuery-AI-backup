from __future__ import annotations

import json
import os
from pathlib import Path

from person1.agent import SpecialistInvocation
from person2.preprocessing import (
	BeforeAfterInput,
	OpticalSarInput,
	RasterPreprocessingProvider,
)
from person3.inference import (
	UnavailableVisionProvider,
	VisionRequest,
	VisionTask,
)
from person4.change_detection import (
	BITLevirChangeDetector,
	BITLevirConfig,
	ChangeDetectionProvider,
)
from person4.optical_sar import OpticalSarProvider
from shared.contracts import Specialist, SpecialistResult, SpecialistStatus


class IntegrationProviderFactory:
	"""Build P1 providers while keeping model and storage details in Person 5."""

	def __init__(self, storage_root: str | Path, bit_checkpoint: str | Path | None = None):
		self.storage_root = Path(storage_root).resolve()
		checkpoint = bit_checkpoint or os.getenv("SATQUERY_BIT_CHECKPOINT")
		self.bit_checkpoint = None if not checkpoint else Path(checkpoint)

	def registry(self):
		from person1.agent import ProviderRegistry

		return ProviderRegistry(
			preprocessing=PreprocessingAdapter(self.storage_root),
			vision=VisionAdapter(self.storage_root),
			change_detection=ChangeAdapter(self.storage_root, self.bit_checkpoint),
			optical_sar=OpticalSarAdapter(self.storage_root),
			fusion=FusionAdapter(),
		)


def _path(root: Path, storage_key: str) -> Path:
	path = (root / storage_key).resolve()
	if root != path and root not in path.parents:
		raise ValueError("asset storage key escapes the configured storage root")
	return path


def _failed(invocation: SpecialistInvocation, message: str, code: str) -> SpecialistResult:
	return SpecialistResult(
		analysis_id=invocation.request.id, step_id=invocation.step.id,
		specialist=invocation.step.specialist, status=SpecialistStatus.FAILED,
		limitations=["The specialist could not process the supplied input."],
		provenance={"provider": "person5.integration", "failure": code}, error_code=code,
	)


class PreprocessingAdapter:
	def __init__(self, storage_root: Path):
		self.provider = RasterPreprocessingProvider(storage_root)

	def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
		try:
			paths = {asset.role.value: _path(self.provider.storage_root, asset.storage_key) for asset in invocation.assets}
			if len(paths) == 1:
				asset = invocation.assets[0]
				return self.provider.validate(asset, str(invocation.request.id), invocation.step.id)
			if "before" in paths and "after" in paths:
				from person2.preprocessing import prepare

				prepare(BeforeAfterInput(paths["before"], paths["after"]))
			elif "optical" in paths and "sar" in paths:
				from person2.preprocessing import prepare

				prepare(OpticalSarInput(paths["optical"], paths["sar"]))
			else:
				raise ValueError("asset roles do not form a supported preprocessing input")
			return SpecialistResult(
				analysis_id=invocation.request.id, step_id=invocation.step.id,
				specialist=Specialist.PREPROCESSING, status=SpecialistStatus.COMPLETED,
				answer="Raster inputs validated and spatially compatible.",
				limitations=["Sensor identity was not inferred by preprocessing."],
				provenance={"component": "person2.preprocessing", "operation": "prepare"},
			)
		except Exception as error:
			return _failed(invocation, str(error), "preprocessing_failed")


class VisionAdapter:
	def __init__(self, storage_root: Path):
		self.storage_root = storage_root
		model_id = os.getenv("SATQUERY_VISION_MODEL")
		self.provider = UnavailableVisionProvider() if not model_id else None
		if model_id:
			from person3.inference import HuggingFaceVisionProvider, VisionModelConfig

			self.provider = HuggingFaceVisionProvider(VisionModelConfig(model_identifier=model_id))

	def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
		asset = invocation.assets[0]
		task = VisionTask.CAPTIONING if invocation.plan.task.value == "caption" else VisionTask(
			invocation.plan.task.value
		)
		request = VisionRequest(
			image_reference=_path(self.storage_root, asset.storage_key), task=task,
			prompt=invocation.request.question, analysis_id=invocation.request.id,
			step_id=invocation.step.id,
		)
		result = self.provider.run(request)
		adapted = result.to_specialist_result()
		if adapted is None:
			return _failed(invocation, "Vision provider did not return an analysis ID.", "vision_contract_error")
		return adapted.model_copy(update={"confidence": result.confidence, "confidence_method": "provider_reported" if result.confidence is not None else None})


class ChangeAdapter:
	def __init__(self, storage_root: Path, bit_checkpoint: Path | None):
		self.storage_root = storage_root
		learned = None
		if bit_checkpoint is not None:
			learned = BITLevirChangeDetector(BITLevirConfig(checkpoint_path=bit_checkpoint))
		self.provider = ChangeDetectionProvider(learned=learned)

	def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
		try:
			return self.provider.run(
				invocation.request.id,
				invocation.step.id,
				_path(self.storage_root, _role_key(invocation, "before")),
				_path(self.storage_root, _role_key(invocation, "after")),
			)
		except Exception as error:
			return _failed(invocation, str(error), "change_detection_failed")


class OpticalSarAdapter:
	def __init__(self, storage_root: Path):
		self.storage_root = storage_root
		self.provider = OpticalSarProvider()

	def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
		try:
			return self.provider.run(
				invocation.request.id,
				invocation.step.id,
				_path(self.storage_root, _role_key(invocation, "optical")),
				_path(self.storage_root, _role_key(invocation, "sar")),
			)
		except Exception as error:
			return _failed(invocation, str(error), "optical_sar_failed")


class FusionAdapter:
	def run(self, invocation: SpecialistInvocation) -> SpecialistResult:
		optical = next((result for result in invocation.previous_results if result.specialist is Specialist.OPTICAL_SAR), None)
		if optical is None:
			return SpecialistResult(
				analysis_id=invocation.request.id, step_id=invocation.step.id, specialist=Specialist.FUSION,
				status=SpecialistStatus.UNAVAILABLE, limitations=["Optical/SAR output was unavailable."],
				provenance={"provider": "person5.fusion_adapter"}, error_code="dependency_unavailable",
			)
		return SpecialistResult(
			analysis_id=invocation.request.id, step_id=invocation.step.id, specialist=Specialist.FUSION,
			status=SpecialistStatus.COMPLETED,
			answer=json.dumps({"optical_sar_step": optical.step_id, "fused": False}, sort_keys=True),
			limitations=["No additional learned fusion interpretation is configured; the specialist output is retained verbatim."],
			provenance={"provider": "person5.fusion_adapter", "source": optical.provenance.get("provider", "person4")},
		)


def _role_key(invocation: SpecialistInvocation, role: str) -> str:
	for asset in invocation.assets:
		if asset.role.value == role:
			return asset.storage_key
	raise ValueError(f"missing {role} asset")