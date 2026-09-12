# Person 3: Vision inference

Person 3 provides a typed inference boundary for satellite-image VQA, captioning, and grounding.
The implementation does not bundle model weights, download models during import, or claim that a
generic vision-language model is remote-sensing adapted.

## Architecture

`VisionRequest` -> `VisionProvider` -> `VisionResult`

- `VisionRequest` validates an image path, task, prompt, analysis ID, and step ID.
- `VisionModelConfig` records the model identifier and whether it is explicitly remote-sensing adapted.
- `HuggingFaceVisionProvider` lazily imports Transformers, PyTorch, and Pillow and loads the model
	only when `run()` is called.
- `VisionResult` carries status, answer, confidence when available, evidence regions, provenance,
	limitations, and an error code.
- `VisionResult.to_specialist_result()` adapts results to the shared orchestration contract when an
	analysis ID is supplied.

## Supported tasks

```python
from person3.inference import VisionRequest, VisionTask

vqa = VisionRequest("scene.tif", VisionTask.VQA, prompt="What land cover is visible?")
caption = VisionRequest("scene.tif", VisionTask.CAPTIONING)
grounding = VisionRequest("scene.tif", VisionTask.GROUNDING, prompt="Where are the buildings?")
```

Grounding evidence uses normalized `[0, 1]` `xyxy` regions through `VisionRegion`. The generic
Transformers provider does not turn free-form text into fabricated regions; it reports grounding
evidence as unavailable unless a future model adapter supplies regions.

## Model configuration and inference

```python
from person3.inference import HuggingFaceVisionProvider, VisionModelConfig

provider = HuggingFaceVisionProvider(
		VisionModelConfig(
				model_identifier="your-local-or-public-vision-model",
				remote_sensing_adapted=False,
				allow_download=False,
		)
)
result = provider.run(vqa)
print(result.as_dict())
```

The default is `allow_download=False`, so the model must already be available in the local
Transformers cache or at a local path. Set it to `True` only for an explicit run that is allowed to
download a public model. No Hugging Face credentials are required for public models. CUDA is not
configured or selected automatically; the default device is CPU.

Install the existing optional ML dependencies when running a real model:

```powershell
pip install -e ".[ml]"
```

## Real, unavailable, and mock modes

- **Real mode:** `HuggingFaceVisionProvider` runs a configured model. It returns an answer only when
	model loading and inference succeed. Generic models are explicitly marked
	`remote_sensing_adapted=False`.
- **Unavailable mode:** missing Transformers/PyTorch/Pillow, missing local model files, invalid
	configuration, or unavailable model assets produce `UNAVAILABLE` with a limitation and error code.
- **Mock mode:** mocks exist only inside tests and are marked `MOCK`; no production provider uses them.

No VRSBench, BigEarthNet, PEFT/LoRA adaptation, satellite-specific checkpoint, or remote-sensing
fine-tuning is included yet. A later adapter can set `remote_sensing_adapted=True` only when backed
by a real documented model or fine-tuning artifact and can provide structured grounding regions.