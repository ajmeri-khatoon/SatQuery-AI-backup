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

## Remote-sensing adaptation path

Research on the target machine (Windows, Python 3.14, CPU-only, 16 GB RAM) produced this choice:

| Option | Practical result |
| --- | --- |
| `MBZUAI/geochat-7B` | Genuine RS-adapted Apache-2.0 GeoChat/LLaVA checkpoint with VQA, captioning, and grounding, but 7B inference and the documented 3x A100 training setup are not suitable for this machine. |
| VRSBench / GeoChat instruction data | Legitimate RS multimodal data, but the public image archive is too large for an unplanned local download. |
| RSVQA | Legitimate VQA data and a good adaptation target; this repository accepts a local export rather than downloading it implicitly. |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Apache-2.0, 0.3B-parameter generic VLM, documented for fine-tuning, and the most realistic CPU LoRA base. It is not remote-sensing adapted until trained with real RS annotations. |

The reproducible Option B path is implemented in `inference/adaptation.py`:

- `RemoteSensingJsonlDataset` reads local JSONL records containing `image`, `question` (or
	`prompt`), and `answer` (or `caption`). It rejects missing labels and never fabricates data.
- `train_lora` uses PEFT/LoRA, saves an adapter checkpoint, and writes `provenance.json` with the
	base model, dataset declaration, method, example count, configuration, and smoke-run limitation.
- `VisionModelConfig(adapter_path=..., remote_sensing_adapted=True, adaptation_name=...)` loads a
	saved adapter through the existing lazy Hugging Face provider. Adapted configuration now requires
	an explicit adapter path, so an unverified model cannot be labelled adapted accidentally.

Install the optional dependencies with `pip install -e ".[ml,adaptation]"`. Then provide a real
RSVQA or GeoChat-derived local annotation export and image directory to the training function. The
default base is `HuggingFaceTB/SmolVLM-256M-Instruct`; its public model card lists Apache-2.0 and
0.3B parameters. Do not infer accuracy from the included one-example CPU smoke test.

This checkout does not contain downloaded model weights, an RS dataset, or a trained adapter.
Therefore it does **not** claim a completed remote-sensing-adapted inference deployment yet. The
provider will remain unavailable until a real local adapter is trained or an explicitly permitted
public adapter is supplied. Grounding remains unavailable for the generic/LoRA path unless the
loaded model adapter returns structured regions; free-form text is never parsed into fake evidence.