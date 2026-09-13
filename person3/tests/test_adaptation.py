import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from person3.inference import (
    LoRAAdaptationConfig,
    RemoteSensingJsonlDataset,
    split_dataset,
    train_lora,
)


def test_jsonl_dataset_preserves_real_records(tmp_path: Path) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (2, 2), color=(20, 40, 60)).save(image)
    annotations = tmp_path / "rsvqa.jsonl"
    annotations.write_text(
        json.dumps({"image": image.name, "question": "What is visible?", "answer": "water"})
        + "\n",
        encoding="utf-8",
    )

    dataset = RemoteSensingJsonlDataset(annotations)

    assert len(dataset) == 1
    assert dataset[0].answer == "water"
    assert dataset[0].image == image


def test_jsonl_dataset_rejects_missing_labels(tmp_path: Path) -> None:
    annotations = tmp_path / "invalid.jsonl"
    annotations.write_text(
        json.dumps({"image": "scene.png", "question": "What?"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid remote-sensing record"):
        RemoteSensingJsonlDataset(annotations)


def test_dataset_split_is_deterministic(tmp_path: Path) -> None:
    annotations = tmp_path / "rsvqa.jsonl"
    annotations.write_text(
        "\n".join(
            json.dumps({"image": f"scene-{index}.png", "question": "What?", "answer": str(index)})
            for index in range(5)
        ),
        encoding="utf-8",
    )
    dataset = RemoteSensingJsonlDataset(annotations)

    first = split_dataset(dataset, validation_fraction=0.4, seed=7)
    second = split_dataset(dataset, validation_fraction=0.4, seed=7)

    assert first == second
    assert len(first.train) == 3
    assert len(first.validation) == 2


def test_cpu_smoke_training_saves_adapter_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, input_ids: object, labels: object) -> object:
            return SimpleNamespace(loss=self.weight.square().sum())

        def save_pretrained(self, destination: Path) -> None:
            (destination / "adapter_model.bin").write_bytes(b"smoke")

    class TinyProcessor:
        def __call__(self, images: object, text: str, return_tensors: str) -> dict[str, object]:
            return {"input_ids": torch.ones((1, 2), dtype=torch.long)}

    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs
    peft.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    peft.get_peft_model = lambda model, config: model
    monkeypatch.setitem(sys.modules, "peft", peft)

    image = tmp_path / "scene.png"
    Image.new("RGB", (2, 2), color=(20, 40, 60)).save(image)
    annotations = tmp_path / "rsvqa.jsonl"
    annotations.write_text(
        json.dumps({"image": image.name, "question": "What is visible?", "answer": "water"})
        + "\n",
        encoding="utf-8",
    )

    output = train_lora(
        TinyModel(),
        TinyProcessor(),
        RemoteSensingJsonlDataset(annotations),
        tmp_path / "adapter",
        LoRAAdaptationConfig(
            dataset_name="synthetic test fixture; not a production dataset",
            dataset_verified=True,
        ),
    )

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert (output / "adapter_model.bin").is_file()
    assert provenance["adaptation_method"] == "LoRA/PEFT"
    assert provenance["train_examples"] == 1
    assert provenance["remote_sensing_adapted"] is True