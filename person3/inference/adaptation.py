"""Small, reproducible PEFT adaptation utilities for remote-sensing VLM data.

The loader accepts only records supplied by the user or a documented public
dataset export. It never creates labels or synthetic images.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class RemoteSensingExample:
    image: Path
    question: str
    answer: str
    task: str = "vqa"

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("question and answer must be non-empty")
        if self.task not in {"vqa", "captioning", "grounding"}:
            raise ValueError("task must be vqa, captioning, or grounding")


class RemoteSensingJsonlDataset:
    """Read a local JSONL export with image/question/answer records."""

    def __init__(self, annotation_file: str | Path, image_root: str | Path | None = None) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = (
            Path(image_root) if image_root is not None else self.annotation_file.parent
        )
        self.examples = tuple(self._read())
        if not self.examples:
            raise ValueError("annotation file contains no examples")

    def _read(self) -> Iterable[RemoteSensingExample]:
        with self.annotation_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    image = self.image_root / str(record["image"])
                    question = str(record.get("question", record.get("prompt", "")))
                    answer = str(record.get("answer", record.get("caption", "")))
                    task = str(record.get("task", "vqa"))
                    yield RemoteSensingExample(image, question, answer, task)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid remote-sensing record at line {line_number}"
                    ) from error

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RemoteSensingExample:
        return self.examples[index]


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[RemoteSensingExample, ...]
    validation: tuple[RemoteSensingExample, ...]


def split_dataset(
    dataset: RemoteSensingJsonlDataset,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """Create a deterministic split without downloading or inventing records."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    import random

    indexes = list(range(len(dataset)))
    random.Random(seed).shuffle(indexes)
    validation_count = int(len(indexes) * validation_fraction)
    if validation_fraction and len(indexes) > 1:
        validation_count = max(1, validation_count)
    validation_indexes = indexes[:validation_count]
    train_indexes = indexes[validation_count:]
    return DatasetSplit(
        train=tuple(dataset[index] for index in train_indexes),
        validation=tuple(dataset[index] for index in validation_indexes),
    )


@dataclass(frozen=True)
class LoRAAdaptationConfig:
    base_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    dataset_name: str = "MBZUAI/GeoChat_Instruct or RSVQA export"
    rank: int = 4
    alpha: int = 8
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 2e-4
    epochs: int = 1
    max_new_tokens: int = 128
    validation_fraction: float = 0.2
    split_seed: int = 42
    dataset_license: str = "unspecified"
    dataset_verified: bool = False

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha < 1 or self.epochs < 1:
            raise ValueError("rank, alpha, and epochs must be positive")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")


class VisionTrainingModel(Protocol):
    def train(self, mode: bool = True) -> Any: ...

    def __call__(self, **inputs: Any) -> Any: ...


def train_lora(
    model: VisionTrainingModel,
    processor: Any,
    dataset: RemoteSensingJsonlDataset,
    output_dir: str | Path,
    config: LoRAAdaptationConfig = LoRAAdaptationConfig(),
) -> Path:
    """Run PEFT training and save an adapter only with explicit provenance.

    Transformers and PEFT are imported only when this function is called. A
    caller may inject a tiny compatible model for a CPU smoke run.
    """
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("install the optional 'adaptation' dependencies first") from error

    if not config.dataset_verified:
        raise ValueError(
            "set dataset_verified=True only after verifying the public remote-sensing dataset, "
            "license, and image/annotation provenance"
        )
    split = split_dataset(dataset, config.validation_fraction, config.split_seed)
    if not split.train:
        raise ValueError("training split is empty")
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    adapted_model = get_peft_model(model, peft_config)
    adapted_model.train()
    optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        for example in split.train:
            with _open_image(example.image) as image:
                prompt = f"{example.question}\nAnswer: {example.answer}"
                inputs = processor(images=image, text=prompt, return_tensors="pt")
            labels = inputs["input_ids"].clone()
            outputs = adapted_model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    adapted_model.save_pretrained(destination)
    (destination / "provenance.json").write_text(
        json.dumps(
            {
                "base_model": config.base_model,
                "dataset": config.dataset_name,
                "dataset_license": config.dataset_license,
                "remote_sensing_adapted": True,
                "adaptation_method": "LoRA/PEFT",
                "train_examples": len(split.train),
                "validation_examples": len(split.validation),
                "config": asdict(config),
                "limitations": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def evaluate_lora(
    model: Any,
    processor: Any,
    examples: Iterable[RemoteSensingExample],
    max_new_tokens: int = 128,
) -> dict[str, float | int]:
    """Evaluate generated answers against supplied references without fabricating metrics."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required for adaptation evaluation") from error
    total = 0
    exact_matches = 0
    for example in examples:
        with _open_image(example.image) as image:
            inputs = processor(
                images=image,
                text=example.question,
                return_tensors="pt",
            )
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        total += 1
        exact_matches += int(answer.casefold() == example.answer.casefold())
    return {
        "examples": total,
        "exact_match": (exact_matches / total) if total else 0.0,
    }


def _open_image(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"dataset image does not exist: {path}")
    from PIL import Image

    return Image.open(path).convert("RGB")