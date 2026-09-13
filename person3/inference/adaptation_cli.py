"""Command-line entry points for reproducible remote-sensing adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate a Person 3 LoRA adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--annotations", type=Path, required=True)
    train.add_argument("--image-root", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--base-model", required=True)
    train.add_argument("--dataset-name", required=True)
    train.add_argument("--dataset-license", required=True)
    train.add_argument(
        "--verified-remote-sensing-dataset",
        action="store_true",
        help="required acknowledgement that annotations/images are a legitimate public RS dataset",
    )
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command != "train":
        raise ValueError("unsupported adaptation command")
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from .adaptation import (
            LoRAAdaptationConfig,
            RemoteSensingJsonlDataset,
            train_lora,
        )
    except ImportError as error:
        raise SystemExit(
            "Install the optional ML/adaptation dependencies before training: "
            "pip install -e .[ml,adaptation]"
        ) from error
    dataset = RemoteSensingJsonlDataset(args.annotations, args.image_root)
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForVision2Seq.from_pretrained(args.base_model)
    output = train_lora(
        model,
        processor,
        dataset,
        args.output,
        LoRAAdaptationConfig(
            base_model=args.base_model,
            dataset_name=args.dataset_name,
            dataset_license=args.dataset_license,
            dataset_verified=args.verified_remote_sensing_dataset,
            validation_fraction=args.validation_fraction,
            split_seed=args.seed,
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
