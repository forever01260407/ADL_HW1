from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    LABELS,
    IntentDataset,
    MultiLabelCollator,
    load_test_records,
    read_json,
    resolve_preprocess_config,
    validate_prediction_csv,
    write_prediction_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for ADL HW1")
    parser.add_argument("--context-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_thresholds(model_dir: Path) -> np.ndarray:
    path = model_dir / "thresholds.json"
    if not path.exists():
        raise FileNotFoundError(f"missing thresholds file: {path}")
    values = read_json(path)
    if not isinstance(values, dict) or set(values) != set(LABELS):
        raise ValueError("thresholds.json must contain exactly the 10 official labels")
    thresholds = np.asarray([float(values[label]) for label in LABELS], dtype=np.float32)
    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError("thresholds must be between 0 and 1")
    return thresholds


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"model config not found in {model_dir}")

    preprocess_path = model_dir / "preprocess_config.json"
    preprocess_config = read_json(preprocess_path) if preprocess_path.exists() else {}
    max_length = args.max_length or int(preprocess_config.get("max_length", 512))
    preprocess_options = resolve_preprocess_config(preprocess_config)
    thresholds = load_thresholds(model_dir)

    records = load_test_records(args.context_file, args.test_file)
    ids = [record["id"] for record in records]

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("expected the saved fast tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    if model.config.num_labels != len(LABELS):
        raise ValueError(f"model has {model.config.num_labels} labels; expected {len(LABELS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.precision == "fp16"
    model.to(device)
    model.eval()

    dataset = IntentDataset(
        records,
        tokenizer=tokenizer,
        max_length=max_length,
        with_labels=False,
        input_order=preprocess_options["input_order"],
        context_max_tokens=preprocess_options["context_max_tokens"],
        turn_aware_context=preprocess_options["turn_aware_context"],
        explicit_role_markers=preprocess_options["explicit_role_markers"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiLabelCollator(tokenizer, pad_to_multiple_of=8),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    probability_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
            probability_batches.append(torch.sigmoid(logits).float().cpu().numpy())

    probabilities = np.concatenate(probability_batches, axis=0)
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)
    write_prediction_csv(args.output_file, ids, predictions)
    validate_prediction_csv(args.output_file, ids)

    print(f"Device: {device}")
    print(f"Examples: {len(ids)}")
    print(f"max_length: {max_length}")
    print(
        "Preprocess: "
        f"v{preprocess_options['preprocess_version']}; "
        f"order={preprocess_options['input_order']}; "
        f"context_max_tokens={preprocess_options['context_max_tokens']}; "
        f"turn_aware={preprocess_options['turn_aware_context']}; "
        f"markers={preprocess_options['explicit_role_markers']}"
    )
    print(f"Output: {Path(args.output_file).resolve()}")
    print("Prediction CSV validation passed.")


if __name__ == "__main__":
    main()

