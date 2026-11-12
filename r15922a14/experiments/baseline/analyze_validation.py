from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    LABELS,
    IntentDataset,
    MultiLabelCollator,
    labels_to_multihot,
    read_json,
    read_jsonl,
    serialize_context,
    validate_train_records,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detailed diagnostics for a saved ADL HW1 validation split"
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--num-error-examples", type=int, default=20)
    return parser.parse_args()


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def score_group(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"samples": 0, "macro_f1": 0.0, "micro_f1": 0.0}
    return {
        "samples": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def raw_token_info(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, Any]:
    context_text = serialize_context(record["context"])
    context_ids = (
        tokenizer.encode(context_text, add_special_tokens=False, verbose=False)
        if context_text
        else []
    )
    utterance_ids = tokenizer.encode(
        record["utterance"], add_special_tokens=False, verbose=False
    )
    pair_specials = tokenizer.num_special_tokens_to_add(pair=bool(context_ids))
    raw_length = len(context_ids) + len(utterance_ids) + pair_specials
    return {
        "raw_token_length": raw_length,
        "context_token_length": len(context_ids),
        "utterance_token_length": len(utterance_ids),
        "truncated": raw_length > max_length,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    model_dir = run_dir / "best_model"
    split_path = run_dir / "split_ids.json"
    if not model_dir.exists():
        raise FileNotFoundError(f"missing best model: {model_dir}")
    if not split_path.exists():
        raise FileNotFoundError(f"missing saved split: {split_path}")

    records = read_jsonl(args.train_file)
    validate_train_records(records)
    records_by_id = {record["id"]: record for record in records}
    split = read_json(split_path)
    validation_ids = split["validation_ids"]
    missing = [record_id for record_id in validation_ids if record_id not in records_by_id]
    if missing:
        raise ValueError(f"validation IDs missing from train file: {missing[:5]}")
    validation_records = [records_by_id[record_id] for record_id in validation_ids]

    preprocess = read_json(model_dir / "preprocess_config.json")
    max_length = int(preprocess.get("max_length", 512))
    threshold_values = read_json(model_dir / "thresholds.json")
    thresholds = np.asarray(
        [float(threshold_values[label]) for label in LABELS], dtype=np.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=True, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.precision == "fp16"
    model.to(device)
    model.eval()

    dataset = IntentDataset(
        validation_records,
        tokenizer=tokenizer,
        max_length=max_length,
        with_labels=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiLabelCollator(tokenizer, pad_to_multiple_of=8),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    logits_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
            logits_parts.append(logits.float().cpu().numpy())
            y_parts.append(labels.numpy())

    logits = np.concatenate(logits_parts)
    y_true = np.concatenate(y_parts).astype(np.int64)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    y_pred = (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    predicted_positive = y_pred.sum(axis=0)

    per_label: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(LABELS):
        tn, fp, fn, tp = [int(value) for value in matrices[index].ravel()]
        row = {
            "label": label,
            "threshold": float(thresholds[index]),
            "support": int(support[index]),
            "predicted_positive": int(predicted_positive[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_positive_rate": safe_ratio(fp, fp + tn),
            "false_negative_rate": safe_ratio(fn, fn + tp),
            "label_accuracy": safe_ratio(tp + tn, tp + tn + fp + fn),
        }
        rows.append(row)
        per_label[label] = row

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    token_info = [raw_token_info(tokenizer, record, max_length) for record in validation_records]
    truncated_mask = np.asarray([item["truncated"] for item in token_info], dtype=bool)
    language_groups = {}
    for language in ["zh", "en"]:
        mask = np.asarray(
            [record["language"] == language for record in validation_records], dtype=bool
        )
        language_groups[language] = score_group(y_true[mask], y_pred[mask])

    length_groups = {
        "truncated": score_group(y_true[truncated_mask], y_pred[truncated_mask]),
        "not_truncated": score_group(y_true[~truncated_mask], y_pred[~truncated_mask]),
    }

    context_groups = {}
    for name, expected_empty in [("empty", True), ("non_empty", False)]:
        mask = np.asarray(
            [(len(record["context"]) == 0) == expected_empty for record in validation_records],
            dtype=bool,
        )
        context_groups[name] = score_group(y_true[mask], y_pred[mask])

    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_label_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    error_examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label_index, label in enumerate(LABELS):
        false_negative_indices = np.where(
            (y_true[:, label_index] == 1) & (y_pred[:, label_index] == 0)
        )[0]
        false_positive_indices = np.where(
            (y_true[:, label_index] == 0) & (y_pred[:, label_index] == 1)
        )[0]
        false_negative_indices = sorted(
            false_negative_indices,
            key=lambda i: probabilities[i, label_index],
        )[: args.num_error_examples]
        false_positive_indices = sorted(
            false_positive_indices,
            key=lambda i: probabilities[i, label_index],
            reverse=True,
        )[: args.num_error_examples]

        def describe(index: int) -> dict[str, Any]:
            record = validation_records[index]
            return {
                "id": record["id"],
                "language": record["language"],
                "probability": float(probabilities[index, label_index]),
                "threshold": float(thresholds[label_index]),
                "true_labels": record["labels"],
                "predicted_labels": [
                    candidate
                    for candidate_index, candidate in enumerate(LABELS)
                    if y_pred[index, candidate_index]
                ],
                "utterance": record["utterance"],
                "context_turns": len(record["context"]),
                **token_info[index],
            }

        error_examples[label] = {
            "false_negatives": [describe(index) for index in false_negative_indices],
            "false_positives": [describe(index) for index in false_positive_indices],
        }

    diagnostics = {
        "model_dir": str(model_dir),
        "validation_samples": len(validation_records),
        "max_length": max_length,
        "overall": {
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
            "exact_match_error_rate": float(1.0 - accuracy_score(y_true, y_pred)),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "hamming_accuracy": float(1.0 - hamming_loss(y_true, y_pred)),
            "true_labels_per_sample": float(y_true.sum(axis=1).mean()),
            "predicted_labels_per_sample": float(y_pred.sum(axis=1).mean()),
        },
        "per_label": per_label,
        "by_language": language_groups,
        "by_input_length": length_groups,
        "by_context": context_groups,
        "raw_token_length": {
            "mean": float(np.mean([item["raw_token_length"] for item in token_info])),
            "max": int(max(item["raw_token_length"] for item in token_info)),
            "truncated_samples": int(truncated_mask.sum()),
            "truncated_rate": float(truncated_mask.mean()),
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "error_examples.json", error_examples)

    print("Overall validation diagnostics")
    for key, value in diagnostics["overall"].items():
        print(f"  {key}: {value:.6f}")
    print("\nPer-label diagnostics")
    print(
        f"{'Label':24} {'Sup':>4} {'Pred':>4} {'Prec':>7} {'Rec':>7} "
        f"{'F1':>7} {'FP':>4} {'FN':>4} {'FPR':>7} {'FNR':>7}"
    )
    for row in rows:
        print(
            f"{row['label']:24} {row['support']:4d} {row['predicted_positive']:4d} "
            f"{row['precision']:7.4f} {row['recall']:7.4f} {row['f1']:7.4f} "
            f"{row['fp']:4d} {row['fn']:4d} "
            f"{row['false_positive_rate']:7.4f} {row['false_negative_rate']:7.4f}"
        )
    print("\nGrouped diagnostics")
    print(f"  by_language: {language_groups}")
    print(f"  by_input_length: {length_groups}")
    print(f"  by_context: {context_groups}")
    print(f"\nSaved detailed files to: {output_dir}")


if __name__ == "__main__":
    main()