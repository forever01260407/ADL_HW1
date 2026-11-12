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
    resolve_preprocess_config,
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


def raw_token_info(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int,
    preprocess_options: dict[str, Any],
) -> dict[str, Any]:
    explicit_markers = preprocess_options["explicit_role_markers"]
    context_text = serialize_context(
        record["context"],
        explicit_role_markers=explicit_markers,
    )
    context_ids = (
        tokenizer.encode(context_text, add_special_tokens=False, verbose=False)
        if context_text
        else []
    )
    utterance_text = (
        f"current user: {record['utterance'].strip()}"
        if explicit_markers
        else record["utterance"]
    )
    utterance_ids = tokenizer.encode(
        utterance_text, add_special_tokens=False, verbose=False
    )
    pair_specials = tokenizer.num_special_tokens_to_add(pair=bool(context_ids))
    raw_length = len(context_ids) + len(utterance_ids) + pair_specials
    available_context_tokens = max(
        0,
        max_length
        - tokenizer.num_special_tokens_to_add(pair=True)
        - len(utterance_ids),
    )
    configured_budget = preprocess_options["context_max_tokens"]
    effective_context_budget = (
        available_context_tokens
        if configured_budget is None
        else min(available_context_tokens, configured_budget)
    )
    context_budget_truncated = len(context_ids) > effective_context_budget
    utterance_truncated = (
        len(utterance_ids) + tokenizer.num_special_tokens_to_add(pair=False)
        > max_length
    )
    return {
        "raw_token_length": raw_length,
        "context_token_length": len(context_ids),
        "utterance_token_length": len(utterance_ids),
        "source_over_model_limit": raw_length > max_length,
        "context_budget_truncated": context_budget_truncated,
        "utterance_truncated": utterance_truncated,
        "effective_context_budget": effective_context_budget,
        # Legacy alias retained so older analysis consumers still work.
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

    preprocess_path = model_dir / "preprocess_config.json"
    preprocess = read_json(preprocess_path) if preprocess_path.exists() else {}
    max_length = int(preprocess.get("max_length", 512))
    preprocess_options = resolve_preprocess_config(preprocess)
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

    token_info = [
        raw_token_info(tokenizer, record, max_length, preprocess_options)
        for record in validation_records
    ]
    over_limit_mask = np.asarray(
        [item["source_over_model_limit"] for item in token_info], dtype=bool
    )
    budget_truncated_mask = np.asarray(
        [item["context_budget_truncated"] for item in token_info], dtype=bool
    )
    language_groups = {}
    for language in ["zh", "en"]:
        mask = np.asarray(
            [record["language"] == language for record in validation_records], dtype=bool
        )
        language_groups[language] = score_group(y_true[mask], y_pred[mask])

    length_groups = {
        "source_over_model_limit": score_group(
            y_true[over_limit_mask], y_pred[over_limit_mask]
        ),
        "source_within_model_limit": score_group(
            y_true[~over_limit_mask], y_pred[~over_limit_mask]
        ),
    }
    context_budget_groups = {
        "context_truncated": score_group(
            y_true[budget_truncated_mask], y_pred[budget_truncated_mask]
        ),
        "context_not_truncated": score_group(
            y_true[~budget_truncated_mask], y_pred[~budget_truncated_mask]
        ),
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
        "preprocess": preprocess_options,
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
        "by_context_budget": context_budget_groups,
        "by_context": context_groups,
        "raw_token_length": {
            "mean": float(np.mean([item["raw_token_length"] for item in token_info])),
            "max": int(max(item["raw_token_length"] for item in token_info)),
            "source_over_model_limit_samples": int(over_limit_mask.sum()),
            "source_over_model_limit_rate": float(over_limit_mask.mean()),
            "context_budget_truncated_samples": int(budget_truncated_mask.sum()),
            "context_budget_truncated_rate": float(budget_truncated_mask.mean()),
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
    print(f"  by_context_budget: {context_budget_groups}")
    print(f"  by_context: {context_groups}")
    print(f"\nSaved detailed files to: {output_dir}")


if __name__ == "__main__":
    main()

