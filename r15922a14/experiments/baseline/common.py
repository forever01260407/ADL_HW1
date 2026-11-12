from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


LABELS = [
    "Ask_Price",
    "Ask_Service",
    "Ask_Spec",
    "Compare_Competitor",
    "Confirm_Order",
    "Doubt",
    "Need_More_Evidence",
    "Need_Time_To_Think",
    "State_Need",
    "Too_Expensive",
]

LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
CSV_HEADER = ["id", *LABELS]


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            records.append(record)
    return records


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def validate_context(context: Any, *, record_id: str) -> list[dict[str, str]]:
    if not isinstance(context, list):
        raise ValueError(f"{record_id}: context must be a list")

    validated: list[dict[str, str]] = []
    for turn_index, turn in enumerate(context):
        if not isinstance(turn, dict):
            raise ValueError(f"{record_id}: context[{turn_index}] must be an object")
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"{record_id}: context[{turn_index}].role must be user or assistant"
            )
        if not isinstance(content, str):
            raise ValueError(f"{record_id}: context[{turn_index}].content must be text")
        validated.append({"role": role, "content": content})
    return validated


def validate_train_records(records: Sequence[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"train record {index}: missing string id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate train id: {record_id}")
        seen_ids.add(record_id)

        if record.get("language") not in {"zh", "en"}:
            raise ValueError(f"{record_id}: language must be zh or en")
        if not isinstance(record.get("utterance"), str) or not record["utterance"].strip():
            raise ValueError(f"{record_id}: utterance must be non-empty text")
        validate_context(record.get("context"), record_id=record_id)

        labels = record.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"{record_id}: labels must be a list")
        unknown = sorted(set(labels) - set(LABELS))
        if unknown:
            raise ValueError(f"{record_id}: unknown labels: {unknown}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"{record_id}: duplicate labels")


def load_test_records(
    context_file: str | Path,
    test_file: str | Path,
) -> list[dict[str, Any]]:
    context_by_id = read_json(context_file)
    test_records = read_json(test_file)

    if not isinstance(context_by_id, dict):
        raise ValueError("context.json must be an object keyed by test id")
    if not isinstance(test_records, list):
        raise ValueError("test.json must be a list")

    seen_ids: set[str] = set()
    joined: list[dict[str, Any]] = []
    for index, record in enumerate(test_records):
        if not isinstance(record, dict):
            raise ValueError(f"test record {index} must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"test record {index}: missing string id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate test id: {record_id}")
        seen_ids.add(record_id)
        if record_id not in context_by_id:
            raise ValueError(f"missing context for test id: {record_id}")
        if record.get("language") not in {"zh", "en"}:
            raise ValueError(f"{record_id}: language must be zh or en")
        if not isinstance(record.get("utterance"), str) or not record["utterance"].strip():
            raise ValueError(f"{record_id}: utterance must be non-empty text")

        joined.append(
            {
                "id": record_id,
                "language": record["language"],
                "context": validate_context(context_by_id[record_id], record_id=record_id),
                "utterance": record["utterance"],
            }
        )

    extra_context_ids = set(context_by_id) - seen_ids
    if extra_context_ids:
        preview = sorted(extra_context_ids)[:5]
        raise ValueError(f"context.json contains IDs not present in test.json: {preview}")
    return joined


def serialize_context(context: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content'].strip()}" for turn in context)


def labels_to_multihot(labels: Iterable[str]) -> list[float]:
    vector = [0.0] * len(LABELS)
    for label in labels:
        vector[LABEL2ID[label]] = 1.0
    return vector


def _head_tail(ids: Sequence[int], budget: int) -> list[int]:
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return list(ids)
    head = budget // 2
    tail = budget - head
    return [*ids[:head], *ids[-tail:]]


def encode_record(
    tokenizer: PreTrainedTokenizerBase,
    context: Sequence[dict[str, str]],
    utterance: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Encode one example while preserving the current utterance and recent context."""
    if max_length < 16:
        raise ValueError("max_length must be at least 16")

    utterance_ids = tokenizer.encode(utterance, add_special_tokens=False)
    context_text = serialize_context(context)
    context_ids = tokenizer.encode(context_text, add_special_tokens=False) if context_text else []

    single_specials = tokenizer.num_special_tokens_to_add(pair=False)
    pair_specials = tokenizer.num_special_tokens_to_add(pair=True)
    single_budget = max_length - single_specials
    pair_budget = max_length - pair_specials
    if single_budget <= 0 or pair_budget <= 0:
        raise ValueError("max_length is too small for tokenizer special tokens")

    # If the utterance itself consumes the full pair budget, omit context and retain
    # both the beginning and end of the current utterance.
    if not context_ids or len(utterance_ids) >= pair_budget:
        utterance_ids = _head_tail(utterance_ids, single_budget)
        encoded = tokenizer.prepare_for_model(
            utterance_ids,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
    else:
        context_budget = pair_budget - len(utterance_ids)
        recent_context_ids = context_ids[-context_budget:] if context_budget > 0 else []
        encoded = tokenizer.prepare_for_model(
            recent_context_ids,
            pair_ids=utterance_ids,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    output = {
        key: value
        for key, value in encoded.items()
        if key in tokenizer.model_input_names and isinstance(value, list)
    }
    if len(output["input_ids"]) > max_length:
        raise AssertionError("encoded input exceeds max_length")
    return output


class IntentDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        with_labels: bool,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded: dict[str, Any] = encode_record(
            self.tokenizer,
            record["context"],
            record["utterance"],
            self.max_length,
        )
        if self.with_labels:
            encoded["labels"] = labels_to_multihot(record["labels"])
        return encoded


class MultiLabelCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self.base = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature.get("labels") for feature in features]
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.base(model_features)
        if labels and labels[0] is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def split_train_validation(
    records: Sequence[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratify approximately by language and each sample's rarest label."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    from sklearn.model_selection import train_test_split

    label_counts = Counter(label for record in records for label in record["labels"])
    strata = []
    for record in records:
        if record["labels"]:
            primary = min(record["labels"], key=lambda label: (label_counts[label], label))
        else:
            primary = "NO_LABEL"
        strata.append(f"{record['language']}|{primary}")

    stratum_counts = Counter(strata)
    can_stratify = all(count >= 2 for count in stratum_counts.values())
    indices = np.arange(len(records))
    try:
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=validation_ratio,
            random_state=seed,
            shuffle=True,
            stratify=strata if can_stratify else None,
        )
    except ValueError:
        languages = [record["language"] for record in records]
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=validation_ratio,
            random_state=seed,
            shuffle=True,
            stratify=languages,
        )

    train_records = [records[int(index)] for index in train_indices]
    validation_records = [records[int(index)] for index in validation_indices]
    return train_records, validation_records


def f1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    per_label_values = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "harmonic_macro_micro": (
            float(2 * macro * micro / (macro + micro)) if macro + micro > 0 else 0.0
        ),
        "per_label_f1": {
            label: float(value) for label, value in zip(LABELS, per_label_values)
        },
    }


def tune_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    candidates: Sequence[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune a global threshold, then one coordinate-descent pass per label."""
    if candidates is None:
        candidates = np.arange(0.10, 0.901, 0.05).tolist()

    best_thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    best_metrics = f1_metrics(y_true, probabilities >= best_thresholds)

    for candidate in candidates:
        thresholds = np.full(len(LABELS), candidate, dtype=np.float32)
        metrics = f1_metrics(y_true, probabilities >= thresholds)
        if metrics["harmonic_macro_micro"] > best_metrics["harmonic_macro_micro"]:
            best_thresholds = thresholds
            best_metrics = metrics

    for label_index in range(len(LABELS)):
        local_threshold = float(best_thresholds[label_index])
        local_metrics = best_metrics
        for candidate in candidates:
            thresholds = best_thresholds.copy()
            thresholds[label_index] = candidate
            metrics = f1_metrics(y_true, probabilities >= thresholds)
            if metrics["harmonic_macro_micro"] > local_metrics["harmonic_macro_micro"]:
                local_threshold = float(candidate)
                local_metrics = metrics
        best_thresholds[label_index] = local_threshold
        best_metrics = local_metrics

    return best_thresholds, best_metrics


def write_prediction_csv(
    output_file: str | Path,
    ids: Sequence[str],
    predictions: np.ndarray,
) -> None:
    if predictions.shape != (len(ids), len(LABELS)):
        raise ValueError(
            f"prediction shape must be {(len(ids), len(LABELS))}, got {predictions.shape}"
        )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for record_id, row in zip(ids, predictions):
            writer.writerow([record_id, *[int(value) for value in row]])


def validate_prediction_csv(
    prediction_file: str | Path,
    expected_ids: Sequence[str],
) -> None:
    with Path(prediction_file).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("prediction CSV is empty") from error
        if header != CSV_HEADER:
            raise ValueError(f"incorrect CSV header: {header}")
        rows = list(reader)

    if len(rows) != len(expected_ids):
        raise ValueError(f"expected {len(expected_ids)} rows, found {len(rows)}")

    actual_ids = [row[0] for row in rows if row]
    if actual_ids != list(expected_ids):
        raise ValueError("prediction IDs are missing, duplicated, or not in test.json order")

    for row_index, row in enumerate(rows, start=2):
        if len(row) != len(CSV_HEADER):
            raise ValueError(f"row {row_index}: expected {len(CSV_HEADER)} columns")
        invalid = [value for value in row[1:] if value not in {"0", "1"}]
        if invalid:
            raise ValueError(f"row {row_index}: predictions must be 0 or 1: {invalid}")

