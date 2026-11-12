from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    IntentDataset,
    MultiLabelCollator,
    f1_metrics,
    read_jsonl,
    set_seed,
    split_train_validation,
    tune_thresholds,
    validate_train_records,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XLM-R for ADL HW1")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="outputs/xlmr_baseline")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="For smoke tests only; limits training records after splitting.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="For smoke tests only; limits validation records after splitting.",
    )
    return parser.parse_args()


def resolve_precision(
    requested: str,
    device: torch.device,
) -> tuple[bool, torch.dtype | None, bool]:
    if device.type != "cuda" or requested == "fp32":
        if requested != "fp32" and device.type != "cuda":
            print("CUDA is unavailable; falling back to fp32.")
        return False, None, False
    if requested == "fp16":
        return True, torch.float16, True
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but this GPU/PyTorch build does not support it")
        return True, torch.bfloat16, False
    return False, None, False


def compute_pos_weight(records: list[dict[str, Any]]) -> torch.Tensor:
    positives = np.zeros(len(LABELS), dtype=np.float64)
    for record in records:
        for label in record["labels"]:
            positives[LABEL2ID[label]] += 1
    negatives = len(records) - positives
    if np.any(positives == 0):
        missing = [LABELS[index] for index in np.where(positives == 0)[0]]
        raise ValueError(f"cannot compute pos_weight; no positive examples for {missing}")
    # Square root dampens extreme weights for the rare labels.
    return torch.tensor(np.sqrt(negatives / positives), dtype=torch.float32)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def save_best_model(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    thresholds: np.ndarray,
    training_config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    write_json(output_dir / "labels.json", LABELS)
    write_json(
        output_dir / "thresholds.json",
        {label: float(value) for label, value in zip(LABELS, thresholds)},
    )
    write_json(output_dir / "preprocess_config.json", training_config)
    write_json(output_dir / "validation_metrics.json", metrics)


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be at least 1")

    output_dir = Path(args.output_dir).resolve()
    best_model_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    records = read_jsonl(args.train_file)
    validate_train_records(records)
    train_records, validation_records = split_train_validation(
        records,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_validation_samples is not None:
        validation_records = validation_records[: args.max_validation_samples]
    if not train_records or not validation_records:
        raise ValueError("training and validation splits must both be non-empty")

    split_info = {
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "train_ids": [record["id"] for record in train_records],
        "validation_ids": [record["id"] for record in validation_records],
    }
    write_json(output_dir / "split_ids.json", split_info)

    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for the final baseline")

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    print("Loading 10-label sequence classifier...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp, amp_dtype, use_scaler = resolve_precision(args.precision, device)
    print(f"Device: {device}; precision: {args.precision if use_amp else 'fp32'}")

    collator = MultiLabelCollator(tokenizer, pad_to_multiple_of=8)
    train_dataset = IntentDataset(
        train_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        with_labels=True,
    )
    validation_dataset = IntentDataset(
        validation_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        with_labels=True,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    pos_weight = None
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(train_records).to(device)
        print("Using sqrt(negative/positive) pos_weight:")
        print({label: round(float(value), 4) for label, value in zip(LABELS, pos_weight)})
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_started = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
                raw_loss = loss_fn(logits, labels)
                loss = raw_loss / args.gradient_accumulation

            scaler.scale(loss).backward()
            epoch_loss += float(raw_loss.detach().cpu())

            should_update = (
                step % args.gradient_accumulation == 0 or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        logits, y_true = evaluate(model, validation_loader, device, use_amp, amp_dtype)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        default_thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
        default_metrics = f1_metrics(y_true, probabilities >= default_thresholds)
        tuned_thresholds, tuned_metrics = tune_thresholds(y_true, probabilities)

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, len(train_loader)),
            "seconds": time.perf_counter() - epoch_started,
            "default_threshold_metrics": default_metrics,
            "tuned_threshold_metrics": tuned_metrics,
            "thresholds": {
                label: float(value) for label, value in zip(LABELS, tuned_thresholds)
            },
        }
        history.append(epoch_record)
        write_json(output_dir / "metrics_history.json", history)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={epoch_record['train_loss']:.5f} | "
            f"macro={tuned_metrics['macro_f1']:.5f} | "
            f"micro={tuned_metrics['micro_f1']:.5f} | "
            f"seconds={epoch_record['seconds']:.1f}"
        )

        score = tuned_metrics["harmonic_macro_micro"]
        if score > best_score:
            best_score = score
            training_config = {
                "base_model": args.model_name,
                "max_length": args.max_length,
                "context_serialization": "one turn per line as '<role>: <content>'",
                "truncation": "preserve utterance; keep most recent context tokens",
                "label_order": LABELS,
                "seed": args.seed,
                "validation_ratio": args.validation_ratio,
                "best_epoch": epoch,
                "precision": args.precision,
                "use_pos_weight": args.use_pos_weight,
            }
            save_best_model(
                model,
                tokenizer,
                best_model_dir,
                tuned_thresholds,
                training_config,
                tuned_metrics,
            )
            print(f"Saved new best model to {best_model_dir}")

    total_seconds = time.perf_counter() - training_started
    write_json(
        output_dir / "run_summary.json",
        {
            "best_harmonic_macro_micro": best_score,
            "training_seconds": total_seconds,
            "arguments": vars(args),
        },
    )
    print(f"Training complete in {total_seconds / 60:.1f} minutes")
    print(f"Best offline model: {best_model_dir}")


if __name__ == "__main__":
    main()

