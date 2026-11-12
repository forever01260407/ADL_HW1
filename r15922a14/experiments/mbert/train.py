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
    PREPROCESS_VERSION,
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

BASELINE_VERSION = "0.2.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a multilingual encoder for ADL HW1"
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="outputs/xlmr_baseline")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Legacy alias: sets split-seed/model-seed unless either is explicit.",
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Fallback LR when a group-specific LR is omitted.",
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=None)
    parser.add_argument("--classifier-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument(
        "--input-order",
        choices=["context-first", "utterance-first"],
        default="context-first",
        help="Which sequence is placed first in the tokenizer pair.",
    )
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=-1,
        help="Maximum context tokens: 0 disables context; -1 uses all remaining space.",
    )
    parser.add_argument(
        "--turn-aware-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pack recent complete dialogue turns instead of an arbitrary token tail.",
    )
    parser.add_argument(
        "--explicit-role-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mark history roles and the current user utterance explicitly.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["fixed_harmonic", "validation_loss"],
        default="fixed_harmonic",
        help="Metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after N non-improving epochs; 0 preserves the old disabled behavior.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
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


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be at least 1")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.context_max_tokens < -1:
        raise ValueError("context-max-tokens must be -1 or greater")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early-stopping-min-delta must be non-negative")

    default_seed = args.seed if args.seed is not None else 42
    args.split_seed = args.split_seed if args.split_seed is not None else default_seed
    args.model_seed = args.model_seed if args.model_seed is not None else default_seed
    if args.seed is not None:
        print(
            "Compatibility note: --seed is a legacy alias. "
            f"Resolved split_seed={args.split_seed}, model_seed={args.model_seed}."
        )
    args.input_order = args.input_order.replace("-", "_")
    args.context_max_tokens = (
        None if args.context_max_tokens == -1 else args.context_max_tokens
    )
    args.encoder_learning_rate = (
        args.encoder_learning_rate
        if args.encoder_learning_rate is not None
        else args.learning_rate
    )
    args.classifier_learning_rate = (
        args.classifier_learning_rate
        if args.classifier_learning_rate is not None
        else args.learning_rate
    )
    return args


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
    return torch.tensor(np.sqrt(negatives / positives), dtype=torch.float32)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    loss_fn: BCEWithLogitsLoss,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
                loss = loss_fn(logits, labels)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        total_loss / max(1, total_examples),
    )


def build_optimizer(
    model: torch.nn.Module,
    encoder_learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
) -> tuple[AdamW, list[dict[str, Any]]]:
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
        ("encoder", True): [],
        ("encoder", False): [],
        ("classifier", True): [],
        ("classifier", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = (
            "classifier"
            if name.startswith("classifier.") or name.startswith("score.")
            else "encoder"
        )
        apply_decay = not (parameter.ndim == 1 or name.endswith(".bias"))
        grouped[(component, apply_decay)].append(parameter)

    optimizer_groups: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []
    seen_parameter_ids: list[int] = []
    for (component, apply_decay), parameters in grouped.items():
        if not parameters:
            continue
        learning_rate = (
            classifier_learning_rate
            if component == "classifier"
            else encoder_learning_rate
        )
        group_weight_decay = weight_decay if apply_decay else 0.0
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": group_weight_decay,
            }
        )
        seen_parameter_ids.extend(id(parameter) for parameter in parameters)
        group_summary.append(
            {
                "component": component,
                "weight_decay": group_weight_decay,
                "learning_rate": learning_rate,
                "parameter_tensors": len(parameters),
                "parameters": int(sum(parameter.numel() for parameter in parameters)),
            }
        )

    expected_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(seen_parameter_ids) != len(set(seen_parameter_ids)):
        raise AssertionError("a trainable parameter appears in multiple optimizer groups")
    if set(seen_parameter_ids) != set(expected_ids):
        raise AssertionError("optimizer groups omit one or more trainable parameters")
    return AdamW(optimizer_groups), group_summary


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    training_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    write_json(output_dir / "labels.json", LABELS)
    write_json(output_dir / "preprocess_config.json", training_config)


def selection_value(
    checkpoint_metric: str,
    validation_loss: float,
    fixed_metrics: dict[str, Any],
) -> float:
    if checkpoint_metric == "validation_loss":
        return validation_loss
    return float(fixed_metrics["harmonic_macro_micro"])


def is_improvement(
    checkpoint_metric: str,
    current: float,
    best: float | None,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if checkpoint_metric == "validation_loss":
        return current < best - min_delta
    return current > best + min_delta


def main() -> None:
    args = resolve_args(parse_args())
    output_dir = Path(args.output_dir).resolve()
    best_model_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.train_file)
    validate_train_records(records)
    train_records, validation_records = split_train_validation(
        records,
        validation_ratio=args.validation_ratio,
        seed=args.split_seed,
    )
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_validation_samples is not None:
        validation_records = validation_records[: args.max_validation_samples]
    if not train_records or not validation_records:
        raise ValueError("training and validation splits must both be non-empty")

    split_info = {
        "seed": args.split_seed,
        "split_seed": args.split_seed,
        "validation_ratio": args.validation_ratio,
        "train_ids": [record["id"] for record in train_records],
        "validation_ids": [record["id"] for record in validation_records],
    }
    write_json(output_dir / "split_ids.json", split_info)

    set_seed(args.model_seed)
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Seeds: split={args.split_seed}; model={args.model_seed}")
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
    print(
        "Preprocess: "
        f"order={args.input_order}; context_max_tokens={args.context_max_tokens}; "
        f"turn_aware={args.turn_aware_context}; markers={args.explicit_role_markers}"
    )

    dataset_options = {
        "input_order": args.input_order,
        "context_max_tokens": args.context_max_tokens,
        "turn_aware_context": args.turn_aware_context,
        "explicit_role_markers": args.explicit_role_markers,
    }
    collator = MultiLabelCollator(tokenizer, pad_to_multiple_of=8)
    train_dataset = IntentDataset(
        train_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        with_labels=True,
        **dataset_options,
    )
    validation_dataset = IntentDataset(
        validation_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        with_labels=True,
        **dataset_options,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.model_seed)
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

    optimizer, optimizer_summary = build_optimizer(
        model,
        encoder_learning_rate=args.encoder_learning_rate,
        classifier_learning_rate=args.classifier_learning_rate,
        weight_decay=args.weight_decay,
    )
    print("Optimizer groups:")
    for group in optimizer_summary:
        print(
            f"  {group['component']}: lr={group['learning_rate']:.2e}; "
            f"weight_decay={group['weight_decay']}; params={group['parameters']:,}"
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

    packing_name = "recent-complete-turns" if args.turn_aware_context else "legacy-token-tail"
    base_training_config = {
        "baseline_version": BASELINE_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "base_model": args.model_name,
        "max_length": args.max_length,
        "input_order": args.input_order,
        "context_max_tokens": args.context_max_tokens,
        "turn_aware_context": args.turn_aware_context,
        "explicit_role_markers": args.explicit_role_markers,
        "context_packing": packing_name,
        "truncation": "preserve current utterance; context uses configured budget",
        "label_order": LABELS,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "validation_ratio": args.validation_ratio,
        "precision": args.precision,
        "use_pos_weight": args.use_pos_weight,
        "checkpoint_metric": args.checkpoint_metric,
        "encoder_learning_rate": args.encoder_learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
    }

    history: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_started = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
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

        logits, y_true, validation_loss = evaluate(
            model,
            validation_loader,
            device,
            use_amp,
            amp_dtype,
            loss_fn,
        )
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        fixed_thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
        fixed_metrics = f1_metrics(y_true, probabilities >= fixed_thresholds)
        current_value = selection_value(
            args.checkpoint_metric,
            validation_loss,
            fixed_metrics,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, len(train_loader)),
            "validation_loss": validation_loss,
            "seconds": time.perf_counter() - epoch_started,
            "fixed_threshold_metrics": fixed_metrics,
            "checkpoint_metric": args.checkpoint_metric,
            "selection_value": current_value,
        }
        history.append(epoch_record)
        write_json(output_dir / "metrics_history.json", history)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={epoch_record['train_loss']:.5f} | "
            f"val_loss={validation_loss:.5f} | "
            f"fixed_macro={fixed_metrics['macro_f1']:.5f} | "
            f"fixed_micro={fixed_metrics['micro_f1']:.5f} | "
            f"seconds={epoch_record['seconds']:.1f}"
        )

        improved = is_improvement(
            args.checkpoint_metric,
            current_value,
            best_value,
            args.early_stopping_min_delta,
        )
        if improved:
            best_value = current_value
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_config = {
                **base_training_config,
                "best_epoch": best_epoch,
                "best_selection_value": best_value,
                "threshold_calibration": "pending until checkpoint selection completes",
            }
            save_checkpoint(model, tokenizer, best_model_dir, checkpoint_config)
            print(f"Saved new best checkpoint to {best_model_dir}")
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    "Early stopping: "
                    f"no {args.checkpoint_metric} improvement for "
                    f"{epochs_without_improvement} epoch(s)."
                )
                break

    if best_epoch == 0 or best_value is None:
        raise RuntimeError("training finished without a saved checkpoint")

    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_dir,
        local_files_only=True,
    ).to(device)
    best_logits, y_true, best_validation_loss = evaluate(
        best_model,
        validation_loader,
        device,
        use_amp,
        amp_dtype,
        loss_fn,
    )
    best_probabilities = torch.sigmoid(torch.from_numpy(best_logits)).numpy()
    fixed_thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    best_fixed_metrics = f1_metrics(y_true, best_probabilities >= fixed_thresholds)
    tuned_thresholds, tuned_metrics = tune_thresholds(y_true, best_probabilities)
    final_metrics = {
        **tuned_metrics,
        "best_epoch": best_epoch,
        "checkpoint_metric": args.checkpoint_metric,
        "best_selection_value": best_value,
        "validation_loss": best_validation_loss,
        "fixed_threshold_metrics": best_fixed_metrics,
        "tuned_threshold_metrics": tuned_metrics,
        "calibration_note": (
            "Thresholds were tuned once after reloading the selected checkpoint. "
            "Checkpoint selection and calibration still share the validation split."
        ),
    }
    write_json(
        best_model_dir / "thresholds.json",
        {label: float(value) for label, value in zip(LABELS, tuned_thresholds)},
    )
    write_json(best_model_dir / "validation_metrics.json", final_metrics)
    write_json(
        best_model_dir / "preprocess_config.json",
        {
            **base_training_config,
            "best_epoch": best_epoch,
            "best_selection_value": best_value,
            "threshold_calibration": "once after reloading selected checkpoint",
        },
    )

    total_seconds = time.perf_counter() - training_started
    write_json(
        output_dir / "run_summary.json",
        {
            "baseline_version": BASELINE_VERSION,
            "best_epoch": best_epoch,
            "checkpoint_metric": args.checkpoint_metric,
            "best_selection_value": best_value,
            "best_harmonic_macro_micro": tuned_metrics["harmonic_macro_micro"],
            "training_seconds": total_seconds,
            "epochs_completed": len(history),
            "optimizer_groups": optimizer_summary,
            "arguments": vars(args),
        },
    )
    print(
        "Final calibrated metrics | "
        f"macro={tuned_metrics['macro_f1']:.5f} | "
        f"micro={tuned_metrics['micro_f1']:.5f} | "
        f"best_epoch={best_epoch}"
    )
    print(f"Training complete in {total_seconds / 60:.1f} minutes")
    print(f"Best offline model: {best_model_dir}")


if __name__ == "__main__":
    main()
