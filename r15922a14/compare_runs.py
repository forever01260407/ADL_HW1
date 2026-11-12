from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FOCUS_LABELS = ["Compare_Competitor", "Too_Expensive"]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_final_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "best_model" / "validation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing validation metrics: {metrics_path}")
    metrics = read_json(metrics_path)
    tuned = metrics.get("tuned_threshold_metrics", metrics)
    return {
        "macro": float(tuned["macro_f1"]),
        "micro": float(tuned["micro_f1"]),
        "harmonic": float(tuned["harmonic_macro_micro"]),
        "best_epoch": metrics.get("best_epoch", "legacy"),
    }


def get_focus_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    diagnostics_path = run_dir / "analysis" / "diagnostics.json"
    if not diagnostics_path.exists():
        return {}
    per_label = read_json(diagnostics_path).get("per_label", {})
    return {
        label: {
            "precision": float(per_label[label]["precision"]),
            "recall": float(per_label[label]["recall"]),
            "f1": float(per_label[label]["f1"]),
        }
        for label in FOCUS_LABELS
        if label in per_label
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare calibrated metrics from multiple ADL HW1 runs"
    )
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = []
    for value in args.run_dirs:
        run_dir = Path(value).resolve()
        final = get_final_metrics(run_dir)
        rows.append((run_dir.name, final, get_focus_metrics(run_dir)))

    print(
        f"{'Run':28} {'Epoch':>7} {'Macro':>8} {'Micro':>8} {'Harmonic':>9} "
        f"{'Compare F1':>11} {'TooExp F1':>10}"
    )
    for name, final, focus in rows:
        compare = focus.get("Compare_Competitor", {}).get("f1")
        too_expensive = focus.get("Too_Expensive", {}).get("f1")
        compare_text = f"{compare:.5f}" if compare is not None else "run analysis"
        too_text = f"{too_expensive:.5f}" if too_expensive is not None else "analysis"
        print(
            f"{name[:28]:28} {str(final['best_epoch']):>7} "
            f"{final['macro']:8.5f} {final['micro']:8.5f} "
            f"{final['harmonic']:9.5f} {compare_text:>11} {too_text:>10}"
        )

    if any(not focus for _, _, focus in rows):
        print(
            "\nFocus-label scores require analyze_validation.py for each run. "
            "Aggregate calibrated scores are available now."
        )


if __name__ == "__main__":
    main()
