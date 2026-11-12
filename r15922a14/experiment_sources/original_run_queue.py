import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


PROJECT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(__file__).resolve().parent
RUNS = [
    ("v02_ctx256_weighted_s17", 17, 256),
    ("v02_ctx128_weighted_s87", 87, 128),
    ("v02_ctx256_weighted_s87", 87, 256),
]
COMPARE_RUNS = [
    "v02_ctx128_weighted", "v02_ctx128_weighted_s17",
    "v02_ctx128_weighted_s87", "v02_ctx256_weighted",
    "v02_ctx256_weighted_s17", "v02_ctx256_weighted_s87",
]
ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")


def execute(name, arguments):
    command = [sys.executable, "-u", *arguments]
    print(f"\nSTART {name}: {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    with (LOG_DIR / f"{name}.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=PROJECT, env=ENV, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}; remaining commands were not started")
    print(f"DONE {name}", flush=True)


def main():
    prerequisite = PROJECT / "outputs/v02_ctx128_weighted_s17"
    for relative in ["run_summary.json", "best_model/validation_metrics.json", "best_model/thresholds.json"]:
        if not (prerequisite / relative).is_file():
            raise RuntimeError(f"Previous run is incomplete: {relative}")
    for name, _, _ in RUNS:
        if (PROJECT / "outputs" / name).exists():
            raise RuntimeError(f"Refusing to overwrite an existing run: {name}")
    for name, seed, context in RUNS:
        execute(name, [
            "train.py", "--train-file", "..\\train.jsonl",
            "--output-dir", f"outputs\\{name}", "--epochs", "12",
            "--gradient-accumulation", "1", "--split-seed", "42",
            "--model-seed", str(seed), "--use-pos-weight",
            "--input-order", "utterance-first", "--context-max-tokens", str(context),
            "--turn-aware-context", "--explicit-role-markers",
            "--early-stopping-patience", "3",
        ])
        summary = PROJECT / "outputs" / name / "run_summary.json"
        metrics = PROJECT / "outputs" / name / "best_model/validation_metrics.json"
        json.loads(summary.read_text(encoding="utf-8"))
        json.loads(metrics.read_text(encoding="utf-8"))
    execute("compare_runs", ["compare_runs.py", *[f"outputs\\{name}" for name in COMPARE_RUNS]])
    print("ALL REQUESTED COMMANDS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
