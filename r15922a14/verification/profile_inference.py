"""Profile the unchanged CLI entry point in the local target-package environment."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

started = time.perf_counter()
import torch

if torch.cuda.is_available():
    # Restrict the allocator to 9 GiB on the local 24 GiB GPU.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 9 * 1024**3 / total))
    torch.cuda.reset_peak_memory_stats()

sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name="__main__")
import transformers
result = {
    "seconds_including_python_imports": time.perf_counter() - started,
    "python": sys.version,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "allocator_limit_gib": 9,
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
    "argv": sys.argv,
}
destination = Path(os.environ["ADL_PROFILE_OUTPUT"])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("PROFILE", json.dumps(result, indent=2))
