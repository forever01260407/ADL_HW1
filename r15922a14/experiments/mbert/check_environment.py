from __future__ import annotations

import importlib.metadata
import platform
import sys

import torch
import transformers


EXPECTED = {
    "Python": "3.10",
    "torch": "2.11.0",
    "transformers": "4.50.0",
}


def optional_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    versions = {
        "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": optional_version("tokenizers"),
        "huggingface-hub": optional_version("huggingface-hub"),
        "sentencepiece": optional_version("sentencepiece"),
        "safetensors": optional_version("safetensors"),
    }
    print(f"OS: {platform.platform()}")
    for name, version in versions.items():
        expected = EXPECTED.get(name)
        suffix = f" (TA target: {expected})" if expected else ""
        print(f"{name}: {version}{suffix}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
        )


if __name__ == "__main__":
    main()

