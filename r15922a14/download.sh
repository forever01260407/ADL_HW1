#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ARCHIVE="${SCRIPT_DIR}/model.tar.gz"

# Fixed Google Drive archive containing the trained model and tokenizer.
MODEL_FILE_ID="1Hb56bCL1iYuIblffYXAyM5bTj81yRqqj"

if [[ -f "${MODEL_ARCHIVE}" ]]; then
  echo "Model archive already exists; skipping download."
  exit 0
fi

if [[ -z "${MODEL_FILE_ID}" ]]; then
  echo "Missing Google Drive model file ID." >&2
  exit 1
fi

# A temporary destination prevents an interrupted download from being reused.
# No extraction, package installation, training, or inference happens here.
python3 -m gdown --no-cookies --timeout 60 --retries 2 \
  "${MODEL_FILE_ID}" --output "${MODEL_ARCHIVE}.part"
mv -- "${MODEL_ARCHIVE}.part" "${MODEL_ARCHIVE}"

