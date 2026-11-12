#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
MODEL_ARCHIVE="${SCRIPT_DIR}/model.tar.gz"
MODEL_SHA256="06354df9b2e11b5f0c09326f4699ce106395440128304d26b74890310343e5d0"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  if [[ -f "${MODEL_ARCHIVE}" ]]; then
    printf '%s  %s\n' "${MODEL_SHA256}" "${MODEL_ARCHIVE}" | sha256sum --check --status
    tar -xzf "${MODEL_ARCHIVE}" -C "${SCRIPT_DIR}"
  else
    echo "Missing ${MODEL_DIR}/config.json and ${MODEL_ARCHIVE}" >&2
    exit 1
  fi
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python3 "${SCRIPT_DIR}/inference.py" \
  --context-file "$1" \
  --test-file "$2" \
  --model-dir "${MODEL_DIR}" \
  --output-file "$3" \
  --batch-size 4 \
  --precision fp16

