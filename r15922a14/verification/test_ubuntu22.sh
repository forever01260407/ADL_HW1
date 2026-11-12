#!/usr/bin/env bash
# Local Linux validation helper. This is separate from the COOL submission ZIP.
set -euo pipefail

SOURCE_DIR="${1:-/mnt/c/Users/yuting/Desktop/HW1_ADL}"
for input in r15922a14.zip context.json test.json; do
  if [[ ! -f "${SOURCE_DIR}/${input}" ]]; then
    printf 'Missing input: %s\n' "${SOURCE_DIR}/${input}" >&2
    exit 1
  fi
done

python3 - <<'PY'
import importlib.metadata as metadata
import sys
import torch
assert sys.version_info[:2] == (3, 10), f"Expected Python 3.10, got {sys.version}"
expected = {"torch": "2.11.0", "transformers": "4.50.0", "numpy": "2.2.6", "scikit-learn": "1.7.2", "gdown": "6.4.0"}
for package, version in expected.items():
    actual = metadata.version(package)
    assert actual.split("+")[0] == version, f"{package}: expected {version}, got {actual}"
assert torch.cuda.is_available(), "CUDA is unavailable in this Python environment"
print("Python and required package versions verified.")
print("CUDA GPU:", torch.cuda.get_device_name(0))
PY

TEST_DIR="$(mktemp -d "${HOME}/adl_hw1_linux_test.XXXXXX")"
printf 'Test directory: %s\n' "${TEST_DIR}"
trap 'result=$?; if [[ "${result}" -ne 0 ]]; then printf "FAILED (exit %s). Logs remain in %s\n" "${result}" "${TEST_DIR}" >&2; fi' EXIT

python3 - "${SOURCE_DIR}/r15922a14.zip" "${TEST_DIR}" <<'PY'
from pathlib import Path, PurePosixPath
import sys
import zipfile
destination = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(sys.argv[1]) as archive:
    assert archive.testzip() is None, "ZIP CRC check failed"
    for name in archive.namelist():
        member = PurePosixPath(name)
        assert member.parts[0] == "r15922a14" and not member.is_absolute() and ".." not in member.parts, f"Unexpected ZIP path: {name}"
    archive.extractall(destination)
print("Actual submission ZIP extracted and verified.")
PY

cd -- "${TEST_DIR}/r15922a14"
bash -n ./download.sh
bash -n ./run.sh
{ time bash ./download.sh; } 2>&1 | tee "${TEST_DIR}/download.log"
printf '%s  %s\n' '06354df9b2e11b5f0c09326f4699ce106395440128304d26b74890310343e5d0' model.tar.gz | sha256sum --check
python3 check_environment.py 2>&1 | tee "${TEST_DIR}/environment.log"
{ time bash ./run.sh "${SOURCE_DIR}/context.json" "${SOURCE_DIR}/test.json" "${TEST_DIR}/prediction.csv"; } 2>&1 | tee "${TEST_DIR}/inference.log"
python3 validate_output.py --test-file "${SOURCE_DIR}/test.json" --prediction-file "${TEST_DIR}/prediction.csv" 2>&1 | tee "${TEST_DIR}/validation.log"
python3 smoke_test_data_joining.py 2>&1 | tee "${TEST_DIR}/data_joining.log"
printf '\nPASS: Linux download, extraction, offline inference, CSV format, and data joining.\n'
printf 'Predictions: %s/prediction.csv\nLogs: %s\n' "${TEST_DIR}" "${TEST_DIR}"
