#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-server}"
REQ_FILE="${REQ_FILE:-requirements/requirements-server-py312-cu130.txt}"

if [[ "${REQ_FILE}" = /* ]]; then
  REQ_PATH="${REQ_FILE}"
else
  REQ_PATH="${REPO_ROOT}/${REQ_FILE}"
fi

echo "[1/4] Creating virtualenv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[2/4] Activating virtualenv"
source "${VENV_DIR}/bin/activate"

echo "[3/4] Installing dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_PATH}"

echo "[4/4] Verifying torch/cuda"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

echo "Done."
