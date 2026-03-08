#!/usr/bin/env bash
set -euo pipefail

SOLWEIG_DIR="${SOLWEIG_DIR:-api_data}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
PYTHON_BIN="${PYTHON_BIN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${REPO_ROOT}/${SOLWEIG_DIR}" ]]; then
  SOLWEIG_DIR="${REPO_ROOT}/${SOLWEIG_DIR}"
elif [[ -d "${SOLWEIG_DIR}" ]]; then
  :
else
  # Keep Render deploy healthy even before first data sync.
  if [[ "${SOLWEIG_DIR}" == /* ]]; then
    mkdir -p "${SOLWEIG_DIR}"
  else
    SOLWEIG_DIR="${REPO_ROOT}/${SOLWEIG_DIR}"
    mkdir -p "${SOLWEIG_DIR}"
  fi
  echo "[popup-api] warning: created empty SOLWEIG_DIR=${SOLWEIG_DIR}" >&2
  echo "[popup-api] warning: API will return 404 until api_data is published." >&2
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python interpreter not found (python/python3)." >&2
    exit 1
  fi
fi

echo "[popup-api] SOLWEIG_DIR=${SOLWEIG_DIR}"
echo "[popup-api] listening on ${HOST}:${PORT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Best-effort LFS sync on startup (useful on platforms where build-time LFS pull
# was skipped or stale).
if git lfs version >/dev/null 2>&1; then
  git lfs install --local >/dev/null 2>&1 || true
  git lfs pull >/dev/null 2>&1 || echo "[popup-api] warning: git lfs pull failed at startup." >&2
else
  echo "[popup-api] warning: git-lfs not available at startup." >&2
fi
exec "${PYTHON_BIN}" -m utci_core.raster_click_server --solweig-dir "${SOLWEIG_DIR}" --host "${HOST}" --port "${PORT}"
