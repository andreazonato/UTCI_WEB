#!/usr/bin/env bash
set -euo pipefail

SOLWEIG_DIR="${SOLWEIG_DIR:-api_data}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
PYTHON_BIN="${PYTHON_BIN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -d "${REPO_ROOT}/${SOLWEIG_DIR}" && ! -d "${SOLWEIG_DIR}" ]]; then
  echo "Popup API data directory not found: ${SOLWEIG_DIR}" >&2
  echo "Set SOLWEIG_DIR to the folder containing <city>/<run>/... tif files." >&2
  exit 1
fi

if [[ -d "${REPO_ROOT}/${SOLWEIG_DIR}" ]]; then
  SOLWEIG_DIR="${REPO_ROOT}/${SOLWEIG_DIR}"
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
exec "${PYTHON_BIN}" -m utci_core.raster_click_server --solweig-dir "${SOLWEIG_DIR}" --host "${HOST}" --port "${PORT}"
