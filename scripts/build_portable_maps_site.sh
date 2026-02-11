#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a deploy-ready UTCI interactive map bundle and (optionally) stage popup API data.

Usage:
  scripts/build_portable_maps_site.sh \
    --city Trento \
    --run-period 2026_02_11_6__2026_02_14_6 \
    [--utci-root /home/user/UTCI] \
    [--site-root /path/to/repo/site] \
    [--python-bin /path/to/python] \
    [--api-url https://utci-popup-api.onrender.com] \
    [--hourly-vars UTCI] \
    [--render-max-dim 900] \
    [--stats-max-dim 300] \
    [--overwrite-png] \
    [--copy-api-data] \
    [--api-data-root /path/to/repo/api_data]

Notes:
  - Output bundle is written to: <site-root>/<city>/<run-period>/
  - Popup API data copy is optional and can be large.
EOF
}

CITY=""
RUN_PERIOD=""
UTCI_ROOT="${UTCI_ROOT:-$HOME/UTCI}"
SITE_ROOT=""
PYTHON_BIN="${PYTHON_BIN:-}"
API_URL="${API_URL:-https://utci-popup-api.onrender.com}"
HOURLY_VARS="${HOURLY_VARS:-UTCI}"
RENDER_MAX_DIM="${RENDER_MAX_DIM:-900}"
STATS_MAX_DIM="${STATS_MAX_DIM:-300}"
OVERWRITE_PNG=0
COPY_API_DATA=0
API_DATA_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --city)
      CITY="${2:-}"
      shift 2
      ;;
    --run-period)
      RUN_PERIOD="${2:-}"
      shift 2
      ;;
    --utci-root)
      UTCI_ROOT="${2:-}"
      shift 2
      ;;
    --site-root)
      SITE_ROOT="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --api-url)
      API_URL="${2:-}"
      shift 2
      ;;
    --hourly-vars)
      HOURLY_VARS="${2:-}"
      shift 2
      ;;
    --render-max-dim)
      RENDER_MAX_DIM="${2:-}"
      shift 2
      ;;
    --stats-max-dim)
      STATS_MAX_DIM="${2:-}"
      shift 2
      ;;
    --overwrite-png)
      OVERWRITE_PNG=1
      shift
      ;;
    --copy-api-data)
      COPY_API_DATA=1
      shift
      ;;
    --api-data-root)
      API_DATA_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${CITY}" || -z "${RUN_PERIOD}" ]]; then
  echo "Both --city and --run-period are required." >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${SITE_ROOT}" ]]; then
  SITE_ROOT="${REPO_ROOT}/site"
fi
if [[ -z "${API_DATA_ROOT}" ]]; then
  API_DATA_ROOT="${REPO_ROOT}/api_data"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "$HOME/.conda/envs/utci_forecast/bin/python" ]]; then
    PYTHON_BIN="$HOME/.conda/envs/utci_forecast/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

RUN_DIR="${UTCI_ROOT}/runs/${CITY}/${RUN_PERIOD}"
OUT_DIR="${SITE_ROOT}/${CITY}/${RUN_PERIOD}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
# Rebuild from scratch to avoid stale PNG folders from old var selections.
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

CMD=(
  "${PYTHON_BIN}" -m utci_core.interactive_maps
  --solweig-dir "${RUN_DIR}"
  --output-dir "${OUT_DIR}"
  --offline
  --raster-click-mode server
  --raster-click-server-url "${API_URL}"
  --raster-click-folder "${CITY}/${RUN_PERIOD}"
  --hourly-vars "${HOURLY_VARS}"
  --render-max-dim "${RENDER_MAX_DIM}"
  --stats-max-dim "${STATS_MAX_DIM}"
)
if [[ "${OVERWRITE_PNG}" -eq 1 ]]; then
  CMD+=(--overwrite-png)
fi

echo "[build] city=${CITY} run=${RUN_PERIOD}"
echo "[build] run_dir=${RUN_DIR}"
echo "[build] out_dir=${OUT_DIR}"
echo "[build] python=${PYTHON_BIN}"
echo "[build] api_url=${API_URL}"
echo "[build] raster_click_folder=${CITY}/${RUN_PERIOD}"
echo "[build] hourly_vars=${HOURLY_VARS}"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${CMD[@]}"

if [[ "${COPY_API_DATA}" -eq 1 ]]; then
  API_DST="${API_DATA_ROOT}/${CITY}/${RUN_PERIOD}"
  rm -rf "${API_DST}"
  mkdir -p "${API_DST}"
  echo "[api-data] staging GeoTIFFs -> ${API_DST}"
  readarray -t _VARS < <(echo "${HOURLY_VARS}" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF')
  INCLUDE_ARGS=(--include '*/' --include 'SVF_static.tif')
  for v in "${_VARS[@]}"; do
    INCLUDE_ARGS+=(--include "${v}_*.tif")
  done
  rsync -a --prune-empty-dirs \
    "${INCLUDE_ARGS[@]}" \
    --exclude '*' \
    "${RUN_DIR}/" "${API_DST}/"
fi

mkdir -p "${SITE_ROOT}"
touch "${SITE_ROOT}/.nojekyll"

INDEX_FILE="${SITE_ROOT}/index.html"
{
  cat <<'HTML_HEAD'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UTCI Interactive Maps</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; }
    h1 { margin-bottom: 0.2rem; }
    .muted { color: #666; margin-bottom: 1rem; }
    li { margin: 0.4rem 0; }
    code { background: #f2f2f2; padding: 0.1rem 0.3rem; border-radius: 3px; }
  </style>
</head>
<body>
  <h1>UTCI Interactive Maps</h1>
  <div class="muted">Auto-generated index of published map bundles.</div>
  <ul>
HTML_HEAD
  while IFS= read -r html; do
    rel="${html#${SITE_ROOT}/}"
    city="$(echo "${rel}" | cut -d/ -f1)"
    run="$(echo "${rel}" | cut -d/ -f2)"
    href="${city}/${run}/maps/interactive_map_raster.html"
    printf '    <li><a href="%s">%s - %s</a></li>\n' "${href}" "${city}" "${run}"
  done < <(find "${SITE_ROOT}" -type f -path '*/maps/interactive_map_raster.html' | sort)
  cat <<'HTML_TAIL'
  </ul>
</body>
</html>
HTML_TAIL
} > "${INDEX_FILE}"

echo "[done] map html: ${OUT_DIR}/maps/interactive_map_raster.html"
echo "[done] site index: ${INDEX_FILE}"
