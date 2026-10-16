#!/usr/bin/env bash
# Asculto quick setup: create the venv, install dependencies, and download
# the HLS-CMDS dataset from its Zenodo source if it is not already present.
#
# Usage:
#   ./setup.sh                      # local data/ + defaults
#   ASCULTO_DATA_DIR=/path ./setup.sh   # point at an existing dataset
#   WRITE_ENV=0 ./setup.sh          # don't write .env (CI / dry runs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA_DIR="${ASCULTO_DATA_DIR:-$ROOT/data/HLS-CMDS}"
ZENODO="https://zenodo.org/api/records/15376628/files"
FILES="HLS_CMDS_README.txt HS.csv LS.csv Mix.csv HS.zip LS.zip Mix.zip"
PY="${ROOT}/.venv/bin/python"

echo "[setup] data dir: $DATA_DIR"

# ---- 1. dataset (download from source only if needed)
if [ -f "$DATA_DIR/Mix.csv" ] && { [ -d "$DATA_DIR/Mix/Mix" ] || ls "$DATA_DIR"/Mix/*.wav >/dev/null 2>&1; }; then
  echo "[setup] dataset already present, skipping download."
else
  echo "[setup] downloading HLS-CMDS (37.8 MB) from Zenodo (doi:10.1109/IEEEDATA.2025.3566012)..."
  mkdir -p "$DATA_DIR"
  for f in $FILES; do
    curl -fL --retry 3 --connect-timeout 30 -o "$DATA_DIR/$f" "$ZENODO/$f/content"
  done
  echo "[setup] extracting..."
  (cd "$DATA_DIR" && unzip -qo HS.zip && unzip -qo LS.zip && unzip -qo Mix.zip \
    && rm -f HS.zip LS.zip Mix.zip)
  rm -rf "$DATA_DIR/__MACOSX"
  echo "[setup] dataset ready at $DATA_DIR"
fi

# ---- 2. venv + dependencies (uv)
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv..."
  python3 -m pip install --quiet uv
fi
if [ ! -x "$PY" ]; then
  echo "[setup] creating venv..."
  uv venv .venv --python 3.11
fi
echo "[setup] installing dependencies..."
uv pip install --python "$PY" -r ml/requirements.txt -r webapp/requirements.txt -r requirements-dev.txt
[ -f notebooks/requirements.txt ] && uv pip install --python "$PY" -r notebooks/requirements.txt || true

echo "[setup] registering the Jupyter kernel..."
"$PY" -m ipykernel install --user --name asculto --display-name "Python 3 (asculto)" || echo "skip: jupyter not installed"

# ---- 3. .env (used by config to find the dataset)
if [ "${WRITE_ENV:-1}" = "1" ]; then
  printf 'ASCULTO_DATA_DIR=%s\n' "$DATA_DIR" > "$ROOT/.env"
  echo "[setup] wrote .env -> ASCULTO_DATA_DIR=$DATA_DIR"
fi

# ---- 4. verify the dataset matches the expected manifest counts
echo "[setup] verifying dataset..."
"$PY" - "$DATA_DIR" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "ml"))
from cardia.data.hls_cmds import build_task_manifest

data = sys.argv[1]
m = build_task_manifest(data, "sound", "binary", "/tmp/setup_check_manifests")
assert len(m) == 535, f"expected 535, got {len(m)}"
h = build_task_manifest(data, "heart", "binary", "/tmp/setup_check_manifests")
l = build_task_manifest(data, "lung", "binary", "/tmp/setup_check_manifests")
assert len(h) == 195 and len(l) == 195
print(f"  OK: sound={len(m)} heart={len(h)} lung={len(l)} recordings verified")
PY

echo "[setup] done."
echo
echo "  Train:      .venv/bin/python -m cardia.cli --task heart --stage 2 --model resnet18"
echo "  Campaign:   cd ml && SMOKE=1 ./scripts/run_sequential.sh"
echo "  Web app:    .venv/bin/python -m uvicorn webapp.app.main:app --port 8010"
