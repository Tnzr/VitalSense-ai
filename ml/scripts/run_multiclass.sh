#!/usr/bin/env bash
# Multiclass benchmark: heart sound types (10-class) and lung sound types
# (6-class), then the combined heart+lung confusion report.
#
# Output defaults to ml/runs/mc (where the web app looks for multiclass checkpoints).
#   OUT=/path/to/out EPOCHS=120 PATIENCE=20 ./run_multiclass.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
ML="$ROOT/ml"
OUT="${OUT:-$ML/runs/mc}"
EXTRA=()
[ -n "${EPOCHS:-}" ] && EXTRA+=(--epochs "$EPOCHS")
[ -n "${PATIENCE:-}" ] && EXTRA+=(--patience "$PATIENCE")

cd "$ML"
mkdir -p "$OUT"

echo "== heart 10-class =="
"$PY" scripts/sequential.py --tasks heart --class-scheme 10class --out-dir "$OUT" "${EXTRA[@]}"

echo "== lung 6-class =="
"$PY" scripts/sequential.py --tasks lung --class-scheme 6class --out-dir "$OUT" "${EXTRA[@]}"

echo "== combined heart + lung confusion report =="
"$PY" -m cardia.report.combined --out-dir "$OUT" \
  --heart-stage 2 --heart-model resnet18 \
  --lung-stage 2 --lung-model resnet18
