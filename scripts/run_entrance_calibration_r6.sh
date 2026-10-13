#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.runtime/full-stack-venv/bin/python"
if [[ ! -x "$PY" ]]; then
  printf 'ERROR: project Python runtime is unavailable: %s\n' "$PY" >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
exec "$PY" -B -m scripts.run_entrance_calibration_r6 "$@"
