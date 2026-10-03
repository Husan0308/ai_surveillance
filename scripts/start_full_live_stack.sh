#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# One explicit project runtime; never discover a different Python or install on
# launch. Helpers return only the interpreter on stdout.
source "$ROOT/scripts/runtime_python.sh"
if [[ "${1:-}" != "" && "${1:-}" != "--preflight" ]]; then
  printf 'Usage: %s [--preflight]\n' "$0" >&2
  exit 2
fi
PY="$(resolve_python)" || exit 1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
if [[ "${1:-}" == "--preflight" ]]; then
  exec "$PY" -I -B "$ROOT/scripts/runtime_foundation.py" check --root "$ROOT"
fi
mkdir -p "$ROOT/.runtime"
# Frozen F1 interpreter resolution above is unchanged; F3 supervises processes.
exec "$PY" -B -m scripts.full_stack_runtime
