#!/usr/bin/env bash
# Source-only helpers. stdout is exactly one interpreter path, never status text.
resolve_python() {
  local candidate="${FULL_STACK_PYTHON:-$ROOT/.runtime/full-stack-venv/bin/python}"
  if [[ "$candidate" != /* || ! -x "$candidate" ]]; then
    printf 'RUNTIME FAIL missing/invalid interpreter: %s\nRun bash scripts/setup_runtime.sh first.\n' "$candidate" >&2
    return 1
  fi
  if ! env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$candidate" -I -B \
       "$ROOT/scripts/runtime_foundation.py" check --root "$ROOT" >/dev/null; then
    printf 'RUNTIME FAIL preflight: %s (no fallback or automatic install)\n' "$candidate" >&2
    return 1
  fi
  printf '%s\n' "$candidate"
}
