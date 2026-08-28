#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="${V11_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
NUMPY_VERSION="${V11_TRT86_NUMPY_VERSION:-1.26.4}"

fail() {
  printf 'V11_TRT86_RUNTIME RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -x "$PYTHON" ]] || fail "python_missing path=$PYTHON"

probe_runtime() {
  "$PYTHON" -I - <<'PY'
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

# Force-load NumPy's compiled core. A partially installed Debian/system NumPy can
# import __init__.py and still fail here with _multiarray_umath missing.
importlib.import_module("numpy.core._multiarray_umath")

if not str(trt.__version__).startswith("8.6.1"):
    raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")
if np.__version__ != "1.26.4":
    raise RuntimeError(f"NumPy 1.26.4 required, got {np.__version__}")

prefix = Path(sys.prefix).resolve()
np_file = Path(np.__file__).resolve()
if prefix not in np_file.parents:
    raise RuntimeError(
        f"NumPy must be local to TRT86 venv: prefix={prefix} numpy_file={np_file}"
    )

print(
    "V11_TRT86_RUNTIME READY "
    f"python={sys.version.split()[0]} executable={sys.executable} "
    f"prefix={prefix} numpy={np.__version__} numpy_file={np_file} "
    f"tensorrt={trt.__version__}",
    flush=True,
)
PY
}

probe_log=""
if probe_log="$(probe_runtime 2>&1)"; then
  printf '%s\n' "$probe_log"
  printf 'V11_TRT86_RUNTIME RESULT=PASS source=reuse\n'
  exit 0
fi

printf 'V11_TRT86_RUNTIME PROBE=FAIL action=repair detail=%q\n' "$probe_log" >&2

# Do not repair or uninstall the host's /usr/lib NumPy. Install a CPython wheel
# inside .venv-trt86 so it shadows any broken system-site-packages copy.
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  fail "pip_missing python=$PYTHON"
fi

printf 'V11_TRT86_RUNTIME REPAIR action=install_local_numpy version=%s python=%s\n' \
  "$NUMPY_VERSION" "$PYTHON"
"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-input \
  --no-cache-dir \
  --only-binary=:all: \
  --ignore-installed \
  "numpy==$NUMPY_VERSION" \
  || fail "numpy_wheel_install_failed version=$NUMPY_VERSION"

probe_log=""
if ! probe_log="$(probe_runtime 2>&1)"; then
  printf '%s\n' "$probe_log" >&2
  fail "runtime_invalid_after_repair"
fi
printf '%s\n' "$probe_log"
printf 'V11_TRT86_RUNTIME RESULT=PASS source=repair\n'
