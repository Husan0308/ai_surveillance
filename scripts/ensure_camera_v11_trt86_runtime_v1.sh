#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="${V11_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
NUMPY_VERSION="1.26.4"
RUNTIME_SITE="${V11_TRT86_RUNTIME_SITE:-$ROOT/artifacts/reid/python_trt86_site}"
MANIFEST_NAME=".v11_runtime_paths.json"

fail() {
  printf 'V11_TRT86_RUNTIME RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -x "$PYTHON" ]] || fail "python_missing path=$PYTHON"

probe_runtime() {
  local site="$1"
  V11_TRT86_PROBE_SITE="$site" V11_TRT86_MANIFEST_NAME="$MANIFEST_NAME" "$PYTHON" -I - <<'PY'
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

runtime_site = Path(os.environ["V11_TRT86_PROBE_SITE"]).resolve()
runtime_site.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(runtime_site))

import numpy as np
import tensorrt as trt

# Force-load NumPy's compiled core. A partial/mixed installation commonly dies
# here with numpy.core._multiarray_umath missing.
importlib.import_module("numpy.core._multiarray_umath")

if not str(trt.__version__).startswith("8.6.1"):
    raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")
if np.__version__ != "1.26.4":
    raise RuntimeError(f"NumPy 1.26.4 required, got {np.__version__}")

np_file = Path(np.__file__).resolve()
if runtime_site not in np_file.parents:
    raise RuntimeError(
        f"NumPy must come from project runtime site: site={runtime_site} numpy_file={np_file}"
    )

trt_file = Path(trt.__file__).resolve()
spec = importlib.util.find_spec("tensorrt")
locations = list(spec.submodule_search_locations or []) if spec is not None else []
if locations:
    trt_root = Path(locations[0]).resolve().parent
else:
    trt_root = trt_file.parent
if not trt_root.is_dir():
    raise RuntimeError(f"TensorRT module root missing: {trt_root}")

manifest = runtime_site / os.environ["V11_TRT86_MANIFEST_NAME"]
manifest.write_text(
    json.dumps(
        {
            "python": str(Path(sys.executable).resolve()),
            "numpy_site": str(runtime_site),
            "numpy_file": str(np_file),
            "tensorrt_root": str(trt_root),
            "tensorrt_file": str(trt_file),
            "tensorrt_version": str(trt.__version__),
        },
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print(
    "V11_TRT86_RUNTIME READY "
    f"python={sys.version.split()[0]} executable={sys.executable} "
    f"prefix={Path(sys.prefix).resolve()} base_prefix={Path(sys.base_prefix).resolve()} "
    f"numpy={np.__version__} numpy_file={np_file} tensorrt={trt.__version__} "
    f"tensorrt_root={trt_root} runtime_site={runtime_site}",
    flush=True,
)
PY
}

probe_worker_mode() {
  local site="$1"
  V11_TRT86_PROBE_SITE="$site" V11_TRT86_MANIFEST_NAME="$MANIFEST_NAME" "$PYTHON" -I -c '
import importlib, json, os, sys
from pathlib import Path
site = Path(os.environ["V11_TRT86_PROBE_SITE"]).resolve()
manifest = json.loads((site / os.environ["V11_TRT86_MANIFEST_NAME"]).read_text(encoding="utf-8"))
trt_root = Path(manifest["tensorrt_root"]).resolve()
sys.path[:0] = [str(site), str(trt_root)]
import numpy as np
import tensorrt as trt
importlib.import_module("numpy.core._multiarray_umath")
if np.__version__ != "1.26.4":
    raise RuntimeError(f"worker-mode NumPy mismatch: {np.__version__}")
if not str(trt.__version__).startswith("8.6.1"):
    raise RuntimeError(f"worker-mode TensorRT mismatch: {trt.__version__}")
print(f"V11_TRT86_RUNTIME WORKER_MODE=PASS numpy={np.__version__} tensorrt={trt.__version__} tensorrt_root={trt_root}", flush=True)
'
}

probe_log=""
if probe_log="$(probe_runtime "$RUNTIME_SITE" 2>&1)"; then
  printf '%s\n' "$probe_log"
  worker_log=""
  if worker_log="$(probe_worker_mode "$RUNTIME_SITE" 2>&1)"; then
    printf '%s\n' "$worker_log"
    printf 'V11_TRT86_RUNTIME RESULT=PASS source=reuse\n'
    exit 0
  fi
  printf 'V11_TRT86_RUNTIME WORKER_MODE=FAIL action=repair detail=%q\n' "$worker_log" >&2
else
  printf 'V11_TRT86_RUNTIME PROBE=FAIL action=repair detail=%q\n' "$probe_log" >&2
fi

# Never pip-install into the interpreter itself. Install a binary NumPy wheel
# into a project-local overlay and explicitly retain the TensorRT module root
# that the verified TRT86 interpreter can import.
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  fail "pip_missing python=$PYTHON"
fi

tmp="${RUNTIME_SITE}.tmp.$$"
rm -rf "$tmp"
mkdir -p "$(dirname "$RUNTIME_SITE")"
trap 'rm -rf "$tmp"' EXIT INT TERM

printf 'V11_TRT86_RUNTIME REPAIR action=install_project_numpy version=%s target=%s python=%s\n' \
  "$NUMPY_VERSION" "$RUNTIME_SITE" "$PYTHON"
"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-input \
  --no-cache-dir \
  --only-binary=:all: \
  --ignore-installed \
  --target "$tmp" \
  "numpy==$NUMPY_VERSION" \
  || fail "numpy_wheel_install_failed version=$NUMPY_VERSION"

probe_log=""
if ! probe_log="$(probe_runtime "$tmp" 2>&1)"; then
  printf '%s\n' "$probe_log" >&2
  fail "runtime_invalid_after_repair"
fi
worker_log=""
if ! worker_log="$(probe_worker_mode "$tmp" 2>&1)"; then
  printf '%s\n' "$worker_log" >&2
  fail "worker_mode_invalid_after_repair"
fi

rm -rf "$RUNTIME_SITE"
mv "$tmp" "$RUNTIME_SITE"
trap - EXIT INT TERM

probe_log="$(probe_runtime "$RUNTIME_SITE" 2>&1)" \
  || fail "runtime_invalid_after_publish"
worker_log="$(probe_worker_mode "$RUNTIME_SITE" 2>&1)" \
  || fail "worker_mode_invalid_after_publish"
printf '%s\n' "$probe_log"
printf '%s\n' "$worker_log"
printf 'V11_TRT86_RUNTIME RESULT=PASS source=repair\n'
