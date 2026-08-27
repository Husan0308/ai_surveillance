#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${CAMERA_V90_PYTHON:-$ROOT/.venv-trt86/bin/python}"

fail() {
  printf 'V90_GI_SETUP FAIL %s\n' "$*" >&2
  exit 2
}

[[ -x "$PY" ]] || fail "python_missing path=$PY"

PYVER="$($PY - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
[[ "$PYVER" == "3.10" ]] || fail "wrong_python expected=3.10 got=$PYVER path=$PY"

INCLUDEPY="$($PY - <<'PY'
import sysconfig
print(sysconfig.get_paths().get('include', ''))
PY
)"
[[ -n "$INCLUDEPY" ]] || fail "python_include_unknown"
[[ -f "$INCLUDEPY/Python.h" ]] || {
  printf 'V90_GI_SETUP FAIL python_headers_missing include=%s\n' "$INCLUDEPY" >&2
  printf 'V90_GI_SETUP next=install matching Python-3.10 development headers for this interpreter, then rerun\n' >&2
  exit 3
}

printf 'V90_GI_SETUP python=%s version=%s include=%s\n' "$PY" "$PYVER" "$INCLUDEPY"

# Ubuntu 24.04/Noble build dependencies for PyGObject from PyPI.
# We intentionally do NOT install python3-dev here because the system Python is
# 3.12 while this environment must stay on Python 3.10 for TensorRT 8.6.1.
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  libcairo2-dev \
  gobject-introspection \
  libgirepository1.0-dev \
  libgirepository-2.0-dev \
  gir1.2-gstreamer-1.0

"$PY" -m pip install --upgrade pip setuptools wheel
# PyGObject 3.48.2 supports Python 3.10 and is close to the DeepStream 7.1 era.
# Keep this pinned while validating the same-process TRT8.6/DeepStream path.
"$PY" -m pip install --upgrade 'pycairo>=1.20,<2' 'PyGObject==3.48.2'

"$PY" - <<'PY'
import sys
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

tracker = Gst.ElementFactory.make('nvtracker', 'v90_gi_setup_tracker_probe')
print(f"V90_GI_SETUP_VERIFY python={sys.version.split()[0]} gi=OK gst=OK nvtracker={'OK' if tracker else 'FAIL'}")
if tracker is None:
    raise SystemExit(10)
PY

printf 'V90_GI_SETUP PASS next=bash scripts/probe_v90_py310_sameprocess.sh\n'
