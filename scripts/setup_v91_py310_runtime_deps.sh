#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${CAMERA_V91_PYTHON:-$ROOT/.venv-trt86/bin/python}"

fail() { printf 'V91_SETUP FAIL %s\n' "$*" >&2; exit 2; }
[[ -x "$PY" ]] || fail "python_missing path=$PY"

"$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"V91_SETUP FAIL wrong_python={sys.version.split()[0]}")
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"V91_SETUP FAIL tensorrt={trt.__version__}")
print(f"V91_SETUP base=OK python={sys.version.split()[0]} trt={trt.__version__} gi=OK gst=OK")
PY

# Only pure-Python project configuration dependencies are added here.  Do not
# upgrade TensorRT, numpy, CUDA, GI, or system Python.
"$PY" -m pip install --upgrade 'PyYAML>=6,<7' 'python-dotenv>=1,<2'

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import yaml
from dotenv import load_dotenv
from services.ml_service.app.config import load_settings
settings = load_settings()
print(f"V91_SETUP_VERIFY yaml={yaml.__version__} dotenv=OK cameras={len(settings.cameras)}")
PY

echo 'V91_SETUP PASS next=bash scripts/run_camera_v2_bbox_v91.sh'
