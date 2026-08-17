#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
VENDOR_DIR=".runtime/vendor/deep-person-reid"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install "numpy>=1.26,<3" "Cython>=3,<4"

if [ ! -d "$VENDOR_DIR/.git" ]; then
  rm -rf "$VENDOR_DIR"
  mkdir -p "$(dirname "$VENDOR_DIR")"
  git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git "$VENDOR_DIR"
else
  git -C "$VENDOR_DIR" fetch --depth 1 origin master
  git -C "$VENDOR_DIR" reset --hard origin/master
fi

# upstream setup.py imports numpy + Cython while setuptools is evaluating the
# editable build. pip build isolation therefore fails before install_requires can
# install numpy. Build against the already prepared venv instead.
"$PYTHON_BIN" -m pip install --no-build-isolation -e "$VENDOR_DIR"

"$PYTHON_BIN" - <<'PY'
import numpy
import torch
import torchreid
print(f"REID_BACKEND_INSTALL numpy={numpy.__version__}")
print(f"REID_BACKEND_INSTALL torch={torch.__version__}")
print(f"REID_BACKEND_INSTALL torchreid={torchreid.__version__}")
print("REID_BACKEND_INSTALL=PASS")
PY

export CAMERA_V2_REID_BACKEND=osnet_ain
"$PYTHON_BIN" scripts/setup_camera_v2_reid.py
