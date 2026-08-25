#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "missing .venv/bin/python" >&2
  exit 1
fi
if [[ ! -x .venv-trt86/bin/python ]]; then
  echo "missing .venv-trt86/bin/python" >&2
  exit 1
fi
if [[ ! -f artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine ]]; then
  echo "missing YOLO26s TRT86 engine" >&2
  exit 1
fi

# Remove leftover experiment selectors. The Python entrypoint installs the exact
# CAM-01 profile before detection.py is imported.
while IFS='=' read -r name _; do
  case "$name" in
    CAMERA_V2_*|QWEN_*|NVDS_*) unset "$name" || true ;;
  esac
done < <(env)

unset CUDA_VISIBLE_DEVICES || true
unset CUDA_MODULE_LOADING || true

export PYTHONUNBUFFERED=1
exec .venv/bin/python -u -m services.camera_v2.cam01_lowlat_trt86
