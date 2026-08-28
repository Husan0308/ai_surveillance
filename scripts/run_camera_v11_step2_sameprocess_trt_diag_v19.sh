#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv-trt86/bin/python"
ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
[[ -x "$PY" ]] || { echo "missing $PY" >&2; exit 1; }
[[ -s "$ENGINE" ]] || { echo "missing $ENGINE" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "DISPLAY is empty" >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V11_STEP2_ENGINE="$ENGINE"
export V11_DIAG_TRT_HZ="${V11_DIAG_TRT_HZ:-12.0}"
export V11_DIAG_TRT_DELAY_SEC="${V11_DIAG_TRT_DELAY_SEC:-5.0}"
exec "$PY" -u -m services.camera_v11.step2_sameprocess_trt_diag_v19
