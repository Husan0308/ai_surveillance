#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
MODE="${1:-full}"
case "$MODE" in
  synthetic-trt|full) ;;
  *) printf 'usage: %s synthetic-trt|full\n' "$0" >&2; exit 2 ;;
esac

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V11_STEP2_MODE="$MODE"
export V11_STEP2_HZ="${V11_STEP2_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP2_CONF:-0.18}"
export V11_STEP2_ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
exec nice -n "${V11_STEP2_NICE:-10}" \
  "$ROOT/.venv-trt86/bin/python" -u -m services.camera_v11.step2_production_fp32_local_trt_v20
