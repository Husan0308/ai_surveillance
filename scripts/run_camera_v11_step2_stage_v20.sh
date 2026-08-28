#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
MODE="${1:-full}"
case "$MODE" in
  extraction|preprocessing|full) ;;
  synthetic-trt) exec nice -n "${V11_STEP2_NICE:-10}" "$ROOT/scripts/run_camera_v11_step2_stage.sh" synthetic-trt ;;
  *) printf 'usage: %s extraction|preprocessing|synthetic-trt|full\n' "$0" >&2; exit 2 ;;
esac

export V11_STEP2_MODE="$MODE"
export V11_STEP2_HZ="${V11_STEP2_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP2_CONF:-0.18}"
export V11_STEP2_ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
export V11_STEP2_TRT86_PYTHON="$ROOT/.venv-trt86/bin/python"
export V11_STEP2_TRT86_WORKER="$ROOT/scripts/yolo26_trt86_step2_worker.py"
exec nice -n "${V11_STEP2_NICE:-10}" \
  "$ROOT/.venv/bin/python" -u -m services.camera_v11.step2_production_fp32_v18
