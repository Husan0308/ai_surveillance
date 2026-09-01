#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export V11_STEP5_RUNTIME_MODULE="${V11_STEP6_RUNTIME_MODULE:-services.camera_v11.step6_global_shadow_runtime_v1}"
export V11_STEP5_DISPLAY_LOG="${V11_STEP6_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP6_GLOBAL_DISPLAY.log}"
export V11_STEP5_GLOBAL_LOG="${V11_STEP6_GLOBAL_LOG:-/tmp/CAMERA_V11_STEP6_GLOBAL.log}"
export V11_STEP5_POWER_LOG="${V11_STEP6_POWER_LOG:-/tmp/CAMERA_V11_STEP6_POWER_V1.log}"
export V11_STEP5_GLOBAL_TSV="${V11_STEP5_GLOBAL_TSV:-$ROOT/artifacts/reid/step5_global_shadow_v1.tsv}"
export V11_STEP6_VERIFY_TSV="${V11_STEP6_VERIFY_TSV:-$ROOT/artifacts/reid/step6_global_verify_v1.tsv}"
export V11_STEP5_RUN_SEC="${V11_STEP6_RUN_SEC:-60}"
export V11_STEP5_HZ="${V11_STEP6_HZ:-2.0}"
export V11_STEP5_DETECTOR_CONF="${V11_STEP6_DETECTOR_CONF:-0.18}"
export V11_STEP5_DISPLAY_WARMUP_SEC="${V11_STEP6_DISPLAY_WARMUP_SEC:-8}"
export V11_STEP5_RUNTIME_CPUS="${V11_STEP6_RUNTIME_CPUS:-0-10}"
export V11_STEP5_MATCH_CPU="${V11_STEP6_MATCH_CPU:-11}"
export V11_STEP5_MIN_ACTIVE_MEMORY_MHZ="${V11_STEP6_MIN_ACTIVE_MEMORY_MHZ:-3000}"
export V11_STEP5_MIN_PRIME_GPU_UTIL="${V11_STEP6_MIN_PRIME_GPU_UTIL:-50}"

printf 'CAMERA_V11_STEP6_GLOBAL_VERIFY_LAUNCH mode=shadow runtime=%s geometry_enabled=0 production_global_id=0 room_id=0 face=0 handoff=0\n' \
  "$V11_STEP5_RUNTIME_MODULE"
exec bash "$ROOT/scripts/run_camera_v11_step5_global_shadow_v1.sh"
