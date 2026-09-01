#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_DIRECT_REID_OUT:-/tmp/camera_v11_direct_global_reid_v1}"
RUN_SEC="${V11_DIRECT_REID_RUN_SEC:-300}"

mkdir -p "$OUT"
printf 'DIRECT GLOBAL REID | CAM-01 + CAM-04 | GID comes from persistent ReID memory\n' >"$OUT/phase_state.txt"

export V11_STEP6_RUNTIME_MODULE="services.camera_v11.step9_direct_global_reid_debug_runtime_v1"
export V11_STEP8_DEBUG_BBOX=1
export V11_STEP8_PHASE_STATE="$OUT/phase_state.txt"
export V11_STEP6_DISPLAY_LOG="$OUT/display.log"
export V11_STEP6_GLOBAL_LOG="$OUT/runtime.log"
export V11_STEP4_PAIR_TSV="$OUT/pairs.tsv"
export V11_STEP4_MATCH_TSV="$OUT/matcher.tsv"
export V11_STEP5_GLOBAL_TSV="$OUT/legacy_global.tsv"
export V11_STEP6_VERIFY_TSV="$OUT/legacy_verify.tsv"
export V11_STEP6_RUN_SEC="$RUN_SEC"

printf '%s\n' \
  'V11_DIRECT_GLOBAL_REID_V1 READY' \
  'Use the second CAM-01/CAM-04 debug preview.' \
  'Identity label format: local T-ID | REID | GID-xxxxxx | tracker state | DIRECT.' \
  'GID-pending for the first few clean ReID samples is expected.' \
  'A local T-ID change should re-associate back to the same GID.' \
  'Two simultaneously active people in one camera must never share one GID.' \
  'No manual person labels or floor calibration are required for this direct ReID test.'

exec bash "$ROOT/scripts/run_camera_v11_step6_global_shadow_v1.sh"
