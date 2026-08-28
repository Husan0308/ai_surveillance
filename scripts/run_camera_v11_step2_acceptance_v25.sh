#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP2_V25_ACCEPTANCE_OUT:-/tmp/camera_v11_step2_acceptance_v25}"
DURATION="${V11_STEP2_V25_DURATION_SEC:-60}"
WARMUP_WINDOWS="${V11_STEP2_V25_WARMUP_WINDOWS:-2}"
mkdir -p "$OUT"

fail() {
  printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration=$DURATION"
[[ "$WARMUP_WINDOWS" =~ ^[0-9]+$ ]] || fail "invalid_warmup_windows=$WARMUP_WINDOWS"
command -v timeout >/dev/null 2>&1 || fail "timeout_missing"

failed=0

printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_START stage=display-only duration=%ss aggregate=1 warmup_windows=%s\n' \
  "$DURATION" "$WARMUP_WINDOWS"
display_only_log="$OUT/display_only.display.log"
: >"$display_only_log"
timeout -s TERM "${DURATION}s" bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$display_only_log" 2>&1
status=$?
if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
  printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_RESULT stage=display-only result=FAIL launcher_status=%s\n' "$status"
  failed=1
fi
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step1_v25_aggregate_log.py" \
  "$display_only_log" --warmup-windows "$WARMUP_WINDOWS" | tee "$OUT/display_only.check.log"
(( PIPESTATUS[0] == 0 )) || failed=1

if (( failed == 0 )); then
  for run in 1 2 3; do
    display_log="$OUT/full_${run}.display.log"
    detector_log="$OUT/full_${run}.detector.log"
    launcher_log="$OUT/full_${run}.launcher.log"
    check_log="$OUT/full_${run}.check.log"
    : >"$display_log"
    : >"$detector_log"
    : >"$launcher_log"
    : >"$check_log"

    printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_START stage=full run=%s duration=%ss aggregate=1 warmup_windows=%s\n' \
      "$run" "$DURATION" "$WARMUP_WINDOWS"
    V11_STEP2_DISPLAY_LOG="$display_log" \
    V11_STEP2_DETECTOR_LOG="$detector_log" \
      timeout -s TERM "$((DURATION + 25))s" \
      bash "$ROOT/scripts/run_camera_v11_step2_production_fp32_v25.sh" \
      >"$launcher_log" 2>&1
    status=$?

    if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
      printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_RESULT stage=full run=%s result=FAIL launcher_status=%s\n' \
        "$run" "$status" | tee -a "$check_log"
      failed=1
      break
    fi

    if ! grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$launcher_log"; then
      printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=no_vram_boost_gate\n' \
        "$run" | tee -a "$check_log"
      failed=1
      break
    fi

    "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step2_production_log_v25.py" \
      --display-log "$display_log" --detector-log "$detector_log" \
      --warmup-windows "$WARMUP_WINDOWS" | tee -a "$check_log"
    check_status=${PIPESTATUS[0]}
    if (( check_status != 0 )); then
      printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=production_checker\n' \
        "$run" | tee -a "$check_log"
      failed=1
      break
    fi

    if pgrep -af 'services\.camera_v11\.(step1_|step2_)|yolo26_trt86_step2_worker\.py' >/dev/null 2>&1; then
      printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=stale_project_process\n' \
        "$run" | tee -a "$check_log"
      pgrep -af 'services\.camera_v11\.(step1_|step2_)|yolo26_trt86_step2_worker\.py' | tee -a "$check_log" || true
      failed=1
      break
    fi

    printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_RESULT stage=full run=%s result=PASS\n' "$run" | tee -a "$check_log"
  done
fi

if (( failed == 0 )); then
  printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_FINAL result=PASS display_only=1 full_consecutive=3 duration=%ss aggregate=1\n' "$DURATION"
else
  printf 'CAMERA_V11_STEP2_V25_ACCEPTANCE_FINAL result=FAIL duration=%ss aggregate=1\n' "$DURATION"
fi
exit "$failed"
