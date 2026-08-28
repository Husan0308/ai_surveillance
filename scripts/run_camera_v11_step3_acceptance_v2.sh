#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP3_V2_ACCEPTANCE_OUT:-/tmp/camera_v11_step3_acceptance_v2}"
DURATION="${V11_STEP3_V2_DURATION_SEC:-60}"
WARMUP_WINDOWS="${V11_STEP3_V2_WARMUP_WINDOWS:-2}"
mkdir -p "$OUT"

fail() {
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration=$DURATION"
[[ "$WARMUP_WINDOWS" =~ ^[0-9]+$ ]] || fail "invalid_warmup_windows=$WARMUP_WINDOWS"
command -v timeout >/dev/null 2>&1 || fail "timeout_missing"

failed=0

printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_START stage=unit\n'
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step3_tracker_v2.py" \
  2>&1 | tee "$OUT/unit.log"
if (( PIPESTATUS[0] != 0 )); then
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=unit result=FAIL\n'
  failed=1
else
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=unit result=PASS\n'
fi

if (( failed == 0 )); then
  for run in 1 2 3; do
    display_log="$OUT/full_${run}.display.log"
    tracker_log="$OUT/full_${run}.tracker.log"
    launcher_log="$OUT/full_${run}.launcher.log"
    check_log="$OUT/full_${run}.check.log"
    : >"$display_log"
    : >"$tracker_log"
    : >"$launcher_log"
    : >"$check_log"

    printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_START stage=full run=%s duration=%ss warmup_windows=%s\n' \
      "$run" "$DURATION" "$WARMUP_WINDOWS"

    V11_STEP3_DISPLAY_LOG="$display_log" \
    V11_STEP3_TRACKER_LOG="$tracker_log" \
      timeout -s TERM "$((DURATION + 15))s" \
      bash "$ROOT/scripts/run_camera_v11_step3_tracker_v2.sh" \
      >"$launcher_log" 2>&1
    status=$?

    if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL launcher_status=%s\n' \
        "$run" "$status" | tee -a "$check_log"
      failed=1
      break
    fi

    if ! grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$launcher_log"; then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=no_vram_boost_gate\n' \
        "$run" | tee -a "$check_log"
      failed=1
      break
    fi

    "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step3_tracker_v2_log.py" \
      --display-log "$display_log" --tracker-log "$tracker_log" \
      --warmup-windows "$WARMUP_WINDOWS" | tee -a "$check_log"
    check_status=${PIPESTATUS[0]}
    if (( check_status != 0 )); then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=tracker_checker\n' \
        "$run" | tee -a "$check_log"
      failed=1
      break
    fi

    if pgrep -af 'services\.camera_v11\.(step1_|step2_|step3_)|yolo26_trt86_step2_worker\.py' >/dev/null 2>&1; then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=stale_project_process\n' \
        "$run" | tee -a "$check_log"
      pgrep -af 'services\.camera_v11\.(step1_|step2_|step3_)|yolo26_trt86_step2_worker\.py' \
        | tee -a "$check_log" || true
      failed=1
      break
    fi

    printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=PASS\n' \
      "$run" | tee -a "$check_log"
  done
fi

if (( failed == 0 )); then
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_FINAL result=PASS unit=1 full_consecutive=3 duration=%ss step2_gate=1\n' \
    "$DURATION"
else
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_FINAL result=FAIL duration=%ss step2_gate=1\n' "$DURATION"
fi
exit "$failed"
