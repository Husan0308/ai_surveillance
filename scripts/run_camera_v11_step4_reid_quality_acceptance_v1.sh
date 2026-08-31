#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP4_QUALITY_ACCEPTANCE_OUT:-/tmp/camera_v11_step4_reid_quality_v1}"
DURATION="${V11_STEP4_QUALITY_DURATION_SEC:-60}"
DISPLAY_LOG="$OUT/display.log"
QUALITY_LOG="$OUT/quality.log"
LAUNCHER_LOG="$OUT/launcher.log"
CHECK_LOG="$OUT/check.log"
mkdir -p "$OUT"

fail() {
  printf 'V11_STEP4_REID_QUALITY_V1 RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration"
(( DURATION >= 60 )) || fail "duration_must_be_at_least_60_seconds"

"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_quality_v1.py" \
  2>&1 | tee "$OUT/unit.log"
(( PIPESTATUS[0] == 0 )) || fail "unit_tests"

: >"$DISPLAY_LOG"
: >"$QUALITY_LOG"
: >"$LAUNCHER_LOG"
: >"$CHECK_LOG"
V11_STEP4_QUALITY_DISPLAY_LOG="$DISPLAY_LOG" \
V11_STEP4_QUALITY_LOG="$QUALITY_LOG" \
V11_STEP4_QUALITY_RUN_SEC="$DURATION" \
  timeout -s TERM "$((DURATION + 60))s" \
  bash "$ROOT/scripts/run_camera_v11_step4_reid_quality_v1.sh" \
  >"$LAUNCHER_LOG" 2>&1
status=$?
if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
  tail -n 80 "$LAUNCHER_LOG" >&2 || true
  tail -n 80 "$QUALITY_LOG" >&2 || true
  fail "launcher_status_$status"
fi
grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$LAUNCHER_LOG" \
  || fail "no_powermizer_gate"

"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step4_reid_quality_v1_log.py" \
  --display-log "$DISPLAY_LOG" --quality-log "$QUALITY_LOG" --warmup-windows 2 \
  2>&1 | tee "$CHECK_LOG"
exit "${PIPESTATUS[0]}"
