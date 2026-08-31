#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP4_GALLERY_ACCEPTANCE_OUT:-/tmp/camera_v11_step4_reid_gallery_v1}"
DURATION="${V11_STEP4_GALLERY_DURATION_SEC:-60}"
DISPLAY_LOG="$OUT/display.log"
GALLERY_LOG="$OUT/gallery.log"
LAUNCHER_LOG="$OUT/launcher.log"
CHECK_LOG="$OUT/check.log"
mkdir -p "$OUT"

fail() {
  printf 'V11_STEP4_REID_GALLERY_V1 RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration"
(( DURATION >= 60 )) || fail "duration_must_be_at_least_60_seconds"

"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_quality_v1.py" \
  2>&1 | tee "$OUT/step1_unit.log"
(( PIPESTATUS[0] == 0 )) || fail "step1_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_gallery_v1.py" \
  2>&1 | tee "$OUT/step2_unit.log"
(( PIPESTATUS[0] == 0 )) || fail "step2_unit_tests"

: >"$DISPLAY_LOG"
: >"$GALLERY_LOG"
: >"$LAUNCHER_LOG"
: >"$CHECK_LOG"
V11_STEP4_GALLERY_DISPLAY_LOG="$DISPLAY_LOG" \
V11_STEP4_GALLERY_LOG="$GALLERY_LOG" \
V11_STEP4_GALLERY_RUN_SEC="$DURATION" \
  timeout -s TERM "$((DURATION + 75))s" \
  bash "$ROOT/scripts/run_camera_v11_step4_reid_gallery_v1.sh" \
  >"$LAUNCHER_LOG" 2>&1
status=$?
if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
  tail -n 100 "$LAUNCHER_LOG" >&2 || true
  tail -n 100 "$GALLERY_LOG" >&2 || true
  fail "launcher_status_$status"
fi
grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$LAUNCHER_LOG" \
  || fail "no_powermizer_gate"

"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step4_reid_gallery_v1_log.py" \
  --display-log "$DISPLAY_LOG" --gallery-log "$GALLERY_LOG" --warmup-windows 2 \
  2>&1 | tee "$CHECK_LOG"
exit "${PIPESTATUS[0]}"
