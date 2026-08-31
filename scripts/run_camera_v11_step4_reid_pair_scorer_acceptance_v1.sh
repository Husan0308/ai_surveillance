#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP4_PAIR_ACCEPTANCE_OUT:-/tmp/camera_v11_step4_reid_pair_scorer_v1}"
DURATION="${V11_STEP4_PAIR_DURATION_SEC:-60}"
DISPLAY_LOG="$OUT/display.log"
PAIR_LOG="$OUT/pair.log"
LAUNCHER_LOG="$OUT/launcher.log"
CHECK_LOG="$OUT/check.log"
PAIR_TSV="$ROOT/artifacts/reid/step4_pair_scores_v1.tsv"
mkdir -p "$OUT"

fail() {
  printf 'V11_STEP4_REID_PAIR_SCORER_V1 RESULT=FAIL reason=%s\n' "$*" >&2
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
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_pair_scorer_v1.py" \
  2>&1 | tee "$OUT/step3_unit.log"
(( PIPESTATUS[0] == 0 )) || fail "step3_unit_tests"

: >"$DISPLAY_LOG"
: >"$PAIR_LOG"
: >"$LAUNCHER_LOG"
: >"$CHECK_LOG"
V11_STEP4_PAIR_DISPLAY_LOG="$DISPLAY_LOG" \
V11_STEP4_PAIR_LOG="$PAIR_LOG" \
V11_STEP4_PAIR_TSV="$PAIR_TSV" \
V11_STEP4_PAIR_RUN_SEC="$DURATION" \
  timeout -s TERM "$((DURATION + 75))s" \
  bash "$ROOT/scripts/run_camera_v11_step4_reid_pair_scorer_v1.sh" \
  >"$LAUNCHER_LOG" 2>&1
status=$?
if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
  tail -n 100 "$LAUNCHER_LOG" >&2 || true
  tail -n 120 "$PAIR_LOG" >&2 || true
  fail "launcher_status_$status"
fi
grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$LAUNCHER_LOG" \
  || fail "no_powermizer_gate"

"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step4_reid_pair_scorer_v1_log.py" \
  --display-log "$DISPLAY_LOG" --pair-log "$PAIR_LOG" --tsv "$PAIR_TSV" \
  --warmup-windows 2 2>&1 | tee "$CHECK_LOG"
exit "${PIPESTATUS[0]}"
