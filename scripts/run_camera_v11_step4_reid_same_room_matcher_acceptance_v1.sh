#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP4_MATCH_ACCEPTANCE_OUT:-/tmp/camera_v11_step4_reid_same_room_matcher_v1}"
DURATION="${V11_STEP4_MATCH_DURATION_SEC:-60}"
DISPLAY_LOG="$OUT/display.log"
MATCH_LOG="$OUT/match.log"
LAUNCHER_LOG="$OUT/launcher.log"
CHECK_LOG="$OUT/check.log"
PAIR_TSV="$ROOT/artifacts/reid/step4_pair_scores_v1.tsv"
MATCH_TSV="$ROOT/artifacts/reid/step4_same_room_matches_v1.tsv"
mkdir -p "$OUT"

fail() {
  printf 'V11_STEP4_REID_SAME_ROOM_MATCHER_V1 RESULT=FAIL reason=%s global_id=0 room_id=0 face=0 handoff=0\n' "$*" >&2
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
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_same_room_evidence_v1.py" \
  2>&1 | tee "$OUT/step4_evidence_unit.log"
(( PIPESTATUS[0] == 0 )) || fail "step4_evidence_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_same_room_matcher_v1.py" \
  2>&1 | tee "$OUT/step4_unit.log"
(( PIPESTATUS[0] == 0 )) || fail "step4_unit_tests"

: >"$DISPLAY_LOG"
: >"$MATCH_LOG"
: >"$LAUNCHER_LOG"
: >"$CHECK_LOG"
V11_STEP4_MATCH_DISPLAY_LOG="$DISPLAY_LOG" \
V11_STEP4_MATCH_LOG="$MATCH_LOG" \
V11_STEP4_PAIR_TSV="$PAIR_TSV" \
V11_STEP4_MATCH_TSV="$MATCH_TSV" \
V11_STEP4_MATCH_RUN_SEC="$DURATION" \
  timeout -s TERM "$((DURATION + 75))s" \
  bash "$ROOT/scripts/run_camera_v11_step4_reid_same_room_matcher_v1.sh" \
  >"$LAUNCHER_LOG" 2>&1
status=$?
if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
  tail -n 100 "$LAUNCHER_LOG" >&2 || true
  tail -n 140 "$MATCH_LOG" >&2 || true
  fail "launcher_status_$status"
fi
grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$LAUNCHER_LOG" \
  || fail "no_powermizer_gate"

"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step4_reid_same_room_matcher_v1_log.py" \
  --display-log "$DISPLAY_LOG" --match-log "$MATCH_LOG" \
  --pair-tsv "$PAIR_TSV" --match-tsv "$MATCH_TSV" \
  --warmup-windows 2 2>&1 | tee "$CHECK_LOG"
exit "${PIPESTATUS[0]}"
