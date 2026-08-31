#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP6_ACCEPTANCE_OUT:-/tmp/camera_v11_step6_global_shadow_v1}"
DURATION="${V11_STEP6_DURATION_SEC:-60}"
DISPLAY_LOG="$OUT/display.log"
MATCH_LOG="$OUT/match.log"
LAUNCHER_LOG="$OUT/launcher.log"
CHECK_LOG="$OUT/check.log"
PAIR_TSV="$ROOT/artifacts/reid/step4_pair_scores_v1.tsv"
MATCH_TSV="$ROOT/artifacts/reid/step4_same_room_matches_v1.tsv"
GLOBAL_TSV="$ROOT/artifacts/reid/step5_global_shadow_v1.tsv"
VERIFY_TSV="$ROOT/artifacts/reid/step6_global_verify_v1.tsv"
mkdir -p "$OUT"

fail() {
  printf 'V11_STEP6_GLOBAL_VERIFY_V1 RESULT=FAIL reason=%s production_global_id=0 room_id=0 face=0 handoff=0 geometry_enabled=0 identity_accuracy_proven=0\n' "$*" >&2
  exit 1
}

[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration"
(( DURATION >= 60 )) || fail "duration_must_be_at_least_60_seconds"

"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_quality_v1.py" >"$OUT/step1_unit.log" 2>&1 || fail "step1_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_gallery_v1.py" >"$OUT/step2_unit.log" 2>&1 || fail "step2_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_pair_scorer_v1.py" >"$OUT/step3_unit.log" 2>&1 || fail "step3_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_same_room_evidence_v1.py" >"$OUT/step4_evidence_unit.log" 2>&1 || fail "step4_evidence_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_reid_same_room_matcher_v1.py" >"$OUT/step4_unit.log" 2>&1 || fail "step4_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step5_global_shadow_v1.py" >"$OUT/step5_unit.log" 2>&1 || fail "step5_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step6_global_shadow_hysteresis_v1.py" 2>&1 | tee "$OUT/step6_unit.log"
(( PIPESTATUS[0] == 0 )) || fail "step6_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_frozen_step123_guard.py" 2>&1 | tee "$OUT/frozen_guard.log"
(( PIPESTATUS[0] == 0 )) || fail "frozen_step123_guard"

: >"$DISPLAY_LOG"
: >"$MATCH_LOG"
: >"$LAUNCHER_LOG"
: >"$CHECK_LOG"
V11_STEP6_DISPLAY_LOG="$DISPLAY_LOG" \
V11_STEP6_GLOBAL_LOG="$MATCH_LOG" \
V11_STEP4_PAIR_TSV="$PAIR_TSV" \
V11_STEP4_MATCH_TSV="$MATCH_TSV" \
V11_STEP5_GLOBAL_TSV="$GLOBAL_TSV" \
V11_STEP6_VERIFY_TSV="$VERIFY_TSV" \
V11_STEP6_RUN_SEC="$DURATION" \
  timeout -s TERM "$((DURATION + 75))s" \
  bash "$ROOT/scripts/run_camera_v11_step6_global_shadow_v1.sh" \
  >"$LAUNCHER_LOG" 2>&1
status=$?
if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
  tail -n 160 "$LAUNCHER_LOG" >&2 || true
  tail -n 240 "$MATCH_LOG" >&2 || true
  fail "launcher_status_$status"
fi

grep -q '^CAMERA_V11_STEP5_GLOBAL_SHADOW_NATURAL_PRIME result=PASS ' "$LAUNCHER_LOG" || fail "natural_prime_not_passed"
grep -q '^CAMERA_V11_STEP5_GLOBAL_SHADOW_PREFLIGHT result=PASS ' "$LAUNCHER_LOG" || fail "no_preflight_pass"
grep -q 'runtime_module=services.camera_v11.step6_global_shadow_runtime_v1' "$LAUNCHER_LOG" || fail "step6_runtime_module_not_selected"
grep -q '^CAMERA_V11_STEP6_GLOBAL_VERIFY_V1_ARCH ' "$MATCH_LOG" || fail "step6_arch_missing"

"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step6_global_shadow_v1_log.py" \
  --display-log "$DISPLAY_LOG" --match-log "$MATCH_LOG" \
  --pair-tsv "$PAIR_TSV" --match-tsv "$MATCH_TSV" --global-tsv "$GLOBAL_TSV" \
  --verify-tsv "$VERIFY_TSV" --warmup-windows 2 2>&1 | tee "$CHECK_LOG"
exit "${PIPESTATUS[0]}"
