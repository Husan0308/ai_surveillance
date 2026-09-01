#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP8_OUT:-/tmp/camera_v11_step8_cam01_cam04_two_person_v1}"
ACK="${V11_STEP8_TWO_PERSON_ACK:-0}"
A_SEC="${V11_STEP8_PHASE_A_SEC:-25}"
B_SEC="${V11_STEP8_PHASE_B_SEC:-25}"
C_SEC="${V11_STEP8_PHASE_C_SEC:-40}"
D_SEC="${V11_STEP8_PHASE_D_SEC:-20}"
E_SEC="${V11_STEP8_PHASE_E_SEC:-20}"
SETTLE_SEC="${V11_STEP8_SETTLE_SEC:-5}"
DURATION="${V11_STEP8_DURATION_SEC:-180}"

DISPLAY_LOG="$OUT/display.log"
MATCH_LOG="$OUT/match.log"
LAUNCHER_LOG="$OUT/launcher.log"
CHECK_LOG="$OUT/step8_check.log"
PHASE_TSV="$OUT/step8_phase_markers.tsv"
PAIR_TSV="$OUT/step4_pair_scores_v1.tsv"
MATCH_TSV="$OUT/step4_same_room_matches_v1.tsv"
GLOBAL_TSV="$OUT/step5_global_shadow_v1.tsv"
VERIFY_TSV="$OUT/step6_global_verify_v1.tsv"
launcher_pid=""

cleanup() {
  if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

fail() {
  printf 'V11_STEP8_CAM01_CAM04_TWO_PERSON_V1 RESULT=FAIL reason=%s physical_people_expected=2 verified_ids_expected=2 wrong_merge_expected=0 id_swap_expected=0 production_global_id=0 identity_accuracy_proven=0\n' "$*" >&2
  exit 1
}

for value in "$A_SEC" "$B_SEC" "$C_SEC" "$D_SEC" "$E_SEC" "$SETTLE_SEC" "$DURATION"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration_value_$value"
done
[[ "$ACK" == "1" ]] || fail "set_V11_STEP8_TWO_PERSON_ACK=1_after_two_people_are_ready"
required=$((A_SEC + B_SEC + C_SEC + D_SEC + E_SEC + SETTLE_SEC * 4 + 15))
(( DURATION >= required )) || fail "duration_${DURATION}_too_short_min_${required}"

# Dedicated Devs ground-truth run. Keep all unrelated people outside CAM-01/CAM-04.
export V11_STEP4_PAIR_REQUIRE_DIFFERENT_ROOM=0

mkdir -p "$OUT"
: >"$DISPLAY_LOG"
: >"$MATCH_LOG"
: >"$LAUNCHER_LOG"
: >"$CHECK_LOG"
printf 'marker\tglobal_rows\tverify_rows\n' >"$PHASE_TSV"
rm -f "$PAIR_TSV" "$MATCH_TSV" "$GLOBAL_TSV" "$VERIFY_TSV"

printf '%s\n' \
  'V11_STEP8_CAM01_CAM04_TWO_PERSON_V1 READY' \
  'This is a STRICT manual ground-truth test for exactly two physical people in Devs.' \
  'Before launch: ONLY Person A must be visible to CAM-01 and CAM-04. Person B stays fully outside both cameras.' \
  'Do not allow any third person into either Devs camera for the entire run.' \
  'Follow the printed phases exactly. A/B identity is established from isolation, not clothing assumptions.'

# Keep cheap deterministic unit guards before the live test. Frozen Step1-3 remains untouched.
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step4_camera_tracklet_v1.py" >"$OUT/step4_tracklet_unit.log" 2>&1 || fail "step4_tracklet_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step5_global_shadow_v1.py" >"$OUT/step5_unit.log" 2>&1 || fail "step5_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step6_global_shadow_hysteresis_v1.py" >"$OUT/step6_unit.log" 2>&1 || fail "step6_unit_tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_frozen_step123_guard.py" >"$OUT/frozen_guard.log" 2>&1 || fail "frozen_step123_guard"

V11_STEP6_DISPLAY_LOG="$DISPLAY_LOG" \
V11_STEP6_GLOBAL_LOG="$MATCH_LOG" \
V11_STEP4_PAIR_TSV="$PAIR_TSV" \
V11_STEP4_MATCH_TSV="$MATCH_TSV" \
V11_STEP5_GLOBAL_TSV="$GLOBAL_TSV" \
V11_STEP6_VERIFY_TSV="$VERIFY_TSV" \
V11_STEP6_RUN_SEC="$DURATION" \
  timeout -s TERM "$((DURATION + 120))s" \
  bash "$ROOT/scripts/run_camera_v11_step6_global_shadow_v1.sh" \
  >"$LAUNCHER_LOG" 2>&1 &
launcher_pid=$!

# Wait until the actual Step6 runtime is alive; prime/warmup time is not counted as Phase A.
ready=0
for _ in $(seq 1 1200); do
  if grep -q '^CAMERA_V11_STEP5_GLOBAL_SHADOW_RUNNING ' "$LAUNCHER_LOG"; then
    ready=1
    break
  fi
  kill -0 "$launcher_pid" 2>/dev/null || break
  sleep 0.1
done
if (( ready != 1 )); then
  tail -n 200 "$LAUNCHER_LOG" >&2 || true
  fail "step6_runtime_not_ready"
fi

count_rows() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    printf '0'
    return
  fi
  local lines
  lines="$(wc -l <"$path")"
  (( lines > 0 )) && printf '%s' "$((lines - 1))" || printf '0'
}

mark_phase() {
  local marker="$1"
  local g v
  g="$(count_rows "$GLOBAL_TSV")"
  v="$(count_rows "$VERIFY_TSV")"
  printf '%s\t%s\t%s\n' "$marker" "$g" "$v" >>"$PHASE_TSV"
  printf 'V11_STEP8_PHASE marker=%s global_rows=%s verify_rows=%s\n' "$marker" "$g" "$v"
}

printf 'V11_STEP8_PHASE_A START seconds=%s condition=ONLY_PERSON_A_VISIBLE\n' "$A_SEC"
printf 'Keep Person A naturally moving in the shared CAM-01/CAM-04 coverage. Person B stays outside.\n'
sleep "$A_SEC"
mark_phase A_END

printf 'V11_STEP8_PHASE_B START settle=%ss measure=%ss condition=A_STAYS_AND_B_JOINS_SEPARATED\n' "$SETTLE_SEC" "$B_SEC"
printf 'Person A stays visible. Person B now enters and is also visible to both cameras. Keep them separated; do NOT cross yet.\n'
sleep "$SETTLE_SEC"
sleep "$B_SEC"
mark_phase B_END

printf 'V11_STEP8_PHASE_C START seconds=%s condition=BOTH_CROSS_AND_OCCLUDE\n' "$C_SEC"
printf 'Both people now walk across each other repeatedly. Cause brief natural occlusion, turns/back-view, and position swaps. Do not deliberately hide for long periods.\n'
sleep "$C_SEC"
mark_phase C_END

printf 'V11_STEP8_PHASE_D TRANSITION settle=%ss condition=ONLY_PERSON_A_REMAINS\n' "$SETTLE_SEC"
printf 'Person B leaves BOTH camera views completely. Person A remains visible.\n'
sleep "$SETTLE_SEC"
mark_phase D_START
printf 'V11_STEP8_PHASE_D MEASURE seconds=%s ground_truth=PERSON_A\n' "$D_SEC"
sleep "$D_SEC"
mark_phase D_END

printf 'V11_STEP8_PHASE_E TRANSITION settle=%ss condition=ONLY_PERSON_B_REMAINS\n' "$SETTLE_SEC"
printf 'Person A now leaves BOTH camera views completely. Person B enters/remains visible alone.\n'
sleep "$SETTLE_SEC"
mark_phase E_START
printf 'V11_STEP8_PHASE_E MEASURE seconds=%s ground_truth=PERSON_B\n' "$E_SEC"
sleep "$E_SEC"
mark_phase E_END

# Stop immediately after the final ground-truth window so no uncontrolled scene can
# append identity evidence after E_END.
if kill -0 "$launcher_pid" 2>/dev/null; then
  kill -TERM "$launcher_pid" 2>/dev/null || true
fi
wait "$launcher_pid" 2>/dev/null
status=$?
launcher_pid=""
if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
  tail -n 200 "$LAUNCHER_LOG" >&2 || true
  tail -n 240 "$MATCH_LOG" >&2 || true
  fail "launcher_status_$status"
fi

grep -q '^CAMERA_V11_STEP5_GLOBAL_SHADOW_NATURAL_PRIME result=PASS ' "$LAUNCHER_LOG" || fail "natural_prime_not_passed"
grep -q '^CAMERA_V11_STEP5_GLOBAL_SHADOW_PREFLIGHT result=PASS ' "$LAUNCHER_LOG" || fail "preflight_not_passed"
grep -q 'runtime_module=services.camera_v11.step6_global_shadow_runtime_v1' "$LAUNCHER_LOG" || fail "step6_runtime_module_not_selected"
grep -q '^CAMERA_V11_STEP6_GLOBAL_VERIFY_V1_ARCH ' "$MATCH_LOG" || fail "step6_arch_missing"

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/check_camera_v11_step8_cam01_cam04_two_person_v1.py" \
  --display-log "$DISPLAY_LOG" \
  --match-log "$MATCH_LOG" \
  --pair-tsv "$PAIR_TSV" \
  --match-tsv "$MATCH_TSV" \
  --global-tsv "$GLOBAL_TSV" \
  --verify-tsv "$VERIFY_TSV" \
  --phase-markers "$PHASE_TSV" \
  --warmup-windows 2 \
  --min-isolation-observations 3 \
  2>&1 | tee "$CHECK_LOG"
exit "${PIPESTATUS[0]}"
