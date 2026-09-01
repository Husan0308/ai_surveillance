#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP7_OUT:-/tmp/camera_v11_step7_cam01_cam04_one_person_v1}"
DURATION="${V11_STEP7_DURATION_SEC:-90}"
ACK="${V11_STEP7_ONE_PERSON_ACK:-0}"
PAIR_TSV="$ROOT/artifacts/reid/step4_pair_scores_v1.tsv"
MATCH_TSV="$ROOT/artifacts/reid/step4_same_room_matches_v1.tsv"
GLOBAL_TSV="$ROOT/artifacts/reid/step5_global_shadow_v1.tsv"
VERIFY_TSV="$ROOT/artifacts/reid/step6_global_verify_v1.tsv"

fail() {
  printf 'V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 RESULT=FAIL reason=%s physical_people_expected=1 verified_ids_expected=1 production_global_id=0 identity_accuracy_proven=0\n' "$*" >&2
  exit 1
}

[[ "$ACK" == "1" ]] || fail "set_V11_STEP7_ONE_PERSON_ACK=1_after_room_is_ready"
[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration"
(( DURATION >= 75 )) || fail "duration_must_be_at_least_75_seconds"

# This is intentionally a Devs-only ground-truth run with exactly one physical
# person. Do not require unrelated Entrance/Main-room pair evidence in the generic
# Step3 pair checker; every other pair-score invariant remains enabled.
export V11_STEP4_PAIR_REQUIRE_DIFFERENT_ROOM=0

mkdir -p "$OUT"
printf '%s\n' \
  'V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 READY' \
  'Physical test condition: EXACTLY ONE person in Devs coverage.' \
  'That same person must be visible to CAM-01 and CAM-04 during the run.' \
  'Do not allow a second person into either Devs camera.' \
  'Move naturally: walk, turn, briefly face away, and cross the shared field of view.' \
  'Acceptance is strict: exactly one confirmed+verified shadow ID, one canonical CAM-01/CAM-04 pair, zero conflicts/holds/expiries.'

# Give the operator a short deterministic setup window after acknowledging the room state.
for value in 5 4 3 2 1; do
  printf 'V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 START_IN=%ss\n' "$value"
  sleep 1
done

V11_STEP6_ACCEPTANCE_OUT="$OUT" \
V11_STEP6_DURATION_SEC="$DURATION" \
  bash "$ROOT/scripts/run_camera_v11_step6_global_shadow_acceptance_v1.sh" \
  2>&1 | tee "$OUT/step6_acceptance.log"
status=${PIPESTATUS[0]}
if (( status != 0 )); then
  fail "step6_acceptance_status_$status"
fi

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/check_camera_v11_step7_cam01_cam04_one_person_v1.py" \
  --display-log "$OUT/display.log" \
  --match-log "$OUT/match.log" \
  --pair-tsv "$PAIR_TSV" \
  --match-tsv "$MATCH_TSV" \
  --global-tsv "$GLOBAL_TSV" \
  --verify-tsv "$VERIFY_TSV" \
  --warmup-windows 2 \
  2>&1 | tee "$OUT/step7_check.log"
exit "${PIPESTATUS[0]}"
