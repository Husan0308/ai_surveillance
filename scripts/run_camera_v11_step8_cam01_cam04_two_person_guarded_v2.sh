#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP8_OUT:-/tmp/camera_v11_step8_cam01_cam04_two_person_v1}"
MATCH_LOG="$OUT/match.log"
PHASE_STATE="$OUT/step8_phase_state.txt"
INNER="$ROOT/scripts/run_camera_v11_step8_cam01_cam04_two_person_acceptance_v1.sh"
GRACE_SEC="${V11_STEP8_SCENE_GUARD_GRACE_SEC:-10}"
MISMATCH_WINDOWS="${V11_STEP8_SCENE_GUARD_MISMATCH_WINDOWS:-2}"
runner_pid=""
monitor_pid=""

fail() {
  printf 'V11_STEP8_SCENE_GUARD_V2 RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -x "$INNER" || -f "$INNER" ]] || fail "inner_runner_missing"
[[ "$GRACE_SEC" =~ ^[0-9]+$ ]] || fail "invalid_grace_sec"
[[ "$MISMATCH_WINDOWS" =~ ^[1-9][0-9]*$ ]] || fail "invalid_mismatch_windows"

cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
    kill -TERM "$runner_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

expected_for_phase() {
  local phase="$1"
  case "$phase" in
    PREP*|"PHASE A"*|"TRANSITION D"*|"PHASE D"*|"TRANSITION E"*|"PHASE E"*)
      printf '1 1'
      ;;
    "PHASE B"*|"PHASE C"*)
      printf '2 2'
      ;;
    *)
      printf '0 0'
      ;;
  esac
}

parse_visible_counts() {
  local line="$1"
  "$ROOT/.venv/bin/python" - "$line" <<'PY'
import re
import sys
line = sys.argv[1]
values = []
for camera in ("CAM-01", "CAM-04"):
    match = re.search(rf"{re.escape(camera)}:[^|]*?visible=(\d+)", line)
    values.append(match.group(1) if match else "-1")
print(" ".join(values))
PY
}

monitor_scene() {
  local last_line=""
  local phase=""
  local phase_started=$SECONDS
  local mismatch=0
  local expected_a expected_b counts visible_a visible_b latest new_phase

  while kill -0 "$runner_pid" 2>/dev/null; do
    if [[ -s "$PHASE_STATE" ]]; then
      new_phase="$(tr '\n' ' ' <"$PHASE_STATE" | sed 's/[[:space:]]\+/ /g')"
    else
      new_phase="PREP"
    fi
    if [[ "$new_phase" != "$phase" ]]; then
      phase="$new_phase"
      phase_started=$SECONDS
      mismatch=0
      last_line=""
      printf 'V11_STEP8_SCENE_GUARD_V2 phase=%q grace_sec=%s\n' "$phase" "$GRACE_SEC"
    fi

    read -r expected_a expected_b <<<"$(expected_for_phase "$phase")"
    if (( expected_a == 0 )); then
      sleep 1
      continue
    fi
    if (( SECONDS - phase_started < GRACE_SEC )); then
      sleep 1
      continue
    fi
    if [[ ! -s "$MATCH_LOG" ]]; then
      sleep 1
      continue
    fi

    latest="$(grep '^CAMERA_V11_STEP3_V2_TRACKER ' "$MATCH_LOG" | tail -n 1 || true)"
    if [[ -z "$latest" || "$latest" == "$last_line" ]]; then
      sleep 1
      continue
    fi
    last_line="$latest"
    counts="$(parse_visible_counts "$latest")"
    read -r visible_a visible_b <<<"$counts"
    if [[ "$visible_a" == "$expected_a" && "$visible_b" == "$expected_b" ]]; then
      mismatch=0
      printf 'V11_STEP8_SCENE_GUARD_V2 scene=OK phase=%q CAM-01=%s CAM-04=%s expected=%s+%s\n' \
        "$phase" "$visible_a" "$visible_b" "$expected_a" "$expected_b"
      sleep 1
      continue
    fi

    mismatch=$((mismatch + 1))
    printf 'V11_STEP8_SCENE_GUARD_V2 scene=MISMATCH phase=%q CAM-01=%s CAM-04=%s expected=%s+%s mismatch_windows=%s/%s\n' \
      "$phase" "$visible_a" "$visible_b" "$expected_a" "$expected_b" "$mismatch" "$MISMATCH_WINDOWS" >&2
    if (( mismatch >= MISMATCH_WINDOWS )); then
      printf 'V11_STEP8_SCENE_GUARD_V2 RESULT=FAIL reason=invalid_ground_truth_scene phase=%q CAM-01=%s CAM-04=%s expected=%s+%s\n' \
        "$phase" "$visible_a" "$visible_b" "$expected_a" "$expected_b" >&2
      kill -TERM "$runner_pid" 2>/dev/null || true
      return 42
    fi
    sleep 1
  done
  return 0
}

mkdir -p "$OUT"
printf 'V11_STEP8_SCENE_GUARD_V2 READY expected=A/D/E:1+1 B/C:2+2 grace_sec=%s mismatch_windows=%s\n' \
  "$GRACE_SEC" "$MISMATCH_WINDOWS"

V11_STEP8_OUT="$OUT" bash "$INNER" &
runner_pid=$!
monitor_scene &
monitor_pid=$!

wait "$runner_pid"
runner_status=$?
runner_pid=""

wait "$monitor_pid"
monitor_status=$?
monitor_pid=""

if (( monitor_status == 42 )); then
  exit 42
fi
exit "$runner_status"
