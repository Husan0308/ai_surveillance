#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="rebuild/service-architecture-v11-clean-step1-cam02-lowlat-v7-20260827"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step2_production_v18.lock"
DISPLAY_LOG="${V11_STEP2_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP2_DISPLAY.log}"
DETECTOR_LOG="${V11_STEP2_DETECTOR_LOG:-/tmp/CAMERA_V11_STEP2_DETECTOR.log}"
fail() { printf 'CAMERA_V11_STEP2_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 8>"$LOCK_FILE"; flock -n 8 || fail "another Step2 production launcher holds $LOCK_FILE"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is empty"
[[ -s "$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine" ]] || fail "FP32 engine missing"
git diff --quiet "$BASE" -- services/camera_v11/step1_cam02_lowlat_v7.py \
  services/camera_v11/step1_independent_egl_v4.py scripts/run_camera_v11_step1_v7.sh \
  scripts/check_camera_v11_step1_v7_log.py || fail "frozen Step1 differs from authoritative V7"
display_pid=""; detector_pid=""
cleanup() {
  trap - EXIT INT TERM
  for pid in "$detector_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$detector_pid" "$display_pid"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
: >"$DISPLAY_LOG"; : >"$DETECTOR_LOG"
printf 'CAMERA_V11_STEP2_PREFLIGHT status=OK base=%s precision=fp32 display_log=%s detector_log=%s\n' \
  "$BASE" "$DISPLAY_LOG" "$DETECTOR_LOG"
bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$DISPLAY_LOG" 2>&1 & display_pid=$!
sleep "${V11_STEP2_DISPLAY_WARMUP_SEC:-8}"
kill -0 "$display_pid" 2>/dev/null || fail "frozen Step1 exited during warmup"
"$ROOT/scripts/run_camera_v11_step2_stage_v18.sh" full >"$DETECTOR_LOG" 2>&1 & detector_pid=$!
printf 'CAMERA_V11_STEP2_RUNNING display_pid=%s detector_pid=%s\n' "$display_pid" "$detector_pid"
while kill -0 "$display_pid" 2>/dev/null && kill -0 "$detector_pid" 2>/dev/null; do sleep 1; done
kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"; wait "$detector_pid"
