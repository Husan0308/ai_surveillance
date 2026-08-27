#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
FROZEN_STEP2_SHA="1e06df789913075ab5a357d174584b6a5ebadf82"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step3_tracker_v1.lock"
DISPLAY_LOG="${V11_STEP3_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP3_DISPLAY.log}"
TRACKER_LOG="${V11_STEP3_TRACKER_LOG:-/tmp/CAMERA_V11_STEP3_TRACKER.log}"
fail() { printf 'CAMERA_V11_STEP3_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 8>"$LOCK_FILE"; flock -n 8 || fail "another Step3 launcher holds $LOCK_FILE"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is empty"
[[ -s "$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine" ]] || fail "FP32 engine missing"

# A stale camera runtime or TensorRT builder can consume extra RTSP/NVDEC/GPU
# resources and make an otherwise-good Step1/Step2 baseline look regressed.
# Fail closed and show the operator exactly what is still alive; never kill
# unrelated processes automatically from a production acceptance launcher.
CONFLICT_PATTERN='services\.camera_v11\.(step1_cam02_lowlat_v7|step2_production_fp32(_v[0-9]+)?|step3_tracking_v1)|yolo26_trt86_step2_worker\.py|build_yolo26s_b1_.*trt86\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting camera/TRT process already running:\n'"$conflicts"

git cat-file -e "${FROZEN_STEP2_SHA}^{commit}" 2>/dev/null || \
  fail "frozen Step2 commit $FROZEN_STEP2_SHA is missing locally"
git merge-base --is-ancestor "$FROZEN_STEP2_SHA" HEAD || \
  fail "current Step3 branch is not based on frozen Step2 commit $FROZEN_STEP2_SHA"

git diff --quiet "$FROZEN_STEP2_SHA" -- \
  services/camera_v11/step1_cam02_lowlat_v7.py \
  services/camera_v11/step1_independent_egl_v4.py \
  services/camera_v11/step2_production_fp32.py \
  services/camera_v11/step2_production_fp32_v12.py \
  services/camera_v11/step2_production_fp32_v13.py \
  services/camera_v11/step2_production_fp32_v18.py \
  services/camera_v11/step2_trt86.py \
  scripts/yolo26_trt86_step2_worker.py \
  scripts/run_camera_v11_step1_v7.sh \
  scripts/check_camera_v11_step1_v7_log.py \
  scripts/check_camera_v11_step2_production_log_v15.py \
  || fail "frozen Step1/Step2 differs from frozen Step2 commit $FROZEN_STEP2_SHA"

display_pid=""; tracker_pid=""
cleanup() {
  trap - EXIT INT TERM
  for pid in "$tracker_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$tracker_pid" "$display_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

: >"$DISPLAY_LOG"
: >"$TRACKER_LOG"
printf 'CAMERA_V11_STEP3_PREFLIGHT status=OK frozen_step2_sha=%s display_log=%s tracker_log=%s\n' \
  "$FROZEN_STEP2_SHA" "$DISPLAY_LOG" "$TRACKER_LOG"

bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$DISPLAY_LOG" 2>&1 & display_pid=$!
sleep "${V11_STEP3_DISPLAY_WARMUP_SEC:-8}"
kill -0 "$display_pid" 2>/dev/null || fail "frozen Step1 exited during warmup"

export V11_STEP2_MODE=full
export V11_STEP2_HZ="${V11_STEP3_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP3_CONF:-0.18}"
export V11_STEP2_ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
export V11_STEP2_TRT86_PYTHON="$ROOT/.venv-trt86/bin/python"
export V11_STEP2_TRT86_WORKER="$ROOT/scripts/yolo26_trt86_step2_worker.py"

"$ROOT/.venv/bin/python" -u -m services.camera_v11.step3_tracking_v1 >"$TRACKER_LOG" 2>&1 & tracker_pid=$!
printf 'CAMERA_V11_STEP3_RUNNING display_pid=%s tracker_pid=%s\n' "$display_pid" "$tracker_pid"
while kill -0 "$display_pid" 2>/dev/null && kill -0 "$tracker_pid" 2>/dev/null; do sleep 1; done
kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"
wait "$tracker_pid"
