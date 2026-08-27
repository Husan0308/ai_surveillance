#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"
PROFILE_ENV="${CAMERA_V2_V74_PROFILE_ENV:-/tmp/camera_v74_profile.env}"

fail() { printf 'CAMERA_BBOX_V74_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

command -v flock >/dev/null 2>&1 || fail "flock missing (install util-linux)"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Camera V2 owner already holds $LOCK_FILE"

# Never stack a new runtime over a repo-owned CUDA sidecar.
mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -nu)
repo_gpu=()
for pid in "${GPU_PIDS[@]:-}"; do
  [[ -r "/proc/$pid/status" ]] || continue
  uid="$(awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
  [[ "$uid" == "$(id -u)" ]] || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  if [[ "$cwd" == "$ROOT" && ( "$cmd" == *"yolo26_trt86_shm_worker"* || "$cmd" == *"services.camera_v2"* || "$cmd" == *"multiprocessing.spawn"* ) ]]; then
    repo_gpu+=("$pid")
  fi
done
((${#repo_gpu[@]} == 0)) || fail "repo CUDA process already active pid(s)=${repo_gpu[*]}; run cleanup first"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing: $CAMERA_V2_TRT86_PYTHON"
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TRT86 engine missing: $CAMERA_V2_TRT86_ENGINE"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

# Measure THIS GPU/engine pair and budget detector cadence from the measured baseline.
# Override CAMERA_V2_SKIP_AUTO_PROFILE=1 only for controlled A/B tests.
if [[ "${CAMERA_V2_SKIP_AUTO_PROFILE:-0}" != "1" ]]; then
  rm -f "$PROFILE_ENV"
  python scripts/auto_profile_camera_v74.py --output "$PROFILE_ENV" --cameras 6 || fail "auto profile failed"
  [[ -s "$PROFILE_ENV" ]] || fail "profile env not written: $PROFILE_ENV"
  # shellcheck disable=SC1090
  source "$PROFILE_ENV"
fi

export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-80}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-8}"

export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

export CAMERA_V2_TRACK_WIDTH="${CAMERA_V2_TRACK_WIDTH:-512}"
export CAMERA_V2_TRACK_HEIGHT="${CAMERA_V2_TRACK_HEIGHT:-288}"
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-10}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.28}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-390}"
export CAMERA_V2_DISPLAY_EMPTY_HOLD_MS="${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS:-350}"

export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN="${CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN:-0.06}"
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN="${CAMERA_V2_DISPLAY_BOX_TOP_MARGIN:-0.04}"
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN="${CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN:-0.07}"
export CAMERA_V2_DISPLAY_SIZE_HOLD_SEC="${CAMERA_V2_DISPLAY_SIZE_HOLD_SEC:-0.22}"
export CAMERA_V2_DISPLAY_SHRINK_ALPHA="${CAMERA_V2_DISPLAY_SHRINK_ALPHA:-0.42}"
export CAMERA_V2_TRACK_JUMP_DIAG_LIMIT="${CAMERA_V2_TRACK_JUMP_DIAG_LIMIT:-1.00}"

export CAMERA_V2_DETECT_HZ="${CAMERA_V2_DETECT_HZ:-0.80}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.18}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-20}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-320}"
export CAMERA_V2_DETECT_ENABLED="${CAMERA_V2_DETECT_ENABLED:-1}"
export CAMERA_V2_ANALYTICS_ENABLED="${CAMERA_V2_ANALYTICS_ENABLED:-1}"

export CAMERA_V2_NVDCF_MIN_DETECTOR_CONF="${CAMERA_V2_NVDCF_MIN_DETECTOR_CONF:-0.18}"
export CAMERA_V2_NVDCF_MIN_IOU_DIFF_NEW_TARGET="${CAMERA_V2_NVDCF_MIN_IOU_DIFF_NEW_TARGET:-0.22}"
export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.50}"
export CAMERA_V2_SOURCE_STALL_SEC="${CAMERA_V2_SOURCE_STALL_SEC:-12}"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

for plugin in nvurisrcbin tee queue nvstreammux nvmultistreamtiler nvvideoconvert appsink nvtracker nvdsosd nveglglessink fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing DeepStream/GStreamer plugin: $plugin"
done

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
import numpy, yaml, dotenv  # noqa: F401
from services.camera_v2.runtime_bbox_v74 import PascalBalancedBBoxRuntime  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import V7.4 runtime"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' \
  "CAMERA_BBOX_V74_PREFLIGHT status=OK single_owner=1" \
  "CAMERA_BBOX_V74_PROFILE trt_baseline=${CAMERA_V2_TRT_BASELINE_MS:-unknown}ms source=${CAMERA_V2_SOURCE_FPS}fps display=${CAMERA_V2_DISPLAY_WIDTH}x${CAMERA_V2_DISPLAY_HEIGHT} tracker=${CAMERA_V2_TRACK_WIDTH}x${CAMERA_V2_TRACK_HEIGHT}@${CAMERA_V2_TRACK_FPS}Hz detector=672x384@${CAMERA_V2_DETECT_HZ}Hz/cam" \
  "CAMERA_BBOX_V74_POLICY nvdcf=ColorNames/level2/HOG0 detector_budget=auto~30pct empty_hold=${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS}ms predictor=0 shadow_render=0" \
  "CAMERA_BBOX_V74_PIPELINE decode-once display=20fps analytics=budgeted gpu=pascal-balanced"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.runtime_bbox_v74
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_BBOX_V74_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
