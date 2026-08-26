#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Stable six-camera DeepStream ingest/display.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=100
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_WALL_WIDTH=1920
export CAMERA_V2_WALL_HEIGHT=720
export CAMERA_V2_RTSP_RECONNECT_ATTEMPTS=3
export CAMERA_V2_SOURCE_WATCHDOG_STALL_SEC=6
export CAMERA_V2_SOURCE_WATCHDOG_GRACE_SEC=6
export CAMERA_V2_SOURCE_RECYCLE_PAUSE_MS=750
export CAMERA_V2_SOURCE_MAX_RECYCLES=2

# Pose gets the larger analysis frame; Ultralytics letterboxes internally to 832.
export CAMERA_V2_DETECT_WIDTH=1280
export CAMERA_V2_DETECT_HEIGHT=720
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_CONF=0.03
export CAMERA_V2_DETECT_IOU=0.80
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_DETECT_STARTUP_DELAY=0.5
export CAMERA_V2_POSE_IMGSZ=832
export CAMERA_V2_POSE_CONF=0.03
export CAMERA_V2_POSE_IOU=0.80

if [[ -f "$ROOT/yolo26s-pose.pt" ]]; then
  export CAMERA_V2_POSE_MODEL="$ROOT/yolo26s-pose.pt"
else
  unset CAMERA_V2_POSE_MODEL || true
fi

# Sparse pose refresh; NvDCF owns every display frame between observations.
export CAMERA_V2_DETECT_TARGET_HZ=2.0
export CAMERA_V2_DETECT_MIN_HZ=2.0
export CAMERA_V2_DETECT_MAX_HZ=2.0
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS=800
export CAMERA_V2_POSE_MAX_PROJECTION_S=0.45
export CAMERA_V2_POSE_PROJECTION_GAIN=0.82
export CAMERA_V2_EMPTY_CONFIRM_MISSES=5

# Local tracker baseline. V2 verifies the final generated YAML before nvtracker.
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF=0.00
export CAMERA_V2_TRACK_BOX_SIDE_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_TOP_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN=0.00
export CAMERA_V2_DEDUP_IOU=0.82
export CAMERA_V2_DEDUP_CONTAINMENT=0.94

# Global identity remains off until bbox tracking is proven green.
export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true
unset CAMERA_V2_TRT86_ENGINE CAMERA_V2_TRT86_PYTHON CAMERA_V2_TRT86_WORKER CAMERA_V2_TRT86_SHM_WORKER || true

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import torch
import ultralytics  # noqa: F401
import services.camera_v2.person_tracking_pose_sticky_v2  # noqa: F401
assert torch.cuda.is_available()
PY
  then
    MAIN_PYTHON="$candidate"
    break
  fi
done

[[ -n "$MAIN_PYTHON" ]] || { echo "CAM01_POSE_V2_PREFLIGHT ERROR: no compatible Python" >&2; exit 1; }

printf '%s\n' \
  "CAM01_POSE_V2_PROFILE capture=1280x720 imgsz=832 conf=0.03 active=CAM-01 target=2.0Hz" \
  "CAM01_POSE_V2_TRACKER NvDCF=512x288 shadow=100frames empty-confirm=5 final-yaml-verified=1 display_conf=0.00" \
  "CAM01_POSE_V2_PIPELINE rtsp=tcp/100ms staggered=1 watchdog=6s/recycle2 process-restart=1 one-shot=1 no-prefetch=1 global-id=off"

# Exit code 75 means the Python watchdog saw an nvurisrcbin that remained wedged
# after bounded per-source recycle attempts. Restart the whole pipeline cleanly;
# source stagger prevents a reconnect storm against the NVR.
restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.person_tracking_pose_sticky_v2
  rc=$?
  set -e

  if [[ $rc -ne 75 ]]; then
    exit "$rc"
  fi

  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAM01_POSE_V2_SUPERVISOR restart=${restart_count} reason=rtsp-watchdog delay=${delay}s" >&2
  sleep "$delay"
done
