#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Keep the proven camera presentation geometry. ML is added around it rather than
# reducing the wall to a blurry low-resolution profile.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-100}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_FRAME_WIDTH="${CAMERA_V2_FRAME_WIDTH:-1280}"
export CAMERA_V2_FRAME_HEIGHT="${CAMERA_V2_FRAME_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"
export CAMERA_V2_MUX_TIMEOUT_US="${CAMERA_V2_MUX_TIMEOUT_US:-50000}"
export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.50}"
export CAMERA_V2_PASCAL_STALL_SEC="${CAMERA_V2_PASCAL_STALL_SEC:-12}"

# Primary detector: Pascal-safe TensorRT 8.6 sidecar. The low raw threshold lets
# pose decide ambiguous candidates instead of throwing them away too early.
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS="${CAMERA_V2_DETECT_ACTIVE_CAMERAS:-CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.08}"
export CAMERA_V2_DETECT_IOU="${CAMERA_V2_DETECT_IOU:-0.70}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"
# 0.25 Hz/camera = 1.5 serial TRT calls/sec across six streams. This protects the
# 20 FPS wall while NvDCF owns the between-detection motion.
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-0.25}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-0.20}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-0.30}"
# CameraPersonTrackingTRT86Fresh caps its adaptive budget at 600 ms. Keep this at
# that ceiling so a bounded CPU pose crop can finish without accepting old frames.
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-600}"
export CAMERA_V2_TRACKER_WIDTH="${CAMERA_V2_TRACKER_WIDTH:-512}"
export CAMERA_V2_TRACKER_HEIGHT="${CAMERA_V2_TRACKER_HEIGHT:-288}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.00}"

# Pose is a validator, not the primary full-frame detector. It runs only on
# ambiguous crops and defaults to CPU so it does not steal GPU time from the
# clean wall/TRT8.6/NvDCF path.
export CAMERA_V2_POSE_GATE_DEVICE="${CAMERA_V2_POSE_GATE_DEVICE:-cpu}"
export CAMERA_V2_POSE_GATE_IMGSZ="${CAMERA_V2_POSE_GATE_IMGSZ:-224}"
export CAMERA_V2_POSE_GATE_THREADS="${CAMERA_V2_POSE_GATE_THREADS:-2}"
export CAMERA_V2_POSE_GATE_MIN_CONF="${CAMERA_V2_POSE_GATE_MIN_CONF:-0.08}"
export CAMERA_V2_POSE_GATE_STRONG_CONF="${CAMERA_V2_POSE_GATE_STRONG_CONF:-0.35}"
export CAMERA_V2_POSE_GATE_FALLBACK_CONF="${CAMERA_V2_POSE_GATE_FALLBACK_CONF:-0.25}"
export CAMERA_V2_POSE_GATE_MODEL_CONF="${CAMERA_V2_POSE_GATE_MODEL_CONF:-0.03}"
export CAMERA_V2_POSE_GATE_MAX_CANDIDATES="${CAMERA_V2_POSE_GATE_MAX_CANDIDATES:-6}"
export CAMERA_V2_POSE_GATE_TIMEOUT_SEC="${CAMERA_V2_POSE_GATE_TIMEOUT_SEC:-0.35}"
export CAMERA_V2_POSE_GATE_PADDING="${CAMERA_V2_POSE_GATE_PADDING:-0.12}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v3.py}"
RESTORE_HELPER="$ROOT/scripts/restore_cam01_trt86_engine.sh"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() { printf 'CAMERA_ML_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin nvstreammux nvtracker nvmultistreamtiler nvvideoconvert nvdsosd nveglglessink appsink tee queue; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing GStreamer/DeepStream plugin: $plugin"
done

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing/not executable: $CAMERA_V2_TRT86_PYTHON"
if [[ ! -s "$CAMERA_V2_TRT86_ENGINE" && -f "$RESTORE_HELPER" ]]; then
  echo "CAMERA_ML_ENGINE missing=1 recovery=stash/local-search" >&2
  bash "$RESTORE_HELPER" "$CAMERA_V2_TRT86_ENGINE" || true
fi
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TRT8.6 engine missing: $CAMERA_V2_TRT86_ENGINE"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

"$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAMERA_ML_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAMERA_ML_TRT tensorrt={trt.__version__}", flush=True)
PY

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy, yaml, dotenv, torch, ultralytics  # noqa: F401
import services.camera_v2.person_tracking_trt86_pose_gate  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import Camera V2 + CPU pose-gate runtime"

printf '%s\n' \
  "CAMERA_ML_PROFILE wall=${CAMERA_V2_WALL_WIDTH}x${CAMERA_V2_WALL_HEIGHT} mux=${CAMERA_V2_FRAME_WIDTH}x${CAMERA_V2_FRAME_HEIGHT} source=6xRTSP@20" \
  "CAMERA_ML_DETECTOR YOLO26s=TRT8.6/B1/FP32/672x384 target=${CAMERA_V2_DETECT_TARGET_HZ}Hz/cam raw_conf=${CAMERA_V2_DETECT_CONF}" \
  "CAMERA_ML_POSE_GATE device=${CAMERA_V2_POSE_GATE_DEVICE} imgsz=${CAMERA_V2_POSE_GATE_IMGSZ} ambiguous=${CAMERA_V2_POSE_GATE_MIN_CONF}-${CAMERA_V2_POSE_GATE_STRONG_CONF} fallback=${CAMERA_V2_POSE_GATE_FALLBACK_CONF}" \
  "CAMERA_ML_FLOW YOLO26s->confidence-gate->pose(crop-only)->NvDCF tracker=${CAMERA_V2_TRACKER_WIDTH}x${CAMERA_V2_TRACKER_HEIGHT}" \
  "CAMERA_ML_DISABLED global_id=off reid=off face=off nvinfer=off trt10=off" \
  "CAMERA_ML_MAIN_PYTHON executable=$MAIN_PYTHON"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.person_tracking_trt86_pose_gate
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count; (( delay > 10 )) && delay=10
  echo "CAMERA_ML_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
