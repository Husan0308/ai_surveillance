#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Keep the exact clear camera-wall geometry that passed the camera-only test.
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

# Primary detector. At 0.35 Hz/camera a new person gets a real YOLO observation
# about every 2.9 s instead of every ~5 s in the previous profile.
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS="${CAMERA_V2_DETECT_ACTIVE_CAMERAS:-CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.06}"
export CAMERA_V2_DETECT_IOU="${CAMERA_V2_DETECT_IOU:-0.70}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-0.35}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-0.30}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-0.40}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-600}"

# Keep more spatial detail than 448x256; reduce NvDCF compute through features and
# 3:3 sub-batching in the Python runtime instead of making the tracker image tiny.
export CAMERA_V2_TRACKER_WIDTH="${CAMERA_V2_TRACKER_WIDTH:-512}"
export CAMERA_V2_TRACKER_HEIGHT="${CAMERA_V2_TRACKER_HEIGHT:-288}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.05}"

# Two geometry de-dup passes remain before NvDCF. NvDCF itself has an additional
# new-target / duplicate guard in the V2 runtime.
export CAMERA_V2_RAW_DEDUP_IOU="${CAMERA_V2_RAW_DEDUP_IOU:-0.72}"
export CAMERA_V2_RAW_DEDUP_CONTAINMENT="${CAMERA_V2_RAW_DEDUP_CONTAINMENT:-0.88}"
export CAMERA_V2_DEDUP_IOU="${CAMERA_V2_DEDUP_IOU:-0.72}"
export CAMERA_V2_DEDUP_CONTAINMENT="${CAMERA_V2_DEDUP_CONTAINMENT:-0.88}"

# Pose is a sparse validator only. S-pose is kept for accuracy, but only one CPU
# thread and max two genuinely new ambiguous crops are allowed. A first negative
# pose result is a provisional hold; only repeated negative evidence rejects it.
export CAMERA_V2_POSE_GATE_MODEL="${CAMERA_V2_POSE_GATE_MODEL:-yolo26s-pose.pt}"
export CAMERA_V2_POSE_GATE_DEVICE="${CAMERA_V2_POSE_GATE_DEVICE:-cpu}"
export CAMERA_V2_POSE_GATE_IMGSZ="${CAMERA_V2_POSE_GATE_IMGSZ:-224}"
export CAMERA_V2_POSE_GATE_THREADS="${CAMERA_V2_POSE_GATE_THREADS:-1}"
export CAMERA_V2_POSE_GATE_MIN_CONF="${CAMERA_V2_POSE_GATE_MIN_CONF:-0.06}"
export CAMERA_V2_POSE_GATE_STRONG_CONF="${CAMERA_V2_POSE_GATE_STRONG_CONF:-0.30}"
export CAMERA_V2_POSE_GATE_FALLBACK_CONF="${CAMERA_V2_POSE_GATE_FALLBACK_CONF:-0.20}"
export CAMERA_V2_POSE_GATE_MODEL_CONF="${CAMERA_V2_POSE_GATE_MODEL_CONF:-0.03}"
export CAMERA_V2_POSE_GATE_MAX_CANDIDATES="${CAMERA_V2_POSE_GATE_MAX_CANDIDATES:-2}"
export CAMERA_V2_POSE_GATE_TIMEOUT_SEC="${CAMERA_V2_POSE_GATE_TIMEOUT_SEC:-0.55}"
export CAMERA_V2_POSE_GATE_PADDING="${CAMERA_V2_POSE_GATE_PADDING:-0.14}"
export CAMERA_V2_POSE_GATE_POSITIVE_TTL_SEC="${CAMERA_V2_POSE_GATE_POSITIVE_TTL_SEC:-15}"
export CAMERA_V2_POSE_GATE_NEGATIVE_TTL_SEC=0
export CAMERA_V2_POSE_TRACK_REUSE_MAX_AGE_SEC="${CAMERA_V2_POSE_TRACK_REUSE_MAX_AGE_SEC:-0.75}"
export CAMERA_V2_POSE_GATE_SOFT_KEEP_CONF="${CAMERA_V2_POSE_GATE_SOFT_KEEP_CONF:-0.14}"
export CAMERA_V2_POSE_GATE_REJECT_HITS="${CAMERA_V2_POSE_GATE_REJECT_HITS:-2}"
export CAMERA_V2_POSE_GATE_REJECT_WINDOW_SEC="${CAMERA_V2_POSE_GATE_REJECT_WINDOW_SEC:-10}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v3.py}"
RESTORE_HELPER="$ROOT/scripts/restore_cam01_trt86_engine.sh"
MODULE="services.camera_v2.person_tracking_trt86_pose_gate_v2"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() { printf 'CAMERA_ML_V2_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin nvstreammux nvtracker nvmultistreamtiler nvvideoconvert nvdsosd nveglglessink appsink tee queue; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing GStreamer/DeepStream plugin: $plugin"
done

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing/not executable: $CAMERA_V2_TRT86_PYTHON"
if [[ ! -s "$CAMERA_V2_TRT86_ENGINE" && -f "$RESTORE_HELPER" ]]; then
  echo "CAMERA_ML_V2_ENGINE missing=1 recovery=stash/local-search" >&2
  bash "$RESTORE_HELPER" "$CAMERA_V2_TRT86_ENGINE" || true
fi
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TRT8.6 engine missing: $CAMERA_V2_TRT86_ENGINE"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

"$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAMERA_ML_V2_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAMERA_ML_V2_TRT tensorrt={trt.__version__}", flush=True)
PY

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy, yaml, dotenv, torch, ultralytics  # noqa: F401
import services.camera_v2.pose_gate_v3  # noqa: F401
import services.camera_v2.person_tracking_trt86_pose_gate_v2  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import ML V2 runtime"

printf '%s\n' \
  "CAMERA_ML_V2_PROFILE source=6xRTSP@20 rtsp=TCP/${CAMERA_V2_RTSP_LATENCY_MS}ms mux=${CAMERA_V2_FRAME_WIDTH}x${CAMERA_V2_FRAME_HEIGHT} wall=${CAMERA_V2_WALL_WIDTH}x${CAMERA_V2_WALL_HEIGHT}" \
  "CAMERA_ML_V2_DETECTOR model=YOLO26s/TRT8.6/672x384/B1 target=${CAMERA_V2_DETECT_TARGET_HZ}Hz/cam raw_conf=${CAMERA_V2_DETECT_CONF}" \
  "CAMERA_ML_V2_TRACKER NvDCF=${CAMERA_V2_TRACKER_WIDTH}x${CAMERA_V2_TRACKER_HEIGHT} subbatch=3:3 ColorNames=1 HOG=0" \
  "CAMERA_ML_V2_POSE model=${CAMERA_V2_POSE_GATE_MODEL} device=cpu imgsz=${CAMERA_V2_POSE_GATE_IMGSZ} threads=${CAMERA_V2_POSE_GATE_THREADS} max_new=${CAMERA_V2_POSE_GATE_MAX_CANDIDATES} reject_hits=${CAMERA_V2_POSE_GATE_REJECT_HITS}" \
  "CAMERA_ML_V2_FLOW YOLO->dedup->track/cache->S-pose(two-hit reject)->NvDCF; display_geometry_preserved=1" \
  "CAMERA_ML_V2_MAIN_PYTHON executable=$MAIN_PYTHON module=$MODULE"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m "$MODULE"
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count; (( delay > 10 )) && delay=10
  echo "CAMERA_ML_V2_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
