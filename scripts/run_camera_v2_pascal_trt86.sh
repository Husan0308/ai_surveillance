#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-100}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
# GTX 1050 Ti smoothness profile. 960x540 remains 16:9 and cuts mux/tracker
# pixels by 44% versus 1280x720. The visible 3x2 wall is 1440x540 so each tile
# is an exact 480x270 16:9 surface.
export CAMERA_V2_FRAME_WIDTH="${CAMERA_V2_FRAME_WIDTH:-960}"
export CAMERA_V2_FRAME_HEIGHT="${CAMERA_V2_FRAME_HEIGHT:-540}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1440}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-540}"
export CAMERA_V2_MUX_TIMEOUT_US="${CAMERA_V2_MUX_TIMEOUT_US:-50000}"
export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.50}"
export CAMERA_V2_PASCAL_STALL_SEC="${CAMERA_V2_PASCAL_STALL_SEC:-12}"

export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS="${CAMERA_V2_DETECT_ACTIVE_CAMERAS:-CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.05}"
export CAMERA_V2_DETECT_IOU="${CAMERA_V2_DETECT_IOU:-0.70}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"
# Measured TRT8.6 B1 FP32 inference is ~160-190 ms. 0.33 Hz/cam means roughly
# two TRT calls/sec total (~32-38% detector GPU occupancy) rather than ~54% at
# 0.50 Hz/cam, leaving enough headroom for NVDEC, NvDCF, tiler and EGL.
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-0.33}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-0.30}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-0.40}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-350}"
# Preserve 16:9-ish tracker geometry while reducing tracker pixels ~22%.
export CAMERA_V2_TRACKER_WIDTH="${CAMERA_V2_TRACKER_WIDTH:-448}"
export CAMERA_V2_TRACKER_HEIGHT="${CAMERA_V2_TRACKER_HEIGHT:-256}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.00}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v3.py}"
RESTORE_HELPER="$ROOT/scripts/restore_cam01_trt86_engine.sh"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() { printf 'CAMERA_PASCAL_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin nvstreammux nvtracker nvmultistreamtiler nvvideoconvert nvdsosd nveglglessink appsink tee queue; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing GStreamer/DeepStream plugin: $plugin"
done

GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
[[ -n "$GPU_LINE" ]] || GPU_LINE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "CAMERA_PASCAL_GPU ${GPU_LINE:-unknown}"

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing/not executable: $CAMERA_V2_TRT86_PYTHON"
if [[ ! -s "$CAMERA_V2_TRT86_ENGINE" && -f "$RESTORE_HELPER" ]]; then
  echo "CAMERA_PASCAL_ENGINE missing=1 recovery=stash/local-search" >&2
  bash "$RESTORE_HELPER" "$CAMERA_V2_TRT86_ENGINE" || true
fi
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TRT8.6 engine missing: $CAMERA_V2_TRT86_ENGINE"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

"$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import sys
import numpy as np
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAMERA_PASCAL_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAMERA_PASCAL_TRT trt_python={sys.executable} tensorrt={trt.__version__} numpy={np.__version__}", flush=True)
PY

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy, yaml, dotenv  # noqa: F401
import services.camera_v2.pascal_runtime  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import Pascal Camera V2 runtime"

printf '%s\n' \
  "CAMERA_PASCAL_PROFILE display=6xRTSP/${CAMERA_V2_FRAME_WIDTH}x${CAMERA_V2_FRAME_HEIGHT}@20 wall=${CAMERA_V2_WALL_WIDTH}x${CAMERA_V2_WALL_HEIGHT} detector=TRT8.6/B1/FP32/672x384 active=${CAMERA_V2_DETECT_ACTIVE_CAMERAS} target=${CAMERA_V2_DETECT_TARGET_HZ}Hz/cam" \
  "CAMERA_PASCAL_PIPELINE DeepStream=RTSP/NVDEC->tee->mux->detector-meta->NvDCF->tiler->OSD->EGL detector=separate-process/SHM nvinfer=0 trt10=0 scaling=bilinear" \
  "CAMERA_PASCAL_RECOVERY internal-retries=3 process-watchdog=${CAMERA_V2_PASCAL_STALL_SEC}s stagger=${CAMERA_V2_STARTUP_STAGGER_SEC}s per-source-recycle=0" \
  "CAMERA_PASCAL_MAIN_PYTHON executable=$MAIN_PYTHON"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.pascal_runtime
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count; (( delay > 10 )) && delay=10
  echo "CAMERA_PASCAL_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
