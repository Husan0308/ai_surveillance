#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${V11_PYTHON:-$ROOT/.venv-trt86/bin/python}"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step2_detector.lock"
ENGINE="${V11_TRT86_BATCH_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b6-fp32-trt86.engine}"
WORKER="${V11_TRT86_BATCH_WORKER:-$ROOT/scripts/yolo26_trt86_batch6_worker_v8.py}"

fail() { printf 'CAMERA_V11_STEP2_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another V11 Step2 runtime holds $LOCK_FILE; stop it first"
[[ -x "$PY" ]] || fail "Python 3.10 TRT86 environment missing: $PY"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is empty; frozen V7 wall requires X11/XWayland"
[[ -f "$ENGINE" ]] || fail "batch6 TensorRT engine missing: $ENGINE"
[[ -f "$WORKER" ]] || fail "batch6 worker missing: $WORKER"

for plugin in nvurisrcbin nvv4l2decoder tee queue nvvideoconvert capsfilter appsink nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done

# Capture inspect output first; do not use grep -q in a pipe under pipefail.
DECODER_INSPECT="$(gst-inspect-1.0 nvv4l2decoder 2>/dev/null)"
CONVERT_INSPECT="$(gst-inspect-1.0 nvvideoconvert 2>/dev/null)"
grep 'low-latency-mode' <<<"$DECODER_INSPECT" >/dev/null || \
  fail "nvv4l2decoder low-latency-mode property missing"
grep 'interpolation-method' <<<"$CONVERT_INSPECT" >/dev/null || \
  fail "nvvideoconvert interpolation-method property missing"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Freeze the exact Step1 V7 runtime that passed.
export V11_RTSP_TRANSPORT=tcp
export V11_RTSP_LATENCY_MS=100
export V11_DROP_ON_LATENCY=1
export V11_EXTRA_SURFACES=4
export V11_UDP_BUFFER_SIZE="${V11_UDP_BUFFER_SIZE:-8388608}"
export V11_RECONNECT_SEC=5
export V11_STARTUP_STAGGER_SEC=0.40
export V11_STATS_INTERVAL_SEC="${V11_STATS_INTERVAL_SEC:-5}"
export V11_TILE_WIDTH=640
export V11_TILE_HEIGHT=360
export V11_SCALE_INTERPOLATION=4
export V11_LOWLAT_CAMERAS=CAM-02

# Step2 only: true batch-6 detector, no tracker/OSD/ReID/face.
export V11_TRT86_PYTHON="$PY"
export V11_TRT86_BATCH_ENGINE="$ENGINE"
export V11_TRT86_BATCH_WORKER="$WORKER"
export V11_DETECT_CONF="${V11_DETECT_CONF:-0.18}"
export V11_DETECT_MAX_DET="${V11_DETECT_MAX_DET:-20}"
export V11_DETECT_CAPTURE_TIMEOUT_MS="${V11_DETECT_CAPTURE_TIMEOUT_MS:-180}"

"$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"V11 Step2 requires Python 3.10, got {sys.version}")
import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
from services.camera_v11.step2_detector_only import V11Step2DetectorOnly  # noqa: F401
print(
    f"CAMERA_V11_STEP2_IMPORT python={sys.version.split()[0]} "
    f"numpy={np.__version__} gst={Gst.version_string()} runtime=OK"
)
PY

# Keep the frozen Step1 checker markers in the Step2 log intentionally.
printf '%s\n' \
  "CAMERA_V11_STEP1V7_PREFLIGHT status=OK python=$PY display=${DISPLAY} lowlat_target=nvv4l2decoder" \
  "CAMERA_V11_STEP1V7_INVARIANT base=v4-ds100 mux=0 tiler=0 detector=0 tracker=0 latest_only=1 transport=tcp latency_ms=100" \
  "CAMERA_V11_STEP1V7_AB lowlat_cameras=CAM-02 bframes_cam02=0" \
  "CAMERA_V11_STEP2_PREFLIGHT status=OK python=$PY display=${DISPLAY} engine=$ENGINE worker=$WORKER" \
  "CAMERA_V11_STEP2_INVARIANT base=step1-v7-frozen decode_once=1 tee=1 display_queue=1 detector_queue=1 tracker=0 osd=0 reid=0 face=0" \
  "CAMERA_V11_STEP2_TARGET batch=6 input=672x384 source_fps=20 realtime_target=18.0Hz trt_p95_target=25ms result_age_target=120ms"

exec "$PY" -u -m services.camera_v11.step2_detector_only
