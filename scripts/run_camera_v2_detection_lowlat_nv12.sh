#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Golden wall contract. ML is not allowed to change camera clarity/cadence.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-100}"
export CAMERA_V2_LOW_LATENCY_MODE="${CAMERA_V2_LOW_LATENCY_MODE:-0}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_FRAME_WIDTH="${CAMERA_V2_FRAME_WIDTH:-1280}"
export CAMERA_V2_FRAME_HEIGHT="${CAMERA_V2_FRAME_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"
export CAMERA_V2_MUX_TIMEOUT_US="${CAMERA_V2_MUX_TIMEOUT_US:-50000}"
export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.50}"
export CAMERA_V2_DETECTION_ONLY_STALL_SEC="${CAMERA_V2_DETECTION_ONLY_STALL_SEC:-12}"
export CAMERA_V2_DISPLAY_BBOX="${CAMERA_V2_DISPLAY_BBOX:-1}"
export CAMERA_V2_DISPLAY_BBOX_TTL_SEC="${CAMERA_V2_DISPLAY_BBOX_TTL_SEC:-3.2}"

# Primary: proven YOLO26s E2E TensorRT 8.6, B1, exact 672x384 input.
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS="${CAMERA_V2_DETECT_ACTIVE_CAMERAS:-CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.08}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-0.33}"
export CAMERA_V2_PRIMARY_HZ_MIN="${CAMERA_V2_PRIMARY_HZ_MIN:-0.28}"
export CAMERA_V2_PRIMARY_HZ_MAX="${CAMERA_V2_PRIMARY_HZ_MAX:-0.40}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-450}"

# Sparse S-pose: only ambiguous candidates, never every frame.
export CAMERA_V2_POSE_GATE_MODEL="${CAMERA_V2_POSE_GATE_MODEL:-yolo26s-pose.pt}"
export CAMERA_V2_POSE_GATE_DEVICE="${CAMERA_V2_POSE_GATE_DEVICE:-cpu}"
export CAMERA_V2_POSE_GATE_IMGSZ="${CAMERA_V2_POSE_GATE_IMGSZ:-224}"
export CAMERA_V2_POSE_GATE_THREADS="${CAMERA_V2_POSE_GATE_THREADS:-1}"
export CAMERA_V2_POSE_GATE_MIN_CONF="${CAMERA_V2_POSE_GATE_MIN_CONF:-0.08}"
export CAMERA_V2_POSE_GATE_STRONG_CONF="${CAMERA_V2_POSE_GATE_STRONG_CONF:-0.30}"
export CAMERA_V2_POSE_GATE_FALLBACK_CONF="${CAMERA_V2_POSE_GATE_FALLBACK_CONF:-0.18}"
export CAMERA_V2_POSE_GATE_MODEL_CONF="${CAMERA_V2_POSE_GATE_MODEL_CONF:-0.03}"
export CAMERA_V2_POSE_GATE_MAX_CANDIDATES="${CAMERA_V2_POSE_GATE_MAX_CANDIDATES:-2}"
export CAMERA_V2_POSE_GATE_TIMEOUT_SEC="${CAMERA_V2_POSE_GATE_TIMEOUT_SEC:-0.55}"
export CAMERA_V2_POSE_GATE_PADDING="${CAMERA_V2_POSE_GATE_PADDING:-0.14}"
export CAMERA_V2_POSE_GATE_POSITIVE_TTL_SEC="${CAMERA_V2_POSE_GATE_POSITIVE_TTL_SEC:-12}"
export CAMERA_V2_POSE_GATE_NEGATIVE_TTL_SEC=0
export CAMERA_V2_POSE_GATE_SOFT_KEEP_CONF="${CAMERA_V2_POSE_GATE_SOFT_KEEP_CONF:-0.12}"
export CAMERA_V2_POSE_GATE_REJECT_HITS="${CAMERA_V2_POSE_GATE_REJECT_HITS:-2}"
export CAMERA_V2_POSE_GATE_REJECT_WINDOW_SEC="${CAMERA_V2_POSE_GATE_REJECT_WINDOW_SEC:-10}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v3.py}"

# CAM-05 rescue. Same 672x384 captured frame: no second converter/resize/branch.
export CAMERA_V2_RESCUE_ENABLED="${CAMERA_V2_RESCUE_ENABLED:-1}"
export CAMERA_V2_RESCUE_CAMERA="${CAMERA_V2_RESCUE_CAMERA:-CAM-05}"
export CAMERA_V2_RESCUE_CONF="${CAMERA_V2_RESCUE_CONF:-0.08}"
export CAMERA_V2_RESCUE_TRIGGER_CONF="${CAMERA_V2_RESCUE_TRIGGER_CONF:-0.18}"
export CAMERA_V2_RESCUE_MAX_DET="${CAMERA_V2_RESCUE_MAX_DET:-40}"
export CAMERA_V2_RESCUE_MIN_INTERVAL_SEC="${CAMERA_V2_RESCUE_MIN_INTERVAL_SEC:-3.0}"
export CAMERA_V2_RESCUE_GPU_DUTY="${CAMERA_V2_RESCUE_GPU_DUTY:-0.08}"
export CAMERA_V2_RESCUE_TRT86_ENGINE="${CAMERA_V2_RESCUE_TRT86_ENGINE:-$ROOT/artifacts/yolo26m_trt86/yolo26m-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_RESCUE_TRT86_SHM_WORKER="${CAMERA_V2_RESCUE_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v3.py}"

unset CAMERA_V2_PARITY_CAPTURE_CAMERAS CAMERA_V2_PARITY_SAMPLES_PER_CAMERA CAMERA_V2_PARITY_DIR || true
export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

PRIMARY_RESTORE="$ROOT/scripts/restore_cam01_trt86_engine.sh"
RESCUE_PREP="$ROOT/scripts/prepare_yolo26m_672_trt86.sh"
MODULE="services.camera_v2.detection_lowlat_nv12"

fail() { printf 'CAMERA_LOWLAT_NV12_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin nvstreammux nvmultistreamtiler nvvideoconvert nveglglessink nvdsosd appsink tee queue; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing GStreamer/DeepStream plugin: $plugin"
done

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing/not executable: $CAMERA_V2_TRT86_PYTHON"
if [[ ! -s "$CAMERA_V2_TRT86_ENGINE" && -f "$PRIMARY_RESTORE" ]]; then
  bash "$PRIMARY_RESTORE" "$CAMERA_V2_TRT86_ENGINE" || true
fi
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "primary TRT86 engine missing: $CAMERA_V2_TRT86_ENGINE"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "primary TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

"$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import tensorrt as trt
if not str(trt.__version__).startswith('8.6.1'):
    raise SystemExit(f'CAMERA_LOWLAT_NV12_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}')
print(f'CAMERA_LOWLAT_NV12_TRT tensorrt={trt.__version__}', flush=True)
PY

if [[ "$CAMERA_V2_RESCUE_ENABLED" == "1" ]]; then
  if [[ ! -s "$CAMERA_V2_RESCUE_TRT86_ENGINE" ]]; then
    echo "CAMERA_LOWLAT_NV12_RESCUE engine_missing=1 action=prepare_m672" >&2
    if ! bash "$RESCUE_PREP"; then
      echo "CAMERA_LOWLAT_NV12_RESCUE WARNING prepare failed; continuing primary-only" >&2
      export CAMERA_V2_RESCUE_ENABLED=0
    fi
  fi
  if [[ "$CAMERA_V2_RESCUE_ENABLED" == "1" && ! -s "$CAMERA_V2_RESCUE_TRT86_ENGINE" ]]; then
    echo "CAMERA_LOWLAT_NV12_RESCUE WARNING engine still missing; continuing primary-only" >&2
    export CAMERA_V2_RESCUE_ENABLED=0
  fi
fi

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: F401
import numpy, yaml, dotenv, torch, ultralytics  # noqa: F401
import services.camera_v2.detection_lowlat_nv12  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import NV12 low-latency runtime"

printf '%s\n' \
  "CAMERA_LOWLAT_NV12_PROFILE source=6xRTSP@20 rtsp=TCP/${CAMERA_V2_RTSP_LATENCY_MS}ms mux=${CAMERA_V2_FRAME_WIDTH}x${CAMERA_V2_FRAME_HEIGHT} wall=${CAMERA_V2_WALL_WIDTH}x${CAMERA_V2_WALL_HEIGHT}" \
  "CAMERA_LOWLAT_NV12_PRIMARY model=YOLO26s engine=TRT8.6/672x384/B1 target=${CAMERA_V2_DETECT_TARGET_HZ}Hz/cam adaptive=${CAMERA_V2_PRIMARY_HZ_MIN}-${CAMERA_V2_PRIMARY_HZ_MAX} conf=${CAMERA_V2_DETECT_CONF}" \
  "CAMERA_LOWLAT_NV12_RESCUE enabled=${CAMERA_V2_RESCUE_ENABLED} camera=${CAMERA_V2_RESCUE_CAMERA} model=YOLO26m engine=TRT8.6/672x384/B1 same_frame=1 duty=${CAMERA_V2_RESCUE_GPU_DUTY} min_interval=${CAMERA_V2_RESCUE_MIN_INTERVAL_SEC}s" \
  "CAMERA_LOWLAT_NV12_FLOW display=wall->nvdsosd(NV12-direct)->EGL rgba_wall_convert=0 ML=isolated-tee gate=preconvert capture_after_gpu_slot=1 nvdcf=0 nms=0 dedup=0" \
  "CAMERA_LOWLAT_NV12_MAIN executable=$MAIN_PYTHON module=$MODULE"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m "$MODULE"
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count; (( delay > 10 )) && delay=10
  echo "CAMERA_LOWLAT_NV12_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
