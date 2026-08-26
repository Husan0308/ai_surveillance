#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Native DeepStream geometry: decode stays NVMM, mux analytics surface is 720p.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=100
export CAMERA_V2_SOURCE_FPS=20
export CAMERA_V2_EXTRA_SURFACES=6
export CAMERA_V2_FRAME_WIDTH=1280
export CAMERA_V2_FRAME_HEIGHT=720
export CAMERA_V2_WALL_WIDTH=1920
export CAMERA_V2_WALL_HEIGHT=720
export CAMERA_V2_STARTUP_STAGGER_SEC=0.5
export CAMERA_V2_NATIVE_STALL_SEC=12

PARSER_DIR="$ROOT/services/camera_v2/native_yolo26"
PARSER_SO="$PARSER_DIR/libnvdsinfer_custom_yolo26_e2e.so"
ONNX="$ROOT/artifacts/yolo26s_deepstream/yolo26s-672x384-b6-e2e.onnx"
ENGINE="$ROOT/.runtime/camera_v2/yolo26s-672x384-b6-fp16-deepstream.engine"

for plugin in nvurisrcbin nvstreammux nvinfer nvtracker nvmultistreamtiler nvvideoconvert nvdsosd nveglglessink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || {
    echo "CAMERA_NATIVE_PREFLIGHT ERROR: missing GStreamer/DeepStream plugin: $plugin" >&2
    exit 1
  }
done

DS_ROOT=""
for candidate in /opt/nvidia/deepstream/deepstream /opt/nvidia/deepstream/deepstream-*; do
  [[ -f "$candidate/sources/includes/nvdsinfer_custom_impl.h" ]] || continue
  DS_ROOT="$candidate"
  break
done
[[ -n "$DS_ROOT" ]] || {
  echo "CAMERA_NATIVE_PREFLIGHT ERROR: DeepStream headers not found" >&2
  exit 1
}

# DeepStream 7.1 dGPU is documented against CUDA 12.6. Prefer that toolkit when
# installed, but support the /usr/local/cuda alternative or an nvcc-derived root.
CUDA_HOME_RESOLVED="${CUDA_HOME:-}"
if [[ -z "$CUDA_HOME_RESOLVED" || ! -f "$CUDA_HOME_RESOLVED/include/cuda_runtime_api.h" ]]; then
  CUDA_HOME_RESOLVED=""
  for candidate in /usr/local/cuda-12.6 /usr/local/cuda; do
    [[ -f "$candidate/include/cuda_runtime_api.h" ]] || continue
    CUDA_HOME_RESOLVED="$(readlink -f "$candidate")"
    break
  done
fi
if [[ -z "$CUDA_HOME_RESOLVED" ]]; then
  NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "$NVCC_BIN" ]]; then
    candidate="$(cd "$(dirname "$NVCC_BIN")/.." && pwd -P)"
    if [[ -f "$candidate/include/cuda_runtime_api.h" ]]; then
      CUDA_HOME_RESOLVED="$candidate"
    fi
  fi
fi
if [[ -z "$CUDA_HOME_RESOLVED" ]]; then
  for candidate in /usr/local/cuda-*; do
    [[ -f "$candidate/include/cuda_runtime_api.h" ]] || continue
    CUDA_HOME_RESOLVED="$candidate"
    break
  done
fi
[[ -n "$CUDA_HOME_RESOLVED" ]] || {
  cat >&2 <<'EOF'
CAMERA_NATIVE_PREFLIGHT ERROR: CUDA Toolkit headers were not found.
Expected cuda_runtime_api.h under one of:
  /usr/local/cuda-12.6/include
  /usr/local/cuda/include
  <nvcc-root>/include

DeepStream 7.1 dGPU officially uses CUDA Toolkit 12.6.
EOF
  exit 1
}
export CUDA_HOME="$CUDA_HOME_RESOLVED"
CUDA_VERSION_TEXT="unknown"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  CUDA_VERSION_TEXT="$($CUDA_HOME/bin/nvcc --version | tail -n 1 | sed 's/^[[:space:]]*//')"
fi
echo "CAMERA_NATIVE_CUDA home=$CUDA_HOME version=$CUDA_VERSION_TEXT"

if [[ ! -f "$PARSER_SO" || "$PARSER_DIR/nvdsparsebbox_yolo26_e2e.cpp" -nt "$PARSER_SO" || "$PARSER_DIR/Makefile" -nt "$PARSER_SO" ]]; then
  echo "CAMERA_NATIVE_BUILD parser=YOLO26-E2E ds_root=$DS_ROOT cuda_home=$CUDA_HOME"
  make -C "$PARSER_DIR" clean all DS_ROOT="$DS_ROOT" CUDA_HOME="$CUDA_HOME"
fi

if [[ ! -f "$ONNX" ]]; then
  cat >&2 <<EOF
CAMERA_NATIVE_PREFLIGHT ERROR: native batch-6 YOLO26 ONNX is missing:
  $ONNX

Create it once with:
  .venv/bin/python scripts/export_yolo26s_deepstream_onnx.py --model yolo26s.pt

The ONNX is local deployment data and is intentionally not committed to Git.
EOF
  exit 2
fi

# Let Gst-nvinfer rebuild a stale engine from ONNX using the TensorRT version
# linked to the installed DeepStream runtime.
if [[ -f "$ENGINE" && "$ONNX" -nt "$ENGINE" ]]; then
  echo "CAMERA_NATIVE_ENGINE stale=1 action=remove path=$ENGINE"
  rm -f "$ENGINE"
fi

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy  # noqa: F401
import yaml  # noqa: F401
import dotenv  # noqa: F401
import services.camera_v2.person_tracking_native_deepstream_v2  # noqa: F401
PY
  then
    MAIN_PYTHON="$candidate"
    break
  fi
done
[[ -n "$MAIN_PYTHON" ]] || {
  echo "CAMERA_NATIVE_PREFLIGHT ERROR: no Python can import Camera V2 runtime" >&2
  exit 1
}

printf '%s\n' \
  "CAMERA_NATIVE_PROFILE source=6xRTSP mux=1280x720/b6 wall=1920x720" \
  "CAMERA_NATIVE_ANALYTICS pgie=nvinfer/YOLO26-E2E/672x384/fp16/interval19 tracker=NvDCF/640x384" \
  "CAMERA_NATIVE_ZERO_COPY appsink=0 numpy-detector=0 pytorch=0 manual-meta-injection=0" \
  "CAMERA_NATIVE_RECOVERY internal=nvurisrcbin whole-process-watchdog=12s per-source-recycle=0"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.person_tracking_native_deepstream_v2
  rc=$?
  set -e

  if [[ $rc -ne 75 ]]; then
    exit "$rc"
  fi

  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_NATIVE_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
