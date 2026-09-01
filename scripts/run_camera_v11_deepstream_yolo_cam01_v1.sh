#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
BRANCH_EXPECTED="rebuild/service-architecture-v11-deepstream-yolo-cam01-v1-20260901"
RUNTIME_DIR="$ROOT/.runtime/camera_v11/ds_yolo_cam01_v1"
ARTIFACT_DIR="$ROOT/artifacts/yolo26s_ds71"
ONNX="${V11_DS_YOLO_ONNX:-$ARTIFACT_DIR/yolo26s-672x384-b1-nms.onnx}"
ENGINE="${V11_DS_YOLO_ENGINE:-$ARTIFACT_DIR/yolo26s-672x384-b1-fp32-trt103.engine}"
PARSER_LIB="${V11_DS_YOLO_PARSER_LIB:-$ARTIFACT_DIR/libnvdsinfer_yolo26_v1.so}"
META_LIB="${V11_DS_YOLO_META_LIB:-$ARTIFACT_DIR/libcamera_v11_ds_yolo_meta_v1.so}"
CONFIG="$RUNTIME_DIR/nvinfer_yolo26_cam01_v1.txt"
LABELS="$RUNTIME_DIR/labels.txt"
LOG="${V11_DS_YOLO_LOG:-/tmp/CAMERA_V11_DS_YOLO_CAM01.log}"
APP_PY="${V11_DS_YOLO_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"

fail() {
  printf 'V11_DS_YOLO_CAM01_PREFLIGHT RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

find_ds_root() {
  if [[ -n "${DEEPSTREAM_ROOT:-}" && -f "$DEEPSTREAM_ROOT/sources/includes/nvdsinfer_custom_impl.h" ]]; then
    printf '%s\n' "$DEEPSTREAM_ROOT"
    return 0
  fi
  for candidate in /opt/nvidia/deepstream/deepstream /opt/nvidia/deepstream/deepstream-7.1; do
    if [[ -f "$candidate/sources/includes/nvdsinfer_custom_impl.h" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

DS_ROOT="$(find_ds_root)" || fail "deepstream_7_1_headers_missing"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
for plugin in nvurisrcbin queue nvstreammux nvinfer nvvideoconvert capsfilter nvdsosd nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing_plugin=$plugin"
done

# DeepStream 7.1 uses TensorRT 10.3. Do not point nvinfer at the accepted TRT 8.6
# sidecar engine. Build a native DS7.1 engine from fixed batch-1 ONNX instead.
OLD_TRT86="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
if [[ -s "$OLD_TRT86" ]]; then
  printf 'V11_DS_YOLO_CAM01_ENGINE_POLICY old_trt86=%s action=ignore reason=ds71_trt103_native_engine_required\n' "$OLD_TRT86"
fi

mkdir -p "$RUNTIME_DIR" "$ARTIFACT_DIR"

bash scripts/build_camera_v11_deepstream_yolo26_parser_v1.sh
[[ -s "$PARSER_LIB" ]] || fail "parser_lib_missing=$PARSER_LIB"
[[ -s "$META_LIB" ]] || fail "meta_lib_missing=$META_LIB"

if [[ ! -s "$ONNX" ]]; then
  MODEL="${V11_DS_YOLO_MODEL:-}"
  if [[ -z "$MODEL" ]]; then
    for candidate in \
      "$ROOT/yolo26s.pt" \
      "$ROOT/artifacts/yolo26s.pt" \
      "$ROOT/models/yolo26s.pt" \
      "$HOME/yolo26s.pt"; do
      if [[ -s "$candidate" ]]; then
        MODEL="$candidate"
        break
      fi
    done
  fi
  [[ -n "$MODEL" && -s "$MODEL" ]] || fail "onnx_missing=$ONNX set_V11_DS_YOLO_MODEL_to_local_yolo26s_pt"
  "$APP_PY" scripts/export_yolo26_b1_onnx_ds71_v1.py \
    --model "$MODEL" \
    --output "$ONNX"
fi
[[ -s "$ONNX" ]] || fail "onnx_missing_after_export=$ONNX"

printf 'Person\n' >"$LABELS"
INTERVAL="${V11_DS_YOLO_INTERVAL:-9}"
CONF="${V11_DS_YOLO_CONF:-0.18}"
GPU_ID="${V11_GPU_ID:-0}"
cat >"$CONFIG" <<EOF
[property]
gpu-id=$GPU_ID
onnx-file=$ONNX
model-engine-file=$ENGINE
labelfile-path=$LABELS
batch-size=1
network-mode=0
num-detected-classes=1
interval=$INTERVAL
gie-unique-id=1
process-mode=1
network-type=0
net-scale-factor=0.00392156862745098
model-color-format=0
maintain-aspect-ratio=1
symmetric-padding=1
cluster-mode=4
custom-lib-path=$PARSER_LIB
parse-bbox-func-name=NvDsInferParseCustomYolo26V1
workspace-size=1024

[class-attrs-all]
pre-cluster-threshold=$CONF
EOF

# Catch old camera/detector processes before opening a second CAM-01 RTSP session.
CONFLICT_PATTERN='services\.camera_v11\.(step1_|step2_|step3_|deepstream_yolo_cam01_v1)|yolo26_trt86_step2_worker\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_process:\n'"$conflicts"

export DEEPSTREAM_ROOT="$DS_ROOT"
export LD_LIBRARY_PATH="$DS_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V11_DS_YOLO_CAMERA="${V11_DS_YOLO_CAMERA:-CAM-01}"
export V11_DS_YOLO_CONFIG="$CONFIG"
export V11_DS_YOLO_META_LIB="$META_LIB"
export V11_DS_YOLO_INTERVAL="$INTERVAL"
export V11_RTSP_TRANSPORT=tcp
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-100}"
export V11_DROP_ON_LATENCY="${V11_DROP_ON_LATENCY:-1}"
export V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}"
export V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"

printf 'V11_DS_YOLO_CAM01_PREFLIGHT RESULT=PASS branch=%s camera=%s onnx=%s engine=%s interval=%s conf=%s ds_root=%s\n' \
  "$BRANCH_EXPECTED" "$V11_DS_YOLO_CAMERA" "$ONNX" "$ENGINE" "$INTERVAL" "$CONF" "$DS_ROOT"
printf 'V11_DS_YOLO_CAM01_NOTE first_run_engine_build=%s log=%s\n' "$([[ -s "$ENGINE" ]] && echo 0 || echo 1)" "$LOG"

: >"$LOG"
"$APP_PY" -u -m services.camera_v11.deepstream_yolo_cam01_v1 2>&1 | tee "$LOG"
