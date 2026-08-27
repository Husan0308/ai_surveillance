#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
TRT_PY="${CAMERA_V8_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
EXPORT_PY="${CAMERA_V8_EXPORT_PYTHON:-$ROOT/.venv/bin/python}"
ONNX_OUT="${CAMERA_V8_YOLO_ONNX_OUT:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b6.onnx}"
ENGINE_OUT="${CAMERA_V8_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b6-fp32-trt86.engine}"
CACHE_OUT="${CAMERA_V8_TRT_TIMING_CACHE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b6-trt86.timing.cache}"
OPT_LEVEL="${CAMERA_V8_TRT_BUILD_OPT_LEVEL:-2}"

fail() { printf 'V8_MODEL_PREP FAIL %s\n' "$*" >&2; exit 1; }
[[ -x "$TRT_PY" ]] || fail "TRT8.6 python missing: $TRT_PY"

validate_engine() {
  "$TRT_PY" - "$1" <<'PY'
import sys
from pathlib import Path
import tensorrt as trt
p = Path(sys.argv[1])
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"TensorRT 8.6.1 required, got {trt.__version__}")
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, "")
engine = trt.Runtime(logger).deserialize_cuda_engine(p.read_bytes())
if engine is None:
    raise SystemExit("deserialize failed")
ctx = engine.create_execution_context()
ins = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
outs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
if len(ins) != 1 or len(outs) != 1:
    raise SystemExit(f"bad bindings inputs={ins} outputs={outs}")
i = tuple(int(v) for v in ctx.get_binding_shape(ins[0]))
o = tuple(int(v) for v in ctx.get_binding_shape(outs[0]))
if i != (6,3,384,672) or o != (6,300,6):
    raise SystemExit(f"wrong shapes input={i} output={o}")
print(f"V8_ENGINE_VALIDATE PASS engine={p} input={i} output={o} trt={trt.__version__}")
PY
}

if [[ -s "$ENGINE_OUT" ]]; then
  if validate_engine "$ENGINE_OUT"; then
    echo "V8_MODEL_PREP status=READY engine=$ENGINE_OUT"
    exit 0
  fi
  echo "V8_MODEL_PREP existing_engine_invalid=$ENGINE_OUT rebuilding=1" >&2
  rm -f "$ENGINE_OUT"
fi

# Only trust an explicitly supplied ONNX or the V8-specific batch-6 export path.
ONNX=""
for candidate in "${CAMERA_V8_YOLO_ONNX:-}" "$ONNX_OUT"; do
  [[ -n "$candidate" && -s "$candidate" ]] || continue
  ONNX="$candidate"
  break
done

if [[ -z "$ONNX" ]]; then
  [[ -x "$EXPORT_PY" ]] || fail "export python missing: $EXPORT_PY"
  if ! "$EXPORT_PY" -c 'import ultralytics' >/dev/null 2>&1; then
    fail "ultralytics missing in $EXPORT_PY; use the environment that originally exported YOLO26 and set CAMERA_V8_EXPORT_PYTHON"
  fi

  MODEL=""
  for candidate in \
    "${CAMERA_V8_YOLO_PT:-}" \
    "$ROOT/yolo26s.pt" \
    "$ROOT/models/yolo26s.pt" \
    "$ROOT/artifacts/yolo26s.pt" \
    "$HOME/.cache/ultralytics/yolo26s.pt"; do
    [[ -n "$candidate" && -s "$candidate" ]] || continue
    MODEL="$candidate"
    break
  done

  if [[ -n "$MODEL" ]]; then
    "$EXPORT_PY" "$ROOT/scripts/export_yolo26_batch6_onnx_v8.py" \
      --model "$MODEL" \
      --output "$ONNX_OUT"
  else
    "$EXPORT_PY" "$ROOT/scripts/export_yolo26_batch6_onnx_v8.py" \
      --model yolo26s.pt \
      --allow-download \
      --output "$ONNX_OUT"
  fi
  ONNX="$ONNX_OUT"
fi

mkdir -p "$(dirname "$ENGINE_OUT")"
echo "V8_MODEL_PREP build_start=1 onnx=$ONNX engine=$ENGINE_OUT opt_level=$OPT_LEVEL timing_cache=$CACHE_OUT"
echo "V8_MODEL_PREP note='First Pascal batch-6 build can take several minutes. TensorRT INFO lines now show progress; do not start the camera runtime until this command finishes.'"
"$TRT_PY" "$ROOT/scripts/build_yolo26_batch6_trt86_v8.py" \
  --onnx "$ONNX" \
  --engine "$ENGINE_OUT" \
  --workspace-gib "${CAMERA_V8_TRT_WORKSPACE_GIB:-1.0}" \
  --optimization-level "$OPT_LEVEL" \
  --timing-cache "$CACHE_OUT"
validate_engine "$ENGINE_OUT"
echo "V8_MODEL_PREP status=READY onnx=$ONNX engine=$ENGINE_OUT timing_cache=$CACHE_OUT"
