#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PT_PY="${CAMERA_V2_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
TRT_PY="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
ONNX="$ROOT/artifacts/yolo26m_trt86/yolo26m-672x384-b1-e2e.onnx"
ENGINE="$ROOT/artifacts/yolo26m_trt86/yolo26m-672x384-b1-fp32-trt86.engine"

fail() { printf 'YOLO26M_RESCUE_PREP ERROR: %s\n' "$*" >&2; exit 1; }
[[ -x "$PT_PY" ]] || fail "main python missing/not executable: $PT_PY"
[[ -x "$TRT_PY" ]] || fail "TRT86 python missing/not executable: $TRT_PY"

"$TRT_PY" - <<'PY'
import tensorrt as trt
if not str(trt.__version__).startswith('8.6.1'):
    raise SystemExit(f'YOLO26M_RESCUE_PREP ERROR: TensorRT 8.6.1 required, got {trt.__version__}')
print(f'YOLO26M_RESCUE_TRT tensorrt={trt.__version__}', flush=True)
PY

if [[ ! -s "$ENGINE" ]]; then
  if [[ ! -s "$ONNX" ]]; then
    echo "YOLO26M_RESCUE_PREP step=export_onnx input=672x384 model=yolo26m" >&2
    "$PT_PY" "$ROOT/scripts/export_yolo26m_672_onnx.py"
  fi
  echo "YOLO26M_RESCUE_PREP step=build_trt86 precision=fp32 batch=1" >&2
  "$TRT_PY" "$ROOT/scripts/build_yolo26m_672_trt86.py"
fi

[[ -s "$ENGINE" ]] || fail "engine missing after build: $ENGINE"

CAMERA_V2_RESCUE_TRT86_ENGINE="$ENGINE" \
"$TRT_PY" "$ROOT/scripts/benchmark_yolo26m_672_trt86.py" \
  --engine "$ENGINE" \
  --dir "$ROOT/.runtime/yolo26_parity" \
  --warmup 2 \
  --runs 3 || true

printf 'YOLO26M_RESCUE_ENGINE_READY path=%s bytes=%s input=672x384 precision=fp32 trt=8.6.1\n' \
  "$ENGINE" "$(stat -c%s "$ENGINE")"
