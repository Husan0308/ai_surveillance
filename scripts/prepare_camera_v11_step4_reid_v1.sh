#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
MODEL_DIR="$ROOT/artifacts/reid"
ONNX="${V11_STEP4_REID_ONNX:-$MODEL_DIR/resnet50_market1501_aicity156.onnx}"
ENGINE="${V11_STEP4_REID_ENGINE:-$MODEL_DIR/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine}"
URL="${V11_STEP4_REID_ONNX_URL:-https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet/deployable_v1.2/files/resnet50_market1501_aicity156.onnx}"
PYTHON="$ROOT/.venv-trt86/bin/python"
mkdir -p "$MODEL_DIR"

fail() {
  printf 'V11_STEP4_REID_PREPARE RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -x "$PYTHON" ]] || fail "trt86_python_missing path=$PYTHON"

if [[ ! -s "$ONNX" ]]; then
  tmp="$ONNX.download"
  rm -f "$tmp"
  printf 'V11_STEP4_REID_PREPARE DOWNLOAD source=official-ngc url=%s\n' "$URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 \
      --output "$tmp" "$URL" || fail "onnx_download_failed tool=curl"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=20 -O "$tmp" "$URL" \
      || fail "onnx_download_failed tool=wget"
  else
    fail "curl_or_wget_missing"
  fi
  size="$(stat -c '%s' "$tmp" 2>/dev/null || printf '0')"
  (( size > 1000000 )) || fail "onnx_download_too_small bytes=$size"
  mv -f "$tmp" "$ONNX"
  printf 'V11_STEP4_REID_PREPARE DOWNLOAD_RESULT=PASS path=%s bytes=%s\n' "$ONNX" "$size"
else
  printf 'V11_STEP4_REID_PREPARE ONNX_REUSE path=%s bytes=%s\n' \
    "$ONNX" "$(stat -c '%s' "$ONNX")"
fi

if [[ -s "$ENGINE" ]]; then
  printf 'V11_STEP4_REID_PREPARE ENGINE_REUSE path=%s bytes=%s\n' \
    "$ENGINE" "$(stat -c '%s' "$ENGINE")"
  printf 'V11_STEP4_REID_PREPARE RESULT=PASS source=reuse engine=%s\n' "$ENGINE"
  exit 0
fi

"$PYTHON" "$ROOT/scripts/build_camera_v11_step4_reid_trt86.py" \
  --onnx "$ONNX" \
  --engine "$ENGINE" \
  --workspace-gib "${V11_STEP4_REID_WORKSPACE_GIB:-1}" \
  --optimization-level "${V11_STEP4_REID_OPT_LEVEL:-3}"

[[ -s "$ENGINE" ]] || fail "engine_not_created path=$ENGINE"
printf 'V11_STEP4_REID_PREPARE RESULT=PASS source=build engine=%s bytes=%s\n' \
  "$ENGINE" "$(stat -c '%s' "$ENGINE")"
