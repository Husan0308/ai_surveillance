#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
MODEL_DIR="$ROOT/artifacts/reid"
ONNX="${V11_STEP4_REID_ONNX:-$MODEL_DIR/resnet50_market1501_aicity156.onnx}"
ENGINE="${V11_STEP4_REID_ENGINE:-$MODEL_DIR/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine}"
URL="${V11_STEP4_REID_ONNX_URL:-https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.2/files/resnet50_market1501_aicity156.onnx}"
EXPECTED_SHA256="${V11_STEP4_REID_ONNX_SHA256-0e21d09278508ec835955f422a9fdd3cd59b2a6ecdef98d705f388f33cebac2b}"
PYTHON="$ROOT/.venv-trt86/bin/python"
mkdir -p "$MODEL_DIR"

fail() {
  printf 'V11_STEP4_REID_PREPARE RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

verify_onnx() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local size
  size="$(stat -c '%s' "$path" 2>/dev/null || printf '0')"
  (( size > 90000000 )) || return 1
  if [[ -n "$EXPECTED_SHA256" ]]; then
    command -v sha256sum >/dev/null 2>&1 || fail "sha256sum_missing"
    local actual_sha256
    actual_sha256="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
      printf 'V11_STEP4_REID_PREPARE ONNX_VERIFY=FAIL path=%s expected_sha256=%s actual_sha256=%s\n' \
        "$path" "$EXPECTED_SHA256" "$actual_sha256" >&2
      return 1
    fi
  fi
  return 0
}

[[ -x "$PYTHON" ]] || fail "trt86_python_missing path=$PYTHON"

if [[ -s "$ONNX" ]] && ! verify_onnx "$ONNX"; then
  printf 'V11_STEP4_REID_PREPARE ONNX_INVALID action=redownload path=%s\n' "$ONNX" >&2
  rm -f "$ONNX"
fi

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
  verify_onnx "$tmp" || fail "onnx_download_verification_failed"
  size="$(stat -c '%s' "$tmp" 2>/dev/null || printf '0')"
  mv -f "$tmp" "$ONNX"
  printf 'V11_STEP4_REID_PREPARE DOWNLOAD_RESULT=PASS path=%s bytes=%s sha256=%s\n' \
    "$ONNX" "$size" "${EXPECTED_SHA256:-unchecked}"
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
