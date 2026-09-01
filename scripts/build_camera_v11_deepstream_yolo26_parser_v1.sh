#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
SRC="$ROOT/services/camera_v11/deepstream_yolo26_parser_v1.cpp"
OUT="${V11_DS_YOLO_PARSER_LIB:-$ROOT/artifacts/yolo26s_ds71/libnvdsinfer_yolo26_v1.so}"

find_ds_root() {
  if [[ -n "${DEEPSTREAM_ROOT:-}" && -f "$DEEPSTREAM_ROOT/sources/includes/nvdsinfer_custom_impl.h" ]]; then
    printf '%s\n' "$DEEPSTREAM_ROOT"
    return 0
  fi
  for candidate in \
    /opt/nvidia/deepstream/deepstream \
    /opt/nvidia/deepstream/deepstream-7.1; do
    if [[ -f "$candidate/sources/includes/nvdsinfer_custom_impl.h" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

DS_ROOT="$(find_ds_root)" || {
  echo "V11_DS_YOLO_PARSER RESULT=FAIL reason=deepstream_headers_missing" >&2
  exit 1
}

command -v g++ >/dev/null 2>&1 || {
  echo "V11_DS_YOLO_PARSER RESULT=FAIL reason=gxx_missing" >&2
  exit 1
}
[[ -f "$SRC" ]] || {
  echo "V11_DS_YOLO_PARSER RESULT=FAIL reason=source_missing path=$SRC" >&2
  exit 1
}

mkdir -p "$(dirname "$OUT")"
g++ -std=c++17 -O2 -fPIC -shared \
  -I"$DS_ROOT/sources/includes" \
  "$SRC" \
  -o "$OUT"

[[ -s "$OUT" ]] || {
  echo "V11_DS_YOLO_PARSER RESULT=FAIL reason=library_missing path=$OUT" >&2
  exit 1
}

echo "V11_DS_YOLO_PARSER RESULT=PASS ds_root=$DS_ROOT output=$OUT bytes=$(stat -c %s "$OUT")"
