#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PARSER_SRC="$ROOT/services/camera_v11/deepstream_yolo26_parser_v1.cpp"
META_SRC="$ROOT/services/camera_v11/deepstream_yolo_meta_v1.c"
PARSER_OUT="${V11_DS_YOLO_PARSER_LIB:-$ROOT/artifacts/yolo26s_ds71/libnvdsinfer_yolo26_v1.so}"
META_OUT="${V11_DS_YOLO_META_LIB:-$ROOT/artifacts/yolo26s_ds71/libcamera_v11_ds_yolo_meta_v1.so}"

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
  echo "V11_DS_YOLO_BUILD RESULT=FAIL reason=deepstream_headers_missing" >&2
  exit 1
}

for tool in g++ gcc pkg-config; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "V11_DS_YOLO_BUILD RESULT=FAIL reason=${tool}_missing" >&2
    exit 1
  }
done
for src in "$PARSER_SRC" "$META_SRC"; do
  [[ -f "$src" ]] || {
    echo "V11_DS_YOLO_BUILD RESULT=FAIL reason=source_missing path=$src" >&2
    exit 1
  }
done

mkdir -p "$(dirname "$PARSER_OUT")"

g++ -std=c++17 -O2 -fPIC -shared \
  -I"$DS_ROOT/sources/includes" \
  "$PARSER_SRC" \
  -o "$PARSER_OUT"

read -r -a PKG_FLAGS <<<"$(pkg-config --cflags --libs gstreamer-1.0 glib-2.0)"
gcc -std=c11 -O2 -fPIC -shared \
  -I"$DS_ROOT/sources/includes" \
  -L"$DS_ROOT/lib" \
  -Wl,-rpath,"$DS_ROOT/lib" \
  "$META_SRC" \
  -o "$META_OUT" \
  "${PKG_FLAGS[@]}" \
  -lnvds_meta -lnvdsgst_meta

[[ -s "$PARSER_OUT" && -s "$META_OUT" ]] || {
  echo "V11_DS_YOLO_BUILD RESULT=FAIL reason=library_missing" >&2
  exit 1
}

echo "V11_DS_YOLO_BUILD RESULT=PASS ds_root=$DS_ROOT parser=$PARSER_OUT meta=$META_OUT"
