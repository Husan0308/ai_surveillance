#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DS_ROOT="${DEEPSTREAM_SOURCE_ROOT:-$HOME/nvidia/DeepStream}"
IMAGE="${MV3DT_BUILD_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
OUT_DIR="${MV3DT_DIAGNOSTIC_BUILD_DIR:-$ROOT/.runtime/mv3dt/diagnostic-bin/cam04-latency}"
OUT_BIN="$OUT_DIR/deepstream-test5-pn263-global-id-proto"

fail() {
  echo "CAM04_DIAGNOSTIC_BUILD status=FAIL reason=$*" >&2
  exit 1
}

[[ -d "$DS_ROOT" ]] || fail "DeepStream source root not found: $DS_ROOT"

UTC_FILE="$(find "$DS_ROOT" -type f -name deepstream_utc.c -print -quit 2>/dev/null || true)"
[[ -n "$UTC_FILE" ]] || fail "deepstream_utc.c not found under $DS_ROOT"
OLD_SOURCE_DIR="$(dirname "$UTC_FILE")"

APP_FILE="$(find "$DS_ROOT" -type f -path '*/apps/sample_apps/deepstream-app/deepstream_app.c' -print -quit 2>/dev/null || true)"
[[ -n "$APP_FILE" ]] || fail "deepstream_app.c not found under $DS_ROOT"
DS_SRC="$(dirname "$(dirname "$(dirname "$(dirname "$APP_FILE")")")")"

if [[ -d "$DS_ROOT/includes" ]]; then
  DS_INCLUDES="$DS_ROOT/includes"
elif [[ -d "$DS_SRC/includes" ]]; then
  DS_INCLUDES="$DS_SRC/includes"
else
  fail "DeepStream includes directory not found"
fi

[[ -f "$ROOT/services/mv3dt_room/native/deepstream_test5_app_main.c" ]] || fail "native source missing"
[[ -f "$ROOT/scripts/dev_room_mv3dt/build_native_binary.sh" ]] || fail "inner build script missing"

mkdir -p "$OUT_DIR"
rm -f "$OUT_BIN"

echo "CAM04_DIAGNOSTIC_BUILD ds_root=$DS_ROOT"
echo "CAM04_DIAGNOSTIC_BUILD old_source=$OLD_SOURCE_DIR"
echo "CAM04_DIAGNOSTIC_BUILD ds_src=$DS_SRC"
echo "CAM04_DIAGNOSTIC_BUILD includes=$DS_INCLUDES"
echo "CAM04_DIAGNOSTIC_BUILD output=$OUT_BIN"

docker run --rm --pull=never   -v "$ROOT/services/mv3dt_room/native:/workspace/room-native:ro"   -v "$OLD_SOURCE_DIR:/workspace/old-source:ro"   -v "$DS_SRC:/workspace/ds-src:ro"   -v "$DS_INCLUDES:/workspace/ds-includes:ro"   -v "$ROOT/scripts/dev_room_mv3dt/build_native_binary.sh:/workspace/build-native.sh:ro"   -v "$OUT_DIR:/workspace/bin"   --entrypoint bash   "$IMAGE" /workspace/build-native.sh

[[ -x "$OUT_BIN" ]] || fail "build completed without executable output"

HASH="$(sha256sum "$OUT_BIN" | awk '{print $1}')"
cat >"$OUT_DIR/build.json" <<EOF
{
  "binary": "$OUT_BIN",
  "sha256": "$HASH",
  "source_branch": "$(git -C "$ROOT" branch --show-current)",
  "source_commit": "$(git -C "$ROOT" rev-parse HEAD)"
}
EOF

echo "CAM04_DIAGNOSTIC_BUILD status=PASS"
echo "MV3DT_BINARY=$OUT_BIN"
echo "MV3DT_DIAGNOSTIC_BINARY_SHA256=$HASH"
