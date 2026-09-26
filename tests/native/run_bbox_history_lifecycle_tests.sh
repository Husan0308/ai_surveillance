#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$(mktemp -d /tmp/pn263-bbox-history-tests.XXXXXX)"
trap 'rm -rf -- "$BUILD_DIR"' EXIT
gcc -std=c11 -Wall -Wextra -Werror -pedantic \
  -I"$ROOT/services/mv3dt_room/native" \
  "$ROOT/services/mv3dt_room/native/bbox_history_lifecycle.c" \
  "$ROOT/tests/native/test_bbox_history_lifecycle.c" \
  -o "$BUILD_DIR/test_bbox_history_lifecycle"
"$BUILD_DIR/test_bbox_history_lifecycle"
