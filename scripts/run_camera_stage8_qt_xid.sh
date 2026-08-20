#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CAMERA="${CAMERA_V2_STAGE8_CAMERA:-CAM-01}"
HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

echo "STAGE8_BUILD branch=${BRANCH} head=${HEAD_SHA} camera=${CAMERA}"
echo "STAGE8_CONTRACT mux=1 batch=1 tiler=1 layout=1x1 convert=1 capsforce=NVMM-RGBA osd=1 sink=nveglglessink qt=1 xid=1 cameras=1 detector=0 tracker=0 rtsp=tcp latency=250ms"

if [[ -n "${DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
export CAMERA_V2_STAGE8_CAMERA="${CAMERA}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-250}"

exec python -m services.camera_v2.stage8_qt_xid
