#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
CAMERA_ID="${CAMERA_V2_STAGE3_CAMERA:-CAM-01}"

echo "STAGE3_BUILD branch=${BRANCH} head=${HEAD_SHA} camera=${CAMERA_ID}"
echo "STAGE3_CONTRACT mux=1 batch=1 tiler=1 layout=1x1 osd=0 detector=0 tracker=0 display=0 qt=0 rtsp=tcp latency=250ms"

export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_STAGE3_CAMERA="${CAMERA_ID}"
export CAMERA_V2_STAGE3_WIDTH="${CAMERA_V2_STAGE3_WIDTH:-2560}"
export CAMERA_V2_STAGE3_HEIGHT="${CAMERA_V2_STAGE3_HEIGHT:-1440}"

exec python -m services.camera_v2.stage3_single_tiler
