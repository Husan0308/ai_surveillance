#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CAMERA="${CAMERA_V2_STAGE2_CAMERA:-CAM-01}"
HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250

echo "STAGE2_BUILD branch=${BRANCH} head=${HEAD_SHA} camera=${CAMERA}"
echo "STAGE2_CONTRACT mux=1 batch=1 tiler=0 osd=0 detector=0 tracker=0 display=0 qt=0 rtsp=tcp latency=250ms"

exec python -m services.camera_v2.stage2_single_mux
