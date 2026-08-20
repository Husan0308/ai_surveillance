#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CAMERA_V2_STAGE1_CAMERA="${CAMERA_V2_STAGE1_CAMERA:-CAM-01}"
export CAMERA_V2_STAGE1_SINK="${CAMERA_V2_STAGE1_SINK:-fake}"
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

echo "STAGE1_BUILD branch=${BRANCH} head=${HEAD_SHA} camera=${CAMERA_V2_STAGE1_CAMERA} sink=${CAMERA_V2_STAGE1_SINK}"
echo "STAGE1_CONTRACT mux=0 tiler=0 osd=0 detector=0 tracker=0 qt=0 rtsp=tcp latency=250ms"

exec python -m services.camera_v2.stage1_single_camera
