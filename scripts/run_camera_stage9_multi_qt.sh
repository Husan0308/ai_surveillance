#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COUNT="${CAMERA_V2_STAGE9_COUNT:-2}"
HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

echo "STAGE9_BUILD branch=${BRANCH} head=${HEAD_SHA} count=${COUNT}"
echo "STAGE9_CONTRACT cameras=${COUNT} mux=1 batch=${COUNT} tiler=1 convert=1 capsforce=NVMM-RGBA osd=1 sink=nveglglessink qt=1 xid=1 detector=0 tracker=0 controller=0 rtsp=tcp latency=250ms"

export CAMERA_V2_STAGE9_COUNT="${COUNT}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-250}"

exec python -m services.camera_v2.stage9_multi_qt
