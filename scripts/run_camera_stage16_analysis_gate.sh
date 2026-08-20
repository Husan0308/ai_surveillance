#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

echo "STAGE16_BUILD branch=${BRANCH} head=${HEAD_SHA} cameras=6"
echo "STAGE16_CONTRACT cameras=6 mux=1 batch=6 tee=1 display=proven analysis_queue=1 gate=1 gate_mode=drop-all requests=0 analysis_tiler=2x3 analysis_wall=1344x1152 analysis_caps=system-BGRx analysis_sink=fakesink appsink=0 map=0 copy=0 metadata=0 detector=0 tracker=0 qt_parent=1 xid_cross_process=1 controller=spawn rtsp=tcp latency=250ms"

export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-250}"

exec python -m services.camera_v2.stage16_process_analysis_gate
