#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.19-r9"

# Old shells/.env files can retain CAMERA_V2_* values from earlier experiments.
# Preserve the known 4MP main-stream geometry through nvstreammux. Detector and
# NvDCF still use their own lightweight resolutions; fullscreen is rendered at
# 1920x1080 by sentinel_video_pro without stretching the source.
INHERITED_DETECT="${CAMERA_V2_DETECT_WIDTH:-unset}x${CAMERA_V2_DETECT_HEIGHT:-unset}"
INHERITED_TRACKER="${CAMERA_V2_TRACKER_WIDTH:-unset}x${CAMERA_V2_TRACKER_HEIGHT:-unset}"
INHERITED_MUX="${CAMERA_V2_FRAME_WIDTH:-unset}x${CAMERA_V2_FRAME_HEIGHT:-unset}"

export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_DETECT_WIDTH=736
export CAMERA_V2_DETECT_HEIGHT=416
export CAMERA_V2_DETECT_CONF=0.05
export CAMERA_V2_DETECT_IOU=0.65
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_MICRO_BATCH=2
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288

# Bound stale async detections; NvDCF owns per-frame motion between detector hits.
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS=220

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} inherited_tracker=${INHERITED_TRACKER} effective_mux=2560x1440 focus=1920x1080 effective_detector=736x416 effective_tracker=512x288"

python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
