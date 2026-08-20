#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.20-r15-camera-only"

# nveglglessink's GstVideoOverlay path needs a stable X11 window handle.
if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
echo "SENTINEL_DISPLAY session=${XDG_SESSION_TYPE:-unknown} qt_platform=${QT_QPA_PLATFORM:-auto} display=${DISPLAY:-unset}"

# Canonical core profile. Only the Qt presentation shell is camera-only; the
# DeepStream/RF-DETR/NvDCF runtime below is intentionally unchanged.
INHERITED_DETECT="${CAMERA_V2_DETECT_WIDTH:-unset}x${CAMERA_V2_DETECT_HEIGHT:-unset}"
INHERITED_TRACKER="${CAMERA_V2_TRACKER_WIDTH:-unset}x${CAMERA_V2_TRACKER_HEIGHT:-unset}"
INHERITED_MUX="${CAMERA_V2_FRAME_WIDTH:-unset}x${CAMERA_V2_FRAME_HEIGHT:-unset}"

export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_DETECT_CONF=0.18
export CAMERA_V2_DETECT_IOU=0.65
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS=220

export CAMERA_V2_TRACK_BOX_SIDE_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_TOP_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN=0.00
export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN=0.04
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN=0.10

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} inherited_tracker=${INHERITED_TRACKER} effective_mux=2560x1440 detector=RF-DETR-S@672x384 threshold=0.18 batch=1 tracker=NvDCF@512x288 ui=camera-only-2x3"

python scripts/preflight_rfdetr_core.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
