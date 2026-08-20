#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.20-r16-pascal-safe"

# nveglglessink's GstVideoOverlay path needs a stable X11 window handle.
if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
echo "SENTINEL_DISPLAY session=${XDG_SESSION_TYPE:-unknown} qt_platform=${QT_QPA_PLATFORM:-auto} display=${DISPLAY:-unset}"

INHERITED_DETECT="${CAMERA_V2_DETECT_WIDTH:-unset}x${CAMERA_V2_DETECT_HEIGHT:-unset}"
INHERITED_MUX="${CAMERA_V2_FRAME_WIDTH:-unset}x${CAMERA_V2_FRAME_HEIGHT:-unset}"

# Camera/mux path: keep a single GPU-native wall. RF-DETR runs on a sparse CPU-
# mapped side branch; the visible wall never passes through gst-nvtracker.
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_DETECT_CONF=0.18
export CAMERA_V2_DETECT_IOU=0.65
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_TARGET_HZ=3.0

# GTX 1050 Ti / Pascal production mode. DeepStream 7.1's validated dGPU matrix
# does not include Pascal and the hardware smoke run showed NvDCF stalling after
# the first mux batch. Never insert gst-nvtracker on this deployment.
export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0

# Bounded temporal smoothing between sparse RF-DETR observations. These values
# affect only the display/motion predictor; detector truth remains unchanged.
export CAMERA_V2_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_BOX_TOP_MARGIN=0.04
export CAMERA_V2_BOX_BOTTOM_MARGIN=0.10
export CAMERA_V2_BOX_MAX_AGE=1.6
export CAMERA_V2_BOX_MAX_PREDICT=0.55

export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN=0.04
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN=0.10

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} effective_mux=2560x1440 detector=RF-DETR-S@672x384 threshold=0.18 batch=1 target=3Hz/cam tracker=motion-predictor nvtracker=disabled pascal_safe=1 ui=camera-only-2x3"

python scripts/preflight_rfdetr_core.py
python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
