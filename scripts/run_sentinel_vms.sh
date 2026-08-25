#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.20-r19-analysis-tiler"

if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
echo "SENTINEL_DISPLAY session=${XDG_SESSION_TYPE:-unknown} qt_platform=${QT_QPA_PLATFORM:-auto} display=${DISPLAY:-unset}"

INHERITED_DETECT="${CAMERA_V2_DETECT_WIDTH:-unset}x${CAMERA_V2_DETECT_HEIGHT:-unset}"
INHERITED_MUX="${CAMERA_V2_FRAME_WIDTH:-unset}x${CAMERA_V2_FRAME_HEIGHT:-unset}"

# Keep RTSP deterministic on the target NVR. The r18 hardware log proved TCP
# negotiation itself succeeds; later rtspsrc errors were downstream backpressure
# propagated upstream, not a transport failure.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_MUX_TIMEOUT_US=50000

export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_WALL_WIDTH=2880
export CAMERA_V2_WALL_HEIGHT=1080
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_ANALYSIS_TILE_WIDTH=1280
export CAMERA_V2_ANALYSIS_TILE_HEIGHT=720
export CAMERA_V2_DETECT_CONF=0.32
export CAMERA_V2_DETECT_IOU=0.65
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01

export CAMERA_V2_DETECT_GPU_DUTY=0.30
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.30
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.30

# GTX 1050 Ti production path. Camera ingest remains direct-to-mux. RF-DETR uses
# a gated secondary analysis tiler; nvstreamdemux is deliberately absent because
# its zero-copy children retained mux batches and exhausted the 8-buffer pool.
export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0

# Start GPU-native EGL; use X11 only when buffers reach the sink but EGL proves
# zero-render for the bounded watchdog interval.
export CAMERA_V2_DISPLAY_BACKEND=egl
export CAMERA_V2_EGL_FAILOVER_SEC=8.0

export CAMERA_V2_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_BOX_TOP_MARGIN=0.04
export CAMERA_V2_BOX_BOTTOM_MARGIN=0.10
export CAMERA_V2_BOX_MAX_AGE=2.0
export CAMERA_V2_BOX_MAX_PREDICT=0.40

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} effective_mux=2560x1440 rtsp=tcp latency=250ms detector=RF-DETR-S@672x384 threshold=0.18 batch=1 scheduler=adaptive-duty:12-30% detector_path=analysis-tiler demux=disabled tracker=motion-predictor nvtracker=disabled display=egl->x11-on-zero-render pascal_safe=1 ui=camera-only-2x3-click-fullscreen"

python scripts/preflight_rfdetr_core.py
python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
