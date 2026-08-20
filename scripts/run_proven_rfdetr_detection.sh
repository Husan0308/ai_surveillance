#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONUNBUFFERED=1
export RF_HOME="${RF_HOME:-$PWD/.runtime/rfdetr}"
mkdir -p "$RF_HOME"

# Keep the old rebuild/gpu-v2-clean camera + tracking geometry.
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_DETECT_WIDTH=736
export CAMERA_V2_DETECT_HEIGHT=416
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288

# RF-DETR-specific detector settings. The downstream detection/tracking logic is
# unchanged from rebuild/gpu-v2-clean. micro_batch=1 is the only memory-safety
# adjustment for the 4GB deployment GPU.
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.12}"
export CAMERA_V2_DETECT_IOU="${CAMERA_V2_DETECT_IOU:-0.65}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"
export CAMERA_V2_MICRO_BATCH="${CAMERA_V2_MICRO_BATCH:-1}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-280}"

# Exact old local NvDCF / display policy.
export CAMERA_V2_TRACK_BOX_SIDE_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_TOP_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN=0.00
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.28}"
export CAMERA_V2_DEDUP_IOU=0.82
export CAMERA_V2_DEDUP_CONTAINMENT=0.94
export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN=0.04
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN=0.10

# Old scheduler: target ~3Hz/camera, with adaptive wall-protection logic inside
# CameraPersonTrackingFinal. RF-DETR may settle lower on GTX 1050 Ti; NvDCF owns
# all intermediate frames.
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-3.0}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-2.2}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-3.6}"

BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
echo "PROVEN_RFDETR_BUILD branch=${BRANCH} head=${HEAD_SHA} source=rebuild/gpu-v2-clean"
echo "PROVEN_RFDETR_PROFILE mux=2560x1440 detector=RF-DETR-S@736x416 threshold=${CAMERA_V2_DETECT_CONF} micro=${CAMERA_V2_MICRO_BATCH} tracker=NvDCF@512x288 old_dedup=1 old_latency=1 custom_smoother=0 display_padding=8/4/10"

exec python -m services.camera_v2.rfdetr_proven_oldstack
