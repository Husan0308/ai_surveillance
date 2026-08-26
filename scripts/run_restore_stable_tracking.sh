#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

# Restored pre-Global-ID local tracking checkpoint.
# NvDCF owns local IDs and bbox propagation every frame; the native display
# smoother only adjusts presentation and never creates ghost objects.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_DETECT_WIDTH="${CAMERA_V2_DETECT_WIDTH:-736}"
export CAMERA_V2_DETECT_HEIGHT="${CAMERA_V2_DETECT_HEIGHT:-416}"
export CAMERA_V2_MICRO_BATCH="${CAMERA_V2_MICRO_BATCH:-2}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.05}"
export CAMERA_V2_DETECT_IOU="${CAMERA_V2_DETECT_IOU:-0.65}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"
export CAMERA_V2_TRACKER_WIDTH="${CAMERA_V2_TRACKER_WIDTH:-512}"
export CAMERA_V2_TRACKER_HEIGHT="${CAMERA_V2_TRACKER_HEIGHT:-288}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.12}"
export CAMERA_V2_DEDUP_IOU="${CAMERA_V2_DEDUP_IOU:-0.82}"
export CAMERA_V2_DEDUP_CONTAINMENT="${CAMERA_V2_DEDUP_CONTAINMENT:-0.94}"
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-3.0}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-2.2}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-3.6}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-280}"

PY="${CAMERA_V2_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "RESTORE_TRACKING ERROR missing python: $PY" >&2; exit 1; }

printf '%s\n' \
  "RESTORE_TRACKING_PROFILE source=6xRTSP local_tracker=NvDCF detector=YOLO26m/736x416/b2" \
  "RESTORE_TRACKING_POLICY bbox_owner=NvDCF smoother=native-display-only global_id=off" \
  "RESTORE_TRACKING_TARGET detector=${CAMERA_V2_DETECT_TARGET_HZ}Hz/cam tracker=${CAMERA_V2_TRACKER_WIDTH}x${CAMERA_V2_TRACKER_HEIGHT}"

exec "$PY" -u -m services.camera_v2.person_tracking_final
