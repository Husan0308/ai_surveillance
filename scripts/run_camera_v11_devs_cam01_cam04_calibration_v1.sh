#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
CAPTURE="$ROOT/scripts/capture_camera_v11_pair_frames.py"
CALIB="$ROOT/scripts/calibrate_camera_v11_room_pair.py"
TMP_DIR="${V11_CALIB_TMP_DIR:-/tmp/camera_v11_devs_cam01_cam04_calibration_v1}"
FRAME_A="$TMP_DIR/CAM-01.jpg"
FRAME_B="$TMP_DIR/CAM-04.jpg"

[[ -x "$PY" ]] || { echo "V11_CAM_PAIR_CALIB RESULT=FAIL reason=venv_python_missing" >&2; exit 1; }
[[ -f "$CAPTURE" ]] || { echo "V11_CAM_PAIR_CALIB RESULT=FAIL reason=capture_helper_missing" >&2; exit 1; }
[[ -f "$CALIB" ]] || { echo "V11_CAM_PAIR_CALIB RESULT=FAIL reason=calibration_script_missing" >&2; exit 1; }

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

"$PY" "$CAPTURE" \
  --camera-a CAM-01 \
  --camera-b CAM-04 \
  --output-a "$FRAME_A" \
  --output-b "$FRAME_B" \
  --timeout-sec "${V11_CALIB_CAPTURE_TIMEOUT_SEC:-6}"

exec "$PY" "$CALIB" \
  --camera-a CAM-01 \
  --camera-b CAM-04 \
  --frame-a "$FRAME_A" \
  --frame-b "$FRAME_B" \
  --points "${V11_CALIB_POINTS:-8}" \
  --ransac-px "${V11_CALIB_RANSAC_PX:-8}" \
  --pass-inlier-ratio "${V11_CALIB_MIN_INLIER_RATIO:-0.75}" \
  --pass-median-px "${V11_CALIB_MAX_MEDIAN_PX:-12}" \
  --pass-p95-px "${V11_CALIB_MAX_P95_PX:-25}" \
  --output "${V11_CALIB_OUTPUT:-artifacts/calibration/devs_cam01_cam04_floor_v1.json}" \
  "$@"
