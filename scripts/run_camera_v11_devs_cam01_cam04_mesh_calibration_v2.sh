#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
CAPTURE="$ROOT/scripts/capture_camera_v11_pair_frames.py"
CALIB="$ROOT/scripts/calibrate_camera_v11_room_pair_mesh_v2.py"
TMP_DIR="${V11_MESH_CALIB_TMP_DIR:-/tmp/camera_v11_devs_cam01_cam04_mesh_v2}"
FRAME_A="$TMP_DIR/CAM-01.jpg"
FRAME_B="$TMP_DIR/CAM-04.jpg"

[[ -x "$PY" ]] || { echo "V11_CAM_PAIR_MESH RESULT=FAIL reason=venv_python_missing" >&2; exit 1; }
[[ -f "$CAPTURE" ]] || { echo "V11_CAM_PAIR_MESH RESULT=FAIL reason=capture_helper_missing" >&2; exit 1; }
[[ -f "$CALIB" ]] || { echo "V11_CAM_PAIR_MESH RESULT=FAIL reason=mesh_calibration_script_missing" >&2; exit 1; }

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
  --control-points "${V11_MESH_CONTROL_POINTS:-12}" \
  --validation-points "${V11_MESH_VALIDATION_POINTS:-4}" \
  --pass-median-pct "${V11_MESH_MAX_MEDIAN_PCT:-1.0}" \
  --pass-p95-pct "${V11_MESH_MAX_P95_PCT:-2.0}" \
  --output "${V11_MESH_CALIB_OUTPUT:-artifacts/calibration/devs_cam01_cam04_floor_mesh_v2.json}" \
  "$@"
