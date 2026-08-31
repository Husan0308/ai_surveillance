#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"

[[ -x "$PY" ]] || { echo "V11_CAM_PAIR_CALIB RESULT=FAIL reason=venv_python_missing" >&2; exit 1; }

exec "$PY" "$ROOT/scripts/calibrate_camera_v11_room_pair.py" \
  --camera-a CAM-01 \
  --camera-b CAM-04 \
  --points "${V11_CALIB_POINTS:-8}" \
  --ransac-px "${V11_CALIB_RANSAC_PX:-8}" \
  --pass-inlier-ratio "${V11_CALIB_MIN_INLIER_RATIO:-0.75}" \
  --pass-median-px "${V11_CALIB_MAX_MEDIAN_PX:-12}" \
  --pass-p95-px "${V11_CALIB_MAX_P95_PX:-25}" \
  --output "${V11_CALIB_OUTPUT:-artifacts/calibration/devs_cam01_cam04_floor_v1.json}" \
  "$@"
