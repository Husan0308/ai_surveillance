#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export CAMERA_V2_PARITY_CAPTURE_CAMERAS="${CAMERA_V2_PARITY_CAPTURE_CAMERAS:-CAM-01,CAM-02,CAM-04,CAM-05}"
export CAMERA_V2_PARITY_SAMPLES_PER_CAMERA="${CAMERA_V2_PARITY_SAMPLES_PER_CAMERA:-2}"
export CAMERA_V2_PARITY_DIR="${CAMERA_V2_PARITY_DIR:-$ROOT/.runtime/yolo26_parity}"

mkdir -p "$CAMERA_V2_PARITY_DIR"
rm -f "$CAMERA_V2_PARITY_DIR"/CAM-*_sample*.npy \
      "$CAMERA_V2_PARITY_DIR"/CAM-*_sample*.json \
      "$CAMERA_V2_PARITY_DIR"/parity_summary.json

printf '%s\n' \
  "CAMERA_TRT_PARITY_MODE suspects=CAM-02,CAM-05 controls=CAM-01,CAM-04" \
  "CAMERA_TRT_PARITY_MODE exact_input=672x384/BGR/NPY samples=${CAMERA_V2_PARITY_SAMPLES_PER_CAMERA} dir=${CAMERA_V2_PARITY_DIR}" \
  "CAMERA_TRT_PARITY_MODE instruction=wait-for-CAMERA_TRT_PARITY_CAPTURE-complete-1-then-Ctrl-C"

exec bash scripts/run_camera_v2_detection_only_pose.sh
