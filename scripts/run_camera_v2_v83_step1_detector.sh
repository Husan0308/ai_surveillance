#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Step 1 only: restore the exact known-good V8 camera/tracker runtime. Do not layer
# V8.1/V8.2 bbox experiments on top until detector latency is back under control.
export CAMERA_V8_TRT86_BATCH_WORKER="$ROOT/scripts/yolo26_trt86_batch6_worker_v83.py"
export CAMERA_V83_TRT_WARMUP_ITERS="${CAMERA_V83_TRT_WARMUP_ITERS:-12}"

# Start hot enough that a single cold first sample cannot immediately trap the
# adaptive controller at 0.70 Hz. If warmup did its job, integrated GPU latency
# should remain close to the original V8 ~90-150 ms range and adaptation can then
# safely decide whether 2 Hz fits the measured budget.
export CAMERA_V8_DETECT_INITIAL_HZ="${CAMERA_V8_DETECT_INITIAL_HZ:-2.00}"
export CAMERA_V8_DETECT_MIN_HZ="${CAMERA_V8_DETECT_MIN_HZ:-0.70}"
export CAMERA_V8_DETECT_MAX_HZ="${CAMERA_V8_DETECT_MAX_HZ:-2.00}"
export CAMERA_V8_DETECT_GPU_BUDGET="${CAMERA_V8_DETECT_GPU_BUDGET:-0.28}"
export CAMERA_V8_DETECT_EMA_ALPHA="${CAMERA_V8_DETECT_EMA_ALPHA:-0.20}"

# Restore original V8 tracker/display values exactly for the A/B test.
export CAMERA_V2_TRACK_WIDTH="${CAMERA_V2_TRACK_WIDTH:-512}"
export CAMERA_V2_TRACK_HEIGHT="${CAMERA_V2_TRACK_HEIGHT:-288}"
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-8}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.28}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-420}"
export CAMERA_V2_DISPLAY_EMPTY_HOLD_MS="${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS:-350}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-60}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"

printf '%s\n' \
  "CAMERA_V83_STEP1 scope=detector-latency-only base=exact-V8 tracker=8Hz bbox_experiments=0" \
  "CAMERA_V83_STEP1_POLICY worker_warmup=${CAMERA_V83_TRT_WARMUP_ITERS} initial_detector=${CAMERA_V8_DETECT_INITIAL_HZ}Hz adaptive=${CAMERA_V8_DETECT_MIN_HZ}-${CAMERA_V8_DETECT_MAX_HZ}Hz"

exec bash "$ROOT/scripts/run_camera_v2_bbox_v8.sh"
