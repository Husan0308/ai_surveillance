# Camera V11 Step2 production FP32

## Root cause

Rejected V11 Step2 branches replaced every frozen
`source -> queue(1) -> display convert -> EGL sink` graph with a tee and a
system-memory appsink branch in the same pipeline/process. Their appsink callback
mapped pixels, allocated and copied a 672x384 frame, and shared Python locks and GPU
conversion with display streaming threads. Standalone FP32 TensorRT is about 14 ms;
the regression was same-process extraction/callback/synchronization coupling, not
raw inference compute.

The mandatory five-stage experiment confirms this: display-only, separate
extraction, separate preprocessing, synthetic FP32 TRT, and complete separate
detector all pass the authoritative frozen Step1 checker.

## Architecture

Frozen display remains byte-for-byte Step1 V7:

`6 x (main RTSP -> NVDEC -> leaky queue(1) -> display convert -> EGL sink)`

Detector is a separate process and connection set:

`6 x (low-res RTSP -> NVDEC -> demand gate -> queue(1) -> resize/BGRx -> appsink(1))`

Streaming probes only update timestamps/counters. A round-robin consumer pulls
ready samples. GStreamer owns at most one pending frame per camera and Python owns
none. Four bounded wall-clock *deadline credits* contain no images; they let bursty
CAM-02 recover 2 Hz while appsink overwrites older pending samples.

The TensorRT 8.6 child preallocates its execution context, pinned host buffers,
device buffers, outputs, and a dedicated non-blocking lowest-priority CUDA stream.
CUDA events time H2D/inference/D2H. It synchronizes only that stream once when the
result is needed and never calls `cudaDeviceSynchronize` in production.

FP16 passed quality but was only 0.7% faster at p50 and was rejected. INT8 failed
quality and remains diagnostic. QAT was not started because FP32 meets budget and
no new untouched local COCO person test split remains.

## Commands

```bash
V11_STEP2_STAGE_DURATION_SEC=20 \
  scripts/run_camera_v11_step2_staged_regression_v18.sh

scripts/run_camera_v11_step2_production_fp32_v18.sh

.venv/bin/python scripts/check_camera_v11_step2_production_log_v15.py \
  --display-log /tmp/CAMERA_V11_STEP2_DISPLAY.log \
  --detector-log /tmp/CAMERA_V11_STEP2_DETECTOR.log
```
