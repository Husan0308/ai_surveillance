# Camera V2 Clean Pascal Runtime

This branch replaces the layered Camera V2 experiment stack with one production
runtime: `services/camera_v2/runtime.py`.

## Why it was rebuilt

The measured source streams are 20 FPS, but the previous wall settled around
9-12 FPS after detector/NvDCF were enabled. Detector cadence itself was already
stable at 0.40 Hz/camera, so continuing to tune YOLO scheduling could not fix the
visible wall. The old graph put detector metadata and NvDCF in the serial path
that every displayed frame had to traverse.

The clean graph makes presentation independent from analytics.

## Production graph

```text
RTSP / NVDEC (one connection, one decode per camera)
                 |
                 v
                tee
       +---------+-----------+
       |         |           |
       v         v           v
   DISPLAY     TRACKER     DETECTOR
 latest q      latest q     latest q
       |        10 Hz gate    JIT gate
       |         |             |
       |     tracker_mux     672x378 BGRx
       |       672x384         |
       |         |          +3/+3 pad114
       |   detector metadata   |
       |         |          TRT8.6 sidecar
       |       NvDCF            |
       |         |         latest result
       |      fakesink           |
       |         +---- track cache
       |                    |
 display_mux 1280x720 <-----+
       |
    tiler 1920x720
       |
     OSD / EGL
```

The display branch never traverses TensorRT or NvDCF. Cached tracker boxes are
injected into the display batch as metadata immediately before the tiler. The
video itself therefore keeps the source/display cadence even when analytics is
slower.

## Default GTX 1050 Ti profile

- RTSP: TCP, bounded 80 ms jitterbuffer, `drop-on-latency=true`, receive-time TCP timestamps.
- Source target: 20 FPS.
- Display mux: 1280x720 per camera, cubic scaling.
- Wall: 1920x720, 3x2.
- NvDCF branch: 672x384 at 10 Hz, `config_tracker_NvDCF_max_perf.yml` derived profile.
- Detector: YOLO26s TRT8.6.1 FP32 B1, 672x384, 0.40 Hz/camera.
- Detector capture: exact 672x378 16:9 BGRx plus 3-pixel value-114 bars top/bottom.
- Queues: one-buffer, downstream-leaky/latest-only on every branch.
- ReID/Global ID: not imported by the camera hot path. Utility modules remain for phase two.

## Run

```bash
bash scripts/run_camera_v2_clean_pascal.sh 2>&1 | tee /tmp/CAMERA_CLEAN.log
```

## Acceptance criteria

After warm-up, the normal target is:

- each `CAMERA_CLEAN_STATS` source near 18-20 FPS;
- display remains fluid even though tracker is about 10 Hz;
- detector actual rate stays around 0.35-0.45 Hz per camera;
- TensorRT result age remains below 350 ms;
- display/tracker queues remain 0 or 1 with no growing backlog;
- no `Cannot keep DAR` warning;
- no continual pipeline restart;
- boxes follow people without forcing video down to tracker cadence.

Exact achievable numbers must be measured on the deployed GTX 1050 Ti/NVR; CI
can verify structure and syntax but cannot prove GPU throughput.

## Isolation test

If the full graph is still below target, disable analytics without changing the
source/display path:

```bash
CAMERA_V2_ANALYTICS_ENABLED=0 CAMERA_V2_DETECT_ENABLED=0 \
  bash scripts/run_camera_v2_clean_pascal.sh 2>&1 | tee /tmp/CAMERA_DISPLAY_ONLY.log
```

If display-only returns near 20 FPS, the remaining bottleneck is analytics GPU
work and `CAMERA_V2_TRACK_FPS` is the first controlled knob to lower. If
`display-only` is still near 10 FPS, the bottleneck is in decode/display/NVR and
must be investigated there instead of changing detector logic.

## Files intentionally removed from this branch

The old chained runtime (`detection.py`, `dynamic_wall.py`, `main.py`,
`secure.py`, `person_tracking*.py`, `pascal_runtime.py`), custom heatmap/display
smoother C modules, old CAM-01 launch/preflight scripts, old Pascal launcher, and
TRT86 diagnostic workers v2/v3 were removed. The previous branches still retain
that history and can be used for comparison; the clean branch has one production
entry point.
