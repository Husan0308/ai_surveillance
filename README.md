# AI Surveillance — Camera V2

Canonical stabilization branch: `cleanup/camera-v2-audited-20260825`.

The current milestone is intentionally narrow: prove the six-camera DeepStream wall
and one-camera YOLO26 TensorRT -> NvDCF path before enabling cross-camera ReID,
identity, API/UI integration or other side models.

## Canonical runtime

Run only:

```bash
bash scripts/run_cam01_trt86_audited.sh
```

Current graph:

```text
6 x RTSP
  -> nvurisrcbin / NVDEC (NVMM)
  -> per-camera tee
       -> display queue(1, leaky) -> nvstreammux(batch=6, live)
       -> CAM-01 sparse JIT detector branch
            -> queue(1, leaky)
            -> nvvideoconvert / fixed 672x384 letterbox surface
            -> appsink(max-buffers=1, drop, async=0)
            -> shared memory
            -> TensorRT 8.6.1 YOLO26
            -> fresh detector result
  -> detector NvDsObjectMeta injected at nvstreammux.src
  -> nvtracker / NvDCF (512x288, per-frame)
  -> nvmultistreamtiler
  -> nvvideoconvert -> RGBA NVMM
  -> nvdsosd
  -> nveglglessink
```

This order follows the DeepStream detector -> tracker -> tiler -> OSD model. The
runtime verifies the critical static links before PLAYING and aborts instead of
silently running a malformed graph.

## Canonical files

- `scripts/run_cam01_trt86_audited.sh` — only CAM-01 TRT86/NvDCF launcher.
- `scripts/preflight_cam01_audited_static.py` — static dependency/cleanup check.
- `scripts/yolo26_trt86_shm_worker.py` — TensorRT 8.6 CUDA runner/base.
- `scripts/yolo26_trt86_shm_worker_v2.py` — class/input diagnostics.
- `scripts/yolo26_trt86_shm_worker_v3.py` — audited fixed-shape letterbox worker.
- `services/camera_v2/person_tracking_trt86_audited.py` — audited entrypoint.
- `services/camera_v2/person_tracking_trt86_fresh.py` — JIT latest-frame scheduler.
- `services/camera_v2/person_tracking_final.py` — freshness + detector-to-NvDCF bridge.
- `services/camera_v2/person_tracking.py` — DeepStream NvDCF insertion.
- `services/camera_v2/detection.py` — source tee/appsink mailbox and detector worker contract.
- `services/camera_v2/detector_latency.py` — bounded detector-latency compensation.
- `services/camera_v2/tracker_profile.py` — generated sparse NvDCF profile.
- `services/camera_v2/yolo_trt86_shm_bridge.py` / `yolo_trt86_fresh_bridge.py` — SHM process bridge.
- `services/camera_v2/main.py` / `dynamic_wall.py` / `secure.py` — RTSP/NVDEC/mux/tiler/EGL core.
- `services/camera_v2/native_bridge.py` + native C sources — NvDs metadata bridge.
- `services/ml_service/app/config.py` + `config/cameras.yaml` — camera configuration.
- `requirements-trt86.txt` — isolated TensorRT 8.6 environment.

## Preserved for later milestones

ReID/global-identity, API service, frontend and Sentinel UI files are deliberately
kept. They are not enabled by the canonical CAM-01 launcher, but they are future
project work rather than disposable experiments.

Historical `stage1..stage22`, RF-DETR experiment scripts/backends and superseded
CAM-01 TRT launchers were removed from the cleanup branch. The older
`fix/cam01-trt86-e2e-20260825` branch remains a rollback reference.

## Static preflight

Before touching the cameras:

```bash
python3 scripts/preflight_cam01_audited_static.py
bash -n scripts/run_cam01_trt86_audited.sh
```

Expected:

```text
CAMERA_V2_AUDITED_STATIC=PASS ...
```

## Runtime proof

Start:

```bash
bash scripts/run_cam01_trt86_audited.sh 2>&1 | tee /tmp/CAM01_AUDITED.log
```

Required startup markers:

```text
CAM01_TRT86_PREFLIGHT ... tensorrt=8.6.1
CAM01_TRT86_SOURCE_HARDENED ...
CAM01_TRT86_LETTERBOX ...
CAMERA_PIPELINE_AUDIT status=OK ...
CAMERA_TRACK_FINAL ready: ...
```

Healthy scheduling should show:

```text
calls > 0
inputs > 0
timeouts = 0
stale_results = 0
result_age < max_result_age
```

For the final detector-to-tracker proof, place a clearly visible person in CAM-01.
A completed end-to-end path must then reach:

```text
boxes > 0
meta_boxes > 0
detector_injected > 0
tracked_now > 0
```

Do not enable ReID or other models until that contract is proven.
