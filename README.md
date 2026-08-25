# AI Surveillance — Camera V2

Canonical stabilization branch: `cleanup/camera-v2-audited-20260825`.

This milestone is deliberately narrow: prove the six-camera DeepStream wall and
CAM-01 YOLO26 TensorRT -> NvDCF path first. Cross-camera ReID/global identity,
API and the separate `services/frontend` application stay in the repository for
later milestones but are not enabled by the current launcher.

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
            -> nvvideoconvert -> fixed 672x384 letterbox surface
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

The runtime checks the critical static links before PLAYING. A malformed graph
fails fast instead of silently bypassing NvDCF or OSD.

## Canonical files

- `scripts/run_cam01_trt86_audited.sh` — only CAM-01 TRT86/NvDCF launcher.
- `scripts/preflight_cam01_audited_static.py` — dependency/cleanup contract.
- `scripts/yolo26_trt86_shm_worker.py` — TensorRT 8.6 CUDA runner/base.
- `scripts/yolo26_trt86_shm_worker_v2.py` — inference/class diagnostics.
- `scripts/yolo26_trt86_shm_worker_v3.py` — audited letterbox worker.
- `services/camera_v2/person_tracking_trt86_audited.py` — audited entrypoint.
- `services/camera_v2/person_tracking_trt86_fresh.py` — JIT latest-frame scheduler.
- `services/camera_v2/person_tracking_final.py` — freshness and detector metadata publication.
- `services/camera_v2/person_tracking.py` — DeepStream NvDCF insertion.
- `services/camera_v2/detection.py` — source tee/appsink/mailbox contract.
- `services/camera_v2/detector_latency.py` — bounded latency compensation.
- `services/camera_v2/tracker_profile.py` — generated sparse NvDCF profile.
- `services/camera_v2/yolo_trt86_shm_bridge.py` / `yolo_trt86_fresh_bridge.py` — SHM process bridge.
- `services/camera_v2/main.py` / `dynamic_wall.py` / `secure.py` — RTSP/NVDEC/mux/tiler/EGL core.
- `services/camera_v2/native_bridge.py` + its required native C sources — NvDs metadata bridge.
- `services/ml_service/app/config.py` + `config/cameras.yaml` — camera configuration.
- `requirements-trt86.txt` — isolated TensorRT 8.6 environment.

## Deliberately preserved for later

The following are future project work, not clutter, so they remain:

- ReID/global identity core (`global_identity.py`, `person_tracking_reid.py`, `reid_*`, `qwen_reid.py`);
- `services/api_service`;
- `services/ml_service`;
- `services/frontend`;
- camera/RTSP diagnostic scripts.

The old Camera-V2 Sentinel/Qt UI stack was removed. Future UI integration should
use the separate frontend/service architecture rather than revive parallel camera
pipelines inside `services/camera_v2`.

`native_heatmap.c` is temporarily retained even though heatmap runtimes were
removed, because `native_bridge.py` currently links that source into the shared
metadata library. Removing it before splitting the native bridge would break the
current startup contract.

## Removed from this cleanup branch

- all `stage1..stage22` pipeline experiments and launchers;
- all RF-DETR experiment/backend files;
- superseded CAM-01 fixed/fresh/non-SHM TRT launchers;
- old pose/ONNX detector variants;
- old motion/temporal tracker fallback and sparse-tracker contract experiments;
- legacy Camera-V2 Sentinel/Qt UI runtime;
- heatmap Python/pose/filter runtime variants;
- stale preflights/tests tied to those removed paths.

The original `fix/cam01-trt86-e2e-20260825` branch remains an untouched rollback
reference.

## Static preflight

```bash
python3 scripts/preflight_cam01_audited_static.py
bash -n scripts/run_cam01_trt86_audited.sh
```

Expected:

```text
CAMERA_V2_AUDITED_STATIC=PASS ...
```

## Runtime proof

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

Healthy scheduling:

```text
calls > 0
inputs > 0
timeouts = 0
stale_results = 0
result_age < max_result_age
```

For the final detector-to-tracker proof, put a clearly visible person in CAM-01.
A completed end-to-end path must reach:

```text
boxes > 0
meta_boxes > 0
detector_injected > 0
tracked_now > 0
```

Do not enable ReID or other models until that contract is proven.
