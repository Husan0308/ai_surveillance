# AI Surveillance — Camera V2

## Current AI gate: YOLO26m DeepStream person detection

Six-camera transport baseline: **PASS** (checkpoint `5db3745`).
YOLO26m person detection through `nvinfer`: **BLOCKED** after the 60-second gate.
All six sources held approximately 20 FPS, FP16 inference and realtime preview
worked, but CAM-04 emitted two near-identical person boxes in a sampled frame.
No NMS or threshold workaround was added. RTSP TCP/100 ms, NVDEC/NVMM,
batch-size 6 and realtime ffplay transport are preserved.
Tracker/ReID/face/pose/heatmap: **NOT STARTED**.

Current retry path: **YOLO26m raw one-to-many `(N,84,8400)` -> custom
person-only proposal parser -> Gst-nvinfer `cluster-mode=2` DeepStream NMS
(`nms-iou-threshold=0.70`)**. The gate now fails closed if any final same-frame
person boxes survive above the configured NMS IoU threshold. The previous
NMS-free `(N,300,6)` run remains preserved as BLOCKED evidence.

See [model preparation, configuration and detector evidence](docs/YOLO26M_DEEPSTREAM91_PERSON.md).

```bash
python3 scripts/validate_yolo26m_person.py --duration 60 --out .runtime/yolo26m-person-short
python3 scripts/yolo26m_person/check_gate.py .runtime/yolo26m-person-short
```

The camera-only stage notes below describe the preserved migration checkpoints.

## Current final camera-only gate: CAM-01 through CAM-06 / DeepStream 9.1

The validated five-camera 100 ms baseline is preserved. This branch adds CAM-06
only, keeps AI disabled, uses source IDs 0..5 and `nvstreammux batch-size=6`.
The live preview is a full 3x2 grid with two cameras per row.

```bash
python3 -m unittest tests.test_cam_six_stability -v
python3 scripts/validate_cam_six.py --duration 60 --out .runtime/cam01-cam02-cam03-cam04-cam05-cam06-visual
```

See [the six-camera validation guide](docs/CAM01_CAM02_CAM03_CAM04_CAM05_CAM06_DEEPSTREAM91_VALIDATION.md).

## Current next gate: CAM-01 through CAM-05 / DeepStream 9.1

The validated four-camera 100 ms baseline is preserved. This branch adds
CAM-05 only, keeps AI disabled, uses source IDs 0..4 and
`nvstreammux batch-size=5`. The live preview is 3x2 so there are still two
cameras per row; the sixth tile is empty.

```bash
python3 -m unittest tests.test_cam_five_stability -v
python3 scripts/validate_cam_five.py --duration 60 --out .runtime/cam01-cam02-cam03-cam04-cam05-visual
```

See [the five-camera validation guide](docs/CAM01_CAM02_CAM03_CAM04_CAM05_DEEPSTREAM91_VALIDATION.md).

## Current next gate: CAM-01 + CAM-02 + CAM-03 + CAM-04 / DeepStream 9.1

The validated three-camera 100 ms baseline is preserved. This branch adds
CAM-04 only, keeps AI disabled, uses deterministic source IDs 0..3,
`nvstreammux batch-size=4`, and fills the existing 2x2 live ffplay grid.

```bash
python3 -m unittest tests.test_cam_four_stability -v
python3 scripts/validate_cam_four.py --duration 60 --out .runtime/cam01-cam02-cam03-cam04-visual
```

See [the four-camera validation guide](docs/CAM01_CAM02_CAM03_CAM04_DEEPSTREAM91_VALIDATION.md).

## Current next gate: CAM-01 + CAM-02 + CAM-03 / DeepStream 9.1

The validated CAM-01 + CAM-02 / 100 ms baseline is preserved. This branch adds
CAM-03 only, with AI still disabled. The three sources use deterministic source
IDs 0/1/2, one `nvstreammux` with `batch-size=3`, and a 2x2 tiled output.
The host ffplay live preview auto-opens as before.

```bash
python3 -m unittest tests.test_cam_three_stability -v
python3 scripts/validate_cam_three.py --duration 60 --out .runtime/cam01-cam02-cam03-visual
```

If all three panes advance normally, continue with the 660-second soak described
in [the three-camera validation guide](docs/CAM01_CAM02_CAM03_DEEPSTREAM91_VALIDATION.md).

## Current validated gate: CAM-01 + CAM-02 / DeepStream 9.1

This branch validates exactly two real RTSP sources in one DeepStream
pipeline. Both sources use `nvurisrcbin -> NVDEC -> NVMM`, then enter one
`nvstreammux` with `batch-size=2`, `live-source=true`,
`batched-push-timeout=50000` and `sync-inputs=false`. The output is a 1x2
NVENC diagnostic recording. No inference, tracker, ReID, face, pose, heatmap or
frontend code is loaded.

The validated default RTSP jitter latency is **100 ms**. A normal run now
auto-opens a host `ffplay` 1x2 live preview while recording the MKV evidence
file; pass `--no-preview` for a headless/evidence-only run.

```bash
python3 scripts/validate_cam_pair.py --duration 45 --out .runtime/cam01-cam02-visual
```

Clean soak:

```bash
python3 scripts/validate_cam_pair.py --duration 660 --out .runtime/cam01-cam02-stability
python3 scripts/cam_pair_validation/check_stability.py .runtime/cam01-cam02-stability
```

Source isolation:

```bash
python3 scripts/validate_cam_pair.py --duration 80 --interrupt-camera CAM-02 --interrupt-at 20 --interrupt-seconds 12 --out .runtime/cam02-isolation
```

The 100 ms profile passed the 660-second soak and both source-isolation
directions. See [CAM-01 + CAM-02 validation](docs/CAM01_CAM02_DEEPSTREAM91_VALIDATION.md).
CAM-03 is the next camera-only stage; AI remains deferred.

## Current camera-only validation: CAM-01 / DeepStream 9.1

The platform is validated: RTX 3060 12 GB, driver **595.91.07**, DeepStream
**9.1.0**, CUDA runtime **13.2**, TensorRT **10.16.1.11**, using the pinned NVIDIA
container in [config/deepstream-platform.json](config/deepstream-platform.json).

CAM-01 camera-only validation **PASS** (2026-09-18): RTSP/authentication, NVIDIA
H.264 decode, real recorded video, same-process reconnect and an uninterrupted
660-second run are verified. The recording contains 13,189 frames; steady-state
throughput averaged 20.002 FPS with constant 321 MiB VRAM.
CAM-02 through CAM-06 and all AI/frontend integration remain deferred.

```bash
python3 scripts/validate_cam01.py --mode record --duration 660 --out .runtime/cam01-stability
python3 scripts/cam01_validation/check_stability.py .runtime/cam01-stability
```

Only the existing CAM-01 URL is opened, with batch-size 1. Video remains in NVMM
through decode/mux and is saved using NVENC to a real diagnostic recording.
The current desktop uses software X11 rendering, so EGL display is excluded from
acceptance. See the [CAM-01 instructions and evidence](docs/CAM01_DEEPSTREAM91_VALIDATION.md)
for playback, settings, reconnect testing and acceptance criteria. The earlier
[migration audit](docs/DEEPSTREAM91_MIGRATION_AUDIT.md) is a historical snapshot.

## Historical runtime notes — not DS9.1 launch instructions

The following predates this migration and includes paths absent from this
checkout. It is retained as historical context.

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

```tex
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
