# Current Canonical Baseline — AI Surveillance

This file is the recovery/source-of-truth note for the current project state. Before changing architecture, camera transport, detector, tracker, UI, or service ownership, read this file together with `docs/ARCHITECTURE_CONTRACT.md`.

Update this document whenever a new stage is hardware-validated and becomes the new baseline.

## Canonical branch and merge rule

- Repository: `Husan0308/ai_surveillance`
- Base branch: `rebuild/gpu-v2-clean`
- Active stabilization branch: `agent/three-service-stabilization`
- Draft PR: #30
- Do **not** merge PR #30 until explicit approval after hardware validation.

## Non-negotiable production architecture

Exactly three production services:

```text
RTSP cameras
    |
    v
ml_service :8001
DeepStream/NVDEC -> YOLO26m -> per-camera ByteTrack -> presentation/video
    |
    v
api_service :8000
REST + application state + ML proxy
    |
    v
frontend
PySide6 operator UI
```

Rules:

- frontend must not start DeepStream, YOLO, ByteTrack, ReID, or Face runtimes.
- api_service must not own RTSP decode or model inference.
- ml_service owns camera ingest and ML runtime.
- frontend reads metadata/state through API.
- local live video uses the current mmap path from `ml_service` to frontend; MJPEG remains fallback/diagnostic.
- `services/camera_v2` is not a fourth production service.

## Explicit stage sequencing

Current sequence:

1. camera ingest — validated
2. person detection — validated
3. local per-camera tracking — validated
4. UI/presentation stabilization — current work
5. heatmap — later
6. ReID — **not enabled yet**
7. Face recognition — **not enabled yet**

Do not add ReID or Face unless explicitly requested.

## Cameras

NVR host:

`192.168.1.210:554/Streaming/Channels/`

Canonical camera mapping:

- CAM-01 -> `/101`
- CAM-02 -> `/201`
- CAM-03 -> `/301`
- CAM-04 -> `/601`
- CAM-05 -> `/501`
- CAM-06 -> `/401`

Per-camera latency baseline:

- CAM-01: 20 ms
- CAM-02: 20 ms
- CAM-03: 20 ms
- CAM-04: 80 ms
- CAM-05: 80 ms
- CAM-06: 80 ms

Known harmless DeepStream 7.1 warning if decoding continues successfully:

`Failed to query video capabilities: Invalid argument`

## Camera ingest baseline

Current ingest is proven on all six cameras:

- `nvurisrcbin`
- NVDEC / DeepStream
- latest-only downstream behavior
- `drop_on_latency=true`
- post-decode queue = 1
- reconnect enabled
- all six feeds hardware-smoked successfully

Do not replace this ingest path without a measured reason.

## Local video transport baseline

Current local desktop video path is the proven old mmap approach restored into the canonical three-service architecture:

```text
ml_service latest presentation frame
    -> SIGBUS-safe latest-only mmap
    -> SmoothMmapFrameReader
    -> PySide6 camera wall
```

Key properties:

- display frame: 960x540
- display target: 20 FPS
- latest-only transport
- no HTTP/JPEG encode-decode in normal camera-wall hot path
- MJPEG retained only as fallback/diagnostic
- mmap smoke passed 6/6 cameras at 960x540
- frontend only repaints when a new sequence arrives
- focused/fullscreen mode pauses hidden camera UI readers

Validated mmap smoke example:

- CAM-01 PASS 960x540
- CAM-02 PASS 960x540
- CAM-03 PASS 960x540
- CAM-04 PASS 960x540
- CAM-05 PASS 960x540
- CAM-06 PASS 960x540
- `MMAP_VIDEO_SMOKE=PASS`

## Detector baseline

Detector remains isolated in its own spawned CUDA process inside `ml_service`.

Current detector profile:

- model: `yolo26m.pt`
- device: `cuda:0`
- PyTorch CUDA worker isolation: `spawn-process`
- batch size: 2
- input: 736x416
- target: 4 FPS per camera
- person class only
- confidence: 0.08
- IoU: 0.70
- max detections: 50
- FP32; do not pass deprecated `half` runtime argument

Hardware compatibility baseline:

- GPU: NVIDIA GeForce GTX 1050 Ti 4GB
- capability: `sm_61`
- installed PyTorch build includes compatible `sm_60` cubin
- actual CUDA kernel probe passed

Do not reinstall/replace Torch just because `sm_61` is not literally listed by `torch.cuda.get_arch_list()`.

## Detector validation

Previously validated hardware smoke:

- all six cameras online
- detector state `ready`
- average batch roughly mid-40 ms range
- person detection smoke passed
- `PERSON_DETECT_SMOKE=PASS`

A key tracking fix was lowering detector confidence so ByteTrack receives low-confidence continuation candidates rather than dropping them before association.

## Tracking baseline

Tracking is **local per camera only**.

- backend: ByteTrack
- scope: per-camera
- no global ID
- no cross-camera identity
- no ReID
- no Face

Current tracker profile:

- `track_high_thresh = 0.25`
- `track_low_thresh = 0.08`
- `new_track_thresh = 0.25`
- `track_buffer_seconds = 2.5`
- `match_thresh = 0.80`
- `fuse_score = true`

ByteTrack remains the only identity/association owner for local `T1/T2/...` IDs.

Presentation smoothing may predict/interpolate boxes visually, but it must not create, merge, or reassign tracker IDs.

## Tracking validation

Hardware tracking smoke passed:

- `PERSON_TRACK_SMOKE=PASS`

Important CAM-03 stability result after low-confidence detector fix:

```text
samples=60
peak=3
distinct_seen=4
created_delta=2
id_set_changes=2
occupancy_changes=1
same_count_id_changes=1
zero_samples=0
PERSON_TRACK_STABILITY=PASS
```

This replaced a much worse baseline and is the current accepted local-tracking state.

## BBox / presentation rules

Current desired behavior:

- ByteTrack local T-ID remains authoritative.
- presentation smoother must reduce visible lag/jitter without changing identity ownership.
- short bounded prediction is allowed.
- large disagreement with new measurement should snap/correct rather than trail far behind.
- draw-time bbox padding may enlarge the visible rectangle to better contain visible head/arms/legs.
- presentation layer must never deduplicate two different ByteTrack IDs into one box.
- crowded/overlapping detections should preserve multiple candidates as much as the detector allows.

Do not claim invisible/fully occluded body parts can always be recovered from a camera image.

## API baseline

`api_service :8000` is validated against `ml_service :8001`.

Canonical endpoints include:

- `/health`
- `/api/v1/ml/health`
- `/api/v1/cameras`
- camera detections proxy
- camera tracks proxy

Validated integration result:

`API_SMOKE=PASS`

Unknown camera remains 404 through API; ML transport/service failure maps to 503.

## Frontend baseline

Canonical frontend entrypoint:

`services.frontend.app.main`

Current UI is PySide6 and must remain presentation-only.

Monitoring requirements:

- dark Apsidal operator shell
- 6 cameras
- fixed 2-column x 3-row grid
- no page scrolling for monitoring wall
- fullscreen/focus per camera
- right-side identity/summary rail around 1/4 width
- Known / Unknown summary
- Recent Views
- camera labels are CAM-01 ... CAM-06
- avoid room labels and unnecessary technical clutter on camera cards
- camera transport / detection / tracking must not be modified during UI-only requests

Latest UI-only baseline uses a darker operator design inspired by the supplied Sentinel VMS reference while preserving the real camera wall and backend integration.

## UI-only guard

When the request is specifically UI work:

- do not change `config/cameras.yaml`
- do not change detector thresholds/resolution
- do not change ByteTrack settings
- do not change mmap transport
- do not change DeepStream ingest
- do not change API contracts unless UI cannot function without it

Prefer changes in frontend presentation files only.

## Current validated commands

ML:

```bash
bash scripts/run_ml_service.sh
```

API:

```bash
bash scripts/run_api_service.sh
```

Frontend:

```bash
bash scripts/run_frontend.sh
```

Useful smoke tests:

```bash
python scripts/smoke_ml_service.py
python scripts/smoke_person_detection.py
python scripts/smoke_person_tracking.py
python scripts/smoke_person_tracking_stability.py
python scripts/smoke_api_service.py
python scripts/smoke_mmap_video.py
```

## Known good acceptance markers

- 6/6 camera ingest
- `ML_SMOKE=PASS`
- `PERSON_DETECT_SMOKE=PASS`
- `PERSON_TRACK_SMOKE=PASS`
- `PERSON_TRACK_STABILITY=PASS`
- `API_SMOKE=PASS`
- `MMAP_VIDEO_SMOKE=PASS`
- Production Static CI success

## Things not to do by accident

- Do not reintroduce a monolith.
- Do not make frontend start ML.
- Do not make API own DeepStream/YOLO.
- Do not add ReID or Face yet.
- Do not use `model.track()` as a replacement for the current detector + one tracker per camera design.
- Do not replace latest-only video with buffering queues.
- Do not blindly reinstall Torch.
- Do not merge draft PR #30 without explicit approval.
- Do not rewrite the camera pipeline from scratch when an old proven implementation already exists in history.

## Recovery instruction for future work

If context is lost or the project is resumed later:

1. read `docs/ARCHITECTURE_CONTRACT.md`
2. read this file
3. inspect the current branch head
4. preserve all validated stages unless the user explicitly asks to change them
5. make one narrowly-scoped change at a time and hardware-smoke it before moving to the next stage
