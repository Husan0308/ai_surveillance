# Vision V3 Clean Architecture

`rebuild/vision-v3-clean` is a clean rebuild. Old experimental runtimes are donors only; they are not the production ownership model.

## Phase 1 — six-camera smooth core

The first acceptance target is deliberately AI-free:

```text
CAM-01..CAM-06 RTSP
  -> DeepStream nvurisrcbin / NVDEC
  -> per-source queue(max-size-buffers=1, leaky=downstream)
  -> nvstreammux(batch=6, live-source=1, sync-inputs=0)
  -> nvmultistreamtiler(3x2)
  -> wall queue(max-size-buffers=1, leaky=downstream)
  -> nveglglessink(sync=0, qos=0)
```

Rules:

- `ml_service` is the only camera owner.
- No OpenCV capture, JPEG/MJPEG, NumPy frame copy, detector, tracker, ReID, face, DB or Qt logic is allowed in this baseline.
- Old frames are dropped instead of queued. Bounded latency is more important than replaying stale frames.
- `sync-inputs=0`: one slow camera must not stall the other five.
- Source decode remains GPU-native; `nvstreammux` scales to the configured working resolution for downstream efficiency.
- Runtime prints per-camera FPS, PTS p50/p95 cadence and source queue level every few seconds.

Primary files:

```text
config/vision_v3_camera.yaml
services/ml_service/vision_v3/camera_core.py
scripts/preflight_vision_v3_camera_core.py
scripts/run_vision_v3_camera_core.sh
```

Camera topology remains six physical feeds:

- Devs: CAM-01 + CAM-04
- Entrance: CAM-02 + CAM-05
- Main Rooms: CAM-03 + CAM-06

Topology does not participate in Phase 1; it is recorded for later room fusion.

## Phase 1 acceptance gate

Do not add inference until the camera core survives a real-machine soak test.

Required behavior:

- all six cameras remain close to their source cadence;
- source queues stay at 0/1 and never grow;
- latency does not increase over time;
- reconnecting one source does not freeze the other five;
- memory usage does not grow continuously;
- NVDEC is used and CPU load remains bounded;
- the wall remains responsive for at least 30–60 minutes.

## Phase 2 — RF-DETR-S person detector

Only after Phase 1 passes:

```text
six-camera DeepStream core
  -> GPU preprocess / detector branch
  -> RF-DETR-S
  -> person-only detections
```

The detector is not allowed to block camera ingest or display. The production target is ONNX/TensorRT FP16 rather than synchronous PyTorch inference in the camera loop. Detector input resolution is independent from the display/source resolution.

Detection quality is measured primarily by person recall on the actual CCTV domain: seated, partially occluded, small/far and edge-of-frame people.

## Later phases

```text
Phase 3: RF-DETR-S -> NvDCF camera-local tracking
Phase 4: tracker metadata -> room geometry + ReID side path
Phase 5: same-room one-to-one fusion -> Global ID
Phase 6: face as identity anchor
Phase 7: persistent long-term identity memory
Phase 8: room-to-room topology and all-six-camera global identity
Phase 9: API/database/frontend integration
```

Hot-path invariant for every later phase:

```text
RTSP -> decode -> detector/tracker -> live video
```

must never wait for:

```text
ReID / face / long-term memory / database / API / UI
```

Each production responsibility has one implementation. Experimental forks such as `*_final`, `*_v2`, `*_reid_heatmap` are not part of the Vision V3 ownership model.
