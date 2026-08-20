# Vision V3 Clean Architecture

`rebuild/vision-v3-clean` is a clean rebuild. Old experimental runtimes are donors only; they are not the production ownership model.

## Phase 1 — six-camera smooth core

The camera foundation stays simple and GPU-native:

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
- Old frames are dropped instead of queued. Bounded latency is more important than replaying stale frames.
- `sync-inputs=0`: one slow camera must not stall the other five.
- Source decode remains GPU-native; `nvstreammux` scales to the configured working resolution for downstream efficiency.
- Runtime prints per-camera FPS, PTS p50/p95 cadence and source queue level every few seconds.

Primary camera files:

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

Topology does not participate in camera ingest; it is reserved for later room fusion.

## Phase 2 — RF-DETR-S person detector

RF-DETR-S is attached as a bounded asynchronous side path. The six-camera display does not wait for inference:

```text
nvurisrcbin / NVDEC
        |
        +-> latest-only display branch -> nvstreammux -> tiler -> OSD -> EGL
        |
        +-> ticketed detector branch
              -> queue(1, leaky)
              -> nvvideoconvert
              -> one requested CPU frame only
              -> isolated RF-DETR-S CUDA process
              -> person-only raw detections
              -> protective display envelope
              -> DeepStream object metadata
```

The detector branch has only one requested fresh frame per selected camera. Frames are not continuously copied to NumPy and detector work is round-robin across all six cameras. GPU duty is capped and automatically backed off when wall p95 cadence becomes unhealthy.

The first production-correctness backend is the official `RFDETRSmall` runtime. Once actual-camera recall and memory use are validated, the same detector contract can be exported to ONNX/TensorRT without changing camera ownership or downstream tracking contracts.

### Raw box vs visible box

Two boxes are intentionally kept conceptually separate:

```text
RF-DETR raw bbox -> future NvDCF / geometry truth
RF-DETR raw bbox -> protective envelope -> OSD only
```

The protective envelope exists because CCTV person boxes can be visually too tight around the head, shoes, hands or a seated/crouched body. It:

- adds asymmetric guard space (more below the feet than above the head);
- adds extra side/bottom guard for short/wide seated or crouched people;
- enforces a minimum pixel guard for small/far people;
- expands quickly when new evidence shows more body;
- shrinks slowly so one tight detector frame does not cut off limbs;
- predicts position briefly between sparse detector observations;
- clamps at image edges without feeding padded geometry back into the detector/tracker truth;
- suppresses near-duplicate high-IoU boxes before display.

Primary detector files:

```text
config/vision_v3_detector.yaml
services/ml_service/vision_v3/rfdetr_detection.py
services/ml_service/vision_v3/native_boxes.py
services/ml_service/vision_v3/native_boxes.c
scripts/setup_vision_v3_rfdetr.sh
scripts/preflight_vision_v3_rfdetr.py
scripts/run_vision_v3_rfdetr.sh
```

Detection quality is measured primarily by person recall on the actual CCTV domain: seated, partially occluded, small/far and edge-of-frame people. A lower starting threshold is used for recall; final threshold calibration must come from labeled frames from these cameras.

## Acceptance gates

Camera core must remain bounded while RF-DETR is active:

- all six streams remain close to source cadence;
- source queues stay at 0/1 and never grow;
- latency does not increase over time;
- reconnecting one source does not freeze the other five;
- memory usage does not grow continuously;
- RF-DETR failure/OOM cannot own or block the GStreamer camera process;
- detector side-path duty backs off before display smoothness is sacrificed;
- visible boxes contain raw RF-DETR boxes and do not suddenly shrink around a person.

## Later phases

```text
Phase 3: RF-DETR-S raw boxes -> NvDCF camera-local tracking
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
