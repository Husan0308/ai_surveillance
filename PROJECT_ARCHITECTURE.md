# Vision V3 Clean Architecture

`rebuild/vision-v3-clean` is a clean rebuild. Old experimental runtimes are donors only; each production responsibility has one implementation.

## Phase 1 — six-camera smooth core

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
- Old frames are dropped instead of queued.
- `sync-inputs=0`: one slow camera must not stall the other five.
- Decode remains GPU-native and the wall is independent from ML side paths.
- Runtime prints per-camera FPS, PTS p50/p95 cadence and source queue level.

Camera topology remains:

- Devs: CAM-01 + CAM-04
- Entrance: CAM-02 + CAM-05
- Main Rooms: CAM-03 + CAM-06

## Phase 2 — RF-DETR-S + proven Core-v1 detection policy

The model is RF-DETR-S, but the detector policy is ported from the mature Core-v1 six-camera implementation instead of inventing another bbox stack.

```text
nvurisrcbin / NVDEC
        |
        +-> latest-only display -> nvstreammux -> tiler -> OSD -> EGL
        |
        +-> ticketed latest-only detector tap
              -> selected fresh frame only
              -> isolated RF-DETR-S CUDA process
              -> low-threshold person observations
              -> optional difficult-camera ROI recovery
              -> hard false-positive masks
              -> full-frame/ROI confidence-first fusion
              -> raw RF-DETR person boxes
              -> Core-v1 adaptive Kalman + Byte visual continuity
              -> display-only body guard
              -> DeepStream OSD metadata
```

### Detection policy carried over from Core-v1

- low raw confidence floor for seated, small and partly occluded people;
- weak detections may continue an existing track but do not immediately create one;
- temporal birth confirmation for new low-confidence people;
- Byte-style high/low confidence association;
- adaptive Kalman motion state `[cx, cy, w, h, vx, vy, vw, vh]`;
- detector capture-time motion compensation, so a delayed box is projected toward the current display frame instead of trailing behind a walking person;
- center and size correction are controlled separately so the rectangle follows motion without breathing on every detector edge change;
- bounded prediction, velocity damping, reversal damping and stale-prediction rejection;
- confidence-first duplicate suppression using IoU + containment + center distance;
- fragment duplicate suppression for cameras with ROI recovery;
- selective ROI `verify` / `augment` second passes and optional 0/90/180/270 recovery;
- per-camera hard exclusion masks for known static false-positive regions;
- one in-flight latest-only detector job and stale-result rejection;
- detector/OOM failure never owns the six-camera display path.

### Raw box vs visible box

```text
RF-DETR raw bbox -> future NvDCF / geometry truth
RF-DETR raw bbox -> Core-v1 visual tracker -> display-only body guard -> OSD
```

The final body guard adds only small asymmetric display padding for visible head, hands and shoes. Padded/predicted geometry is never fed back into RF-DETR and will not become NvDCF/ReID/room-geometry evidence.

Primary Phase-2 files:

```text
config/vision_v3_detector.yaml
services/ml_service/vision_v3/rfdetr_worker_v2.py
services/ml_service/vision_v3/visual_tracker.py
services/ml_service/vision_v3/core_v1_visual_adapter.py
services/ml_service/vision_v3/rfdetr_runtime.py
services/ml_service/vision_v3/native_boxes.py
services/ml_service/vision_v3/native_boxes.c
scripts/setup_vision_v3_rfdetr.sh
scripts/preflight_vision_v3_rfdetr.py
scripts/run_vision_v3_rfdetr.sh
```

`visual_tracker.py` is the mature Core-v1 adaptive Kalman/Byte implementation copied into the clean Vision V3 ownership tree. The previous prototype smoothers and compatibility runtime were removed.

## Acceptance gates

- all six streams remain close to source cadence;
- source queues stay at 0/1;
- latency does not grow over time;
- one reconnecting source does not freeze the other five;
- RF-DETR failure/OOM cannot block camera display;
- detector duty backs off before display smoothness is sacrificed;
- difficult/occluded people improve without one-frame weak false positives becoming stable boxes;
- duplicates from full-frame + ROI passes are fused conservatively;
- boxes follow current people rather than visibly trailing old detector frames.

## Later phases

```text
Phase 3: RF-DETR-S RAW boxes -> NvDCF camera-local tracking
Phase 4: tracker metadata -> room geometry + ReID side path
Phase 5: same-room one-to-one fusion -> Global ID
Phase 6: face as identity anchor
Phase 7: persistent long-term identity memory
Phase 8: room-to-room topology and all-six-camera global identity
Phase 9: API/database/frontend integration
```

Hot-path invariant:

```text
RTSP -> decode -> detector/tracker -> live video
```

must never wait for:

```text
ReID / face / long-term memory / database / API / UI
```
