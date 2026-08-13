# Core v1 Architecture

## Realtime hot path

```text
RTSP
 -> DeepStream `nvurisrcbin` when available, otherwise GStreamer/NVDEC
 -> decoded-frame queue bounded to one
 -> `LatestFrameStore`
 -> latest-only YOLO bridge
 -> isolated spawned CUDA process
 -> Kalman + Byte-style camera-local visual tracker
 -> JPEG publisher
 -> persistent HTTP reader in PySide6
```

The display path does not wait for ReID. There is one detector batch in flight; stale input/results are dropped instead of queued. `nvstreammux` and TensorRT `nvinfer` are not part of this GTX-1050-Ti Core-v1 path.

## ReID v2 side path

ReID consumes only exact real detector frames retained in a tiny bounded history. Prediction-only visual boxes cannot produce identity evidence.

```text
real YOLO observation
 -> crop quality gate
 -> local ReID track
 -> 3-6 diverse quality crops
 -> OSNet tracklet embeddings
 -> quality-weighted descriptor
 -> same-room pair similarity matrix
 -> one-to-one assignment + margin/evidence confirmation
 -> Global ID merge
```

Physical same-room pairs:

- ROOM-1: CAM-01 <-> CAM-04
- ROOM-2: CAM-02 <-> CAM-05
- ROOM-3: CAM-03 <-> CAM-06

Concurrent cross-room Global-ID sharing is blocked. ReID remains a bounded asynchronous side workload so camera capture, detection and smooth display keep running even if ReID is slow or unavailable.

## Repository ownership

```text
config/
  cameras.yaml
  core_v1.yaml
requirements/
  base.txt
  api.txt
  ml.txt
  frontend.txt
services/
  api_service/core_v1/
  frontend/core_v1/
  ml_service/core_v1/
  ml_service/cameras/{deepstream.py,gstreamer.py}
shared/config/
scripts/{core_v1_soak.py,core_v1_reid_v2_check.py}
docs/core_v1_freeze_checklist.md
```

Everything else from the previous monolithic/legacy architecture is intentionally excluded from this clean branch.
