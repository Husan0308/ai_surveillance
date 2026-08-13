# AI Surveillance — Core v1 Clean

This branch contains only the current three-service rebuild and the files it actually uses.

## Current architecture

```text
6 x RTSP cameras
  -> DeepStream nvurisrcbin / GStreamer NVDEC capture
  -> per-camera LatestFrameStore (latest-only hot path + tiny bounded ReID history)
  -> isolated PyTorch/Ultralytics YOLO CUDA worker, one in-flight batch
  -> per-camera Kalman + Byte-style visual tracker
  -> smooth JPEG presentation -> PySide6 frontend
  -> side-path ReID v2: quality-gated multi-frame tracklets -> OSNet -> same-room pair assignment -> Global ID
```

Camera pairs that view the same room are CAM-01/CAM-04, CAM-02/CAM-05, and CAM-03/CAM-06. Visual prediction boxes are presentation-only and are never ReID evidence.

## Services

- `services/ml_service/core_v1`: camera orchestration, YOLO, smooth visual tracking, ReID v2, Global ID, MJPEG/frame endpoints and telemetry.
- `services/api_service/core_v1`: lightweight API facade for the current rebuild.
- `services/frontend/core_v1`: six-camera PySide6 viewer using persistent HTTP connections to ML.
- `services/ml_service/cameras`: only the two capture backends required by Core v1 (`deepstream.py`, `gstreamer.py`).
- `shared/config`: camera config loader with optional ignored local overrides.

## Configuration

- `config/cameras.yaml` — canonical camera definitions.
- `config/cameras.local.yaml` — optional local/secret overrides; ignored by git.
- `config/core_v1.yaml` — current runtime, detector, tracker and ReID-v2 settings.

## Run

ML:

```bash
python -m services.ml_service.core_v1.main
```

API:

```bash
python -m services.api_service.core_v1.main
```

Frontend:

```bash
python -m services.frontend.core_v1.main
```

Useful checks:

```bash
python scripts/core_v1_soak.py --minutes 30
python scripts/core_v1_reid_v2_check.py --seconds 60
```

Machine-local models, databases, captures, `.runtime/`, logs and `cameras.local.yaml` must not be committed.
