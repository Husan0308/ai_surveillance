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
  -> side-path ReID: exact-frame tracklets -> OSNet -> optional calibrated spatial fusion -> Global ID
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
- `config/core_v1.yaml` — current runtime, detector, tracker and ReID settings.
- `config/room_mapping.yaml` — verified room pairs, normalized floor calibration and fusion settings.

## Room calibration

Open `Room Map` in the PySide6 UI. Automatic relation checking is on-demand and
never saves a guessed floor plane. If its confidence is insufficient, select a
camera and click 6–8 matching stationary floor landmarks in the live image and
normalized room map. Spatial fusion stays disabled for that room until both
cameras have a valid persisted homography.

The mapping API is available at `/room-mapping`; assisted calibration uses
`/room-mapping/calibrate`. Person room coordinates always come from the real
detector box bottom-center, never from presentation-predicted boxes.

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
