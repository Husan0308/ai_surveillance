# AI Surveillance — Core v1 Clean

This branch contains only the current three-service rebuild and the files it actually uses.

## Current architecture

```text
6 x RTSP cameras
  -> DeepStream nvurisrcbin / GStreamer NVDEC capture
  -> per-camera LatestFrameStore (latest-only hot path + tiny bounded ReID history)
  -> isolated PyTorch/Ultralytics CUDA worker
       -> YOLO26m detector, one in-flight batch
       -> sparse crop-based YOLO26m-pose in the SAME process/CUDA context
  -> per-camera Kalman + Byte-style visual tracker
  -> smooth JPEG presentation -> PySide6 frontend
  -> side-path ReID: exact-frame tracklets -> OSNet -> optional calibrated spatial fusion -> Global ID
  -> pose ankles -> calibrated normalized room floor -> one-hour-hold heatmap
```

Camera pairs that view the same room are CAM-01/CAM-04, CAM-02/CAM-05, and CAM-03/CAM-06. Visual prediction boxes are presentation-only and are never ReID evidence. Pose is sampled after detector results are published, so the primary detector/tracker does not wait for pose. Detector and pose intentionally share one spawned CUDA process to avoid a second PyTorch CUDA process/context.

## Services

- `services/ml_service/core_v1`: camera orchestration, YOLO26m detection, sparse YOLO26m-pose, smooth visual tracking, ReID v2, Global ID, floor heatmaps, MJPEG/frame endpoints and telemetry.
- `services/api_service/core_v1`: lightweight API facade for the current rebuild.
- `services/frontend/core_v1`: six-camera PySide6 viewer using persistent HTTP connections to ML.
- `services/ml_service/cameras`: only the two capture backends required by Core v1 (`deepstream.py`, `gstreamer.py`).
- `shared/config`: camera config loader with optional ignored local overrides.

## Configuration

- `config/cameras.yaml` — canonical camera definitions.
- `config/cameras.local.yaml` — optional local/secret overrides; ignored by git.
- `config/core_v1.yaml` — current runtime, detector, tracker, unified pose and ReID settings.
- `config/room_mapping.yaml` — verified room pairs, normalized floor calibration and fusion settings.

## Room calibration

Open `Room Map` in the PySide6 UI. Automatic relation checking is on-demand and
never saves a guessed floor plane. If its confidence is insufficient, select a
camera and click 6–8 matching stationary floor landmarks in the live image and
normalized room map. Spatial fusion stays disabled for that room until both
cameras have a valid persisted homography.

The mapping API is available at `/room-mapping`; assisted calibration uses
`/room-mapping/calibrate`. Person room coordinates always come from the real
detector box bottom-center, never from presentation-predicted boxes. Floor
heatmap coordinates come from YOLO26m-pose ankle keypoints projected through the
same validated camera-to-floor homography.

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
python -m unittest tests.test_floor_heatmap tests.test_unified_pose_worker -v
python scripts/core_v1_soak.py --minutes 30
python scripts/core_v1_reid_v2_check.py --seconds 60
```

After startup, `/health` should report detector `cuda_topology` as
`single_process_detector_and_pose`; the pose metrics should show
`shared_cuda_process: true` and model `yolo26m-pose.pt`.

Machine-local models, databases, captures, `.runtime/`, logs and `cameras.local.yaml` must not be committed.
