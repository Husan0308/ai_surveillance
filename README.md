# AI Surveillance — Camera Foundation

Fresh three-service architecture. Current phase is intentionally limited to **six RTSP cameras + NVIDIA DeepStream display transport**.

## Architecture

```text
                         CONTROL / METADATA
Frontend ──────────────> API :8000 ──────────────> ML :8001
   │
   │                     VIDEO PLANE
   └─────────────────────────────────────────────> ML :8001/video/CAM-xx

Inside ML:

CAM-01 RTSP -> nvurisrcbin -> latest-only queue -> nvvideoconvert -> appsink -> LatestFrameStore(1) -> JPEG
CAM-02 RTSP -> nvurisrcbin -> latest-only queue -> nvvideoconvert -> appsink -> LatestFrameStore(1) -> JPEG
CAM-03 RTSP -> nvurisrcbin -> latest-only queue -> nvvideoconvert -> appsink -> LatestFrameStore(1) -> JPEG
CAM-04 RTSP -> nvurisrcbin -> latest-only queue -> nvvideoconvert -> appsink -> LatestFrameStore(1) -> JPEG
CAM-05 RTSP -> nvurisrcbin -> latest-only queue -> nvvideoconvert -> appsink -> LatestFrameStore(1) -> JPEG
CAM-06 RTSP -> nvurisrcbin -> latest-only queue -> nvvideoconvert -> appsink -> LatestFrameStore(1) -> JPEG
```

There is deliberately **no nvstreammux in the camera display path**. Each camera has its own DeepStream/NVDEC pipeline, so one slow or reconnecting camera cannot stall the other five.

The frontend keeps one persistent MJPEG HTTP connection per camera and always replaces the previous image with the newest one. There is no per-frame HTTP polling and no frame backlog.

## Camera display sources

- CAM-01: `.../101`
- CAM-02: `.../201`
- CAM-03: `.../301`
- CAM-04: `.../601`
- CAM-05: `.../501`
- CAM-06: `.../401`

These are the display sources preserved from the earlier working project.

## Phase intentionally excludes

- YOLO / inference
- tracking
- ReID
- face recognition
- database
- heatmap
- recording
- alerts

## ML setup

```bash
sudo apt update
sudo apt install -y python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 python3-venv

rm -rf .venv-ml
python3 -m venv --system-site-packages .venv-ml
source .venv-ml/bin/activate
pip install -r services/ml_service/requirements.txt

./scripts/check_deepstream.sh
python -m services.ml_service.app.main
```

Expected startup logs include:

```text
[CAMERA] CAM-01 first frame 736x416
[MJPEG] CAM-01 first JPEG ...
```

Tests:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/cameras
curl -v http://127.0.0.1:8001/video/CAM-01 -o /dev/null
```

## API

```bash
source .venv-api/bin/activate
pip install -r services/api_service/requirements.txt
python -m services.api_service.app.main
```

## Frontend

```bash
source .venv-frontend/bin/activate
pip install -r services/frontend/requirements.txt
python -m services.frontend.app.main
```

The PySide6 frontend renders six persistent MJPEG streams in a 3x2 wall.
