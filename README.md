# AI Surveillance — Camera Foundation

Three independent services: `ml_service`, `api_service`, and `frontend`.
Current phase is limited to six RTSP cameras and NVIDIA hardware decode/display transport.

## Current video architecture

```text
CAM-01 RTSP -> rtspsrc -> depay -> parser -> nvv4l2decoder -> latest queue(1) -> nvvideoconvert -> appsink
CAM-02 RTSP -> rtspsrc -> depay -> parser -> nvv4l2decoder -> latest queue(1) -> nvvideoconvert -> appsink
CAM-03 RTSP -> rtspsrc -> depay -> parser -> nvv4l2decoder -> latest queue(1) -> nvvideoconvert -> appsink
CAM-04 RTSP -> rtspsrc -> depay -> parser -> nvv4l2decoder -> latest queue(1) -> nvvideoconvert -> appsink
CAM-05 RTSP -> rtspsrc -> depay -> parser -> nvv4l2decoder -> latest queue(1) -> nvvideoconvert -> appsink
CAM-06 RTSP -> rtspsrc -> depay -> parser -> nvv4l2decoder -> latest queue(1) -> nvvideoconvert -> appsink
                                                               |
                                                               v
                                                      LatestFrameStore(1)
                                                               |
                                                               v
                                                        MJPEG publisher
                                                               |
                                                               v
                                                /video/CAM-01 ... CAM-06
                                                               |
                                                               v
                                                       PySide6 3x2 wall
```

Each camera owns an independent capture pipeline. A bad/reconnecting camera cannot block the other five. The one-frame leaky queue and one-frame appsink keep the path latest-only.

## Display sources

- CAM-01: `.../101`, H.264
- CAM-02: `.../201`, H.264
- CAM-03: `.../301`, H.264
- CAM-04: `.../601`, H.265
- CAM-05: `.../501`, H.265
- CAM-06: `.../401`, H.265

RTSP transport defaults to `auto`, matching the earlier working project. GStreamer can negotiate UDP/TCP automatically.

## RTSP authentication

The RTSP URL stays clean in `config/cameras.yaml`. Credentials are stored only in the gitignored `.env` file and are passed to GStreamer through the native `rtspsrc user-id/user-pw` properties.

When all six channels use one NVR account:

```bash
python scripts/setup_rtsp_auth.py
```

This creates/updates `.env` with mode `600`:

```text
SURVEILLANCE_RTSP_USERNAME=...
SURVEILLANCE_RTSP_PASSWORD=...
```

Optional per-camera overrides are also supported:

```text
CAM_01_RTSP_USERNAME=...
CAM_01_RTSP_PASSWORD=...
```

## ML setup

```bash
sudo apt update
sudo apt install -y python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 python3-venv

python3 -m venv --system-site-packages .venv-ml
source .venv-ml/bin/activate
pip install -r services/ml_service/requirements.txt

./scripts/check_deepstream.sh
```

Before starting ML, diagnose the RTSP server itself:

```bash
python scripts/probe_rtsp_server.py
```

Important outcomes:

```text
DESCRIBE RTSP/1.0 200 OK        -> path reachable without auth challenge
DESCRIBE RTSP/1.0 401 ...       -> configure RTSP credentials
DESCRIBE RTSP/1.0 404 ...       -> wrong camera/channel path
connection refused/timeout      -> network/NVR/RTSP-port problem
```

Then test actual NVIDIA decoding sequentially:

```bash
python scripts/probe_cameras.py
```

Only after the probe reaches `6/6` start ML:

```bash
python -m services.ml_service.app.main
```

Healthy startup prints one pair per camera:

```text
[CAMERA] CAM-01 first frame 736x416 ...
[MJPEG] CAM-01 first JPEG ...
```

Diagnostics:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/cameras | python -m json.tool
curl --max-time 3 http://127.0.0.1:8001/video/CAM-01 -o /tmp/cam1.mjpeg
```

## API service

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

## Intentionally not implemented yet

YOLO, tracking, ReID, face recognition, database, heatmap, recording, alerts, and business UI are still excluded until the six-camera transport is stable.
