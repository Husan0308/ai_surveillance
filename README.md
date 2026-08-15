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

`nvurisrcbin` is not used in the display hot path. Explicit `rtspsrc` gives the application direct RTSP errors, while NVIDIA `nvv4l2decoder` and `nvvideoconvert` keep decode/scale accelerated. The worker owns reconnection, so there is only one reconnect controller instead of an internal reconnect loop plus an external restart loop.

## Display sources

- CAM-01: `.../101`, H.264
- CAM-02: `.../201`, H.264
- CAM-03: `.../301`, H.264
- CAM-04: `.../601`, H.265
- CAM-05: `.../501`, H.265
- CAM-06: `.../401`, H.265

The default RTSP transport is TCP. Optional camera credentials are environment-only, for example `CAM_01_RTSP_USERNAME` and `CAM_01_RTSP_PASSWORD`.

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
