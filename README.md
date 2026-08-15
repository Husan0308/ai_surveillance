# AI Surveillance — Fresh Foundation

Fresh three-service foundation. Phase 1 is deliberately limited to **six RTSP cameras + NVIDIA DeepStream**.

## Architecture

```text
6 RTSP cameras
      |
      v
+---------------------------+
| ml_service :8001          |
|                           |
| nvurisrcbin x6            |
|      |                    |
|      v                    |
| nvstreammux batch=6       |
|      |                    |
|      v                    |
| nvmultistreamtiler 3x2    |
|      |                    |
|      v                    |
| nveglglessink             |
+---------------------------+

+---------------------------+       +---------------------------+
| api_service :8000         |       | frontend                  |
| FastAPI boundary only     |       | PySide6 client shell      |
+---------------------------+       +---------------------------+
```

The three services do not import each other's application code. Camera ingest/decode belongs only to `ml_service`.

## Phase 1 intentionally excludes

- YOLO / inference
- tracking
- ReID
- face recognition
- database
- heatmap
- recording
- alerts
- business UI

## System requirements

DeepStream, GStreamer and PyGObject (`gi`) are system dependencies, not normal pip-only dependencies.

```bash
sudo apt update
sudo apt install -y \
  python3-gi \
  python3-gst-1.0 \
  gir1.2-gstreamer-1.0 \
  python3-venv

./scripts/check_deepstream.sh
```

The expected DeepStream plugins are:

- `nvurisrcbin`
- `nvstreammux`
- `nvmultistreamtiler`
- `nveglglessink`

## Run each service independently

### ML service

`python3-gi` is installed by APT into the system Python site-packages. Therefore the ML virtual environment must be allowed to see system site-packages.

```bash
rm -rf .venv-ml
python3 -m venv --system-site-packages .venv-ml
source .venv-ml/bin/activate
pip install -r services/ml_service/requirements.txt

python -c 'import gi; gi.require_version("Gst", "1.0"); from gi.repository import Gst; print("GI/GStreamer OK", Gst.version_string())'

python -m services.ml_service.app.main
```

Health:

```bash
curl http://127.0.0.1:8001/health
```

The ML process starts the DeepStream six-camera graph and, with `display.enabled: true`, opens the 3x2 DeepStream view.

### API service

```bash
python3 -m venv .venv-api
source .venv-api/bin/activate
pip install -r services/api_service/requirements.txt
python -m services.api_service.app.main
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

### Frontend

```bash
python3 -m venv .venv-frontend
source .venv-frontend/bin/activate
pip install -r services/frontend/requirements.txt
python -m services.frontend.app.main
```

The frontend is intentionally only a shell in phase 1.

## Camera config

Edit `config/cameras.yaml`, or override a camera URI without changing Git history:

```bash
export CAM_01_URI='rtsp://...'
```

Equivalent variables exist through `CAM_06_URI`.

## DeepStream baseline

The baseline uses:

- `nvurisrcbin` for each RTSP source and NVIDIA decoder path
- `rtsp-reconnect-interval` with unlimited reconnect attempts
- `drop-on-latency=true`
- GPU decoder memory (`cudadec-memtype=0`)
- `nvstreammux batch-size=6`
- `live-source=true`
- `sync-inputs=false` so a slow camera does not intentionally synchronize all six feeds
- 3x2 `nvmultistreamtiler`

No AI element is inserted until this camera-only baseline is stable.
