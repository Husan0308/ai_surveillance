# AI Surveillance — Sentinel Camera V2

Production base branch: `rebuild/gpu-v2-clean`.

This repository now keeps one production path instead of several competing camera, API, frontend, ReID and reference-UI experiments.

## Production path

```text
config/cameras.yaml
        ↓
6 RTSP cameras
        ↓
DeepStream nvurisrcbin / NVDEC
        ↓
latest-frame queues + nvstreammux
        ↓
YOLO person detection
        ↓
per-camera NvDCF tracking
        ↓
native metadata / labels
        ↓
optional camera-space heatmap
        ↓
nvmultistreamtiler
        ↓
Sentinel PySide6 monitoring wall
```

The production launcher is:

```bash
bash scripts/run_sentinel_vms.sh
```

It runs the static Sentinel/UI preflight and Camera V2 core preflight before starting:

```bash
python -m services.camera_v2.monitor_ui
```

## What is intentionally not claimed

Cross-camera person ReID, Qwen identity verification and face recognition are **not wired into the current production runtime**. They were removed from this cleanup because the active launcher did not use them and keeping experimental implementations next to production made failures and ownership ambiguous.

The current monitoring metrics therefore must not pretend that a detected person is known. Enrollment persists a worker profile and ten selected face images locally, but it does not yet make the live detector recognize that person.

When identity is added again, it should be integrated as one tested production component with explicit persistence, thresholds, failure behavior and end-to-end tests instead of another parallel pipeline.

## Camera configuration

`config/cameras.yaml` is authoritative for camera ID, display name, room, enabled state and RTSP URI.

Current topology:

- `Devs`: CAM-01 + CAM-04
- `Entrance`: CAM-02 + CAM-05
- `Main Rooms`: CAM-03 + CAM-06

The Settings page writes the same configuration file. Up to 16 cameras can be configured, while the current monitoring wall supports at most 6 enabled cameras.

RTSP credentials belong in `.env`, never in Git:

```bash
cp .env.example .env
python scripts/setup_rtsp_auth.py
```

Global variables:

```text
SURVEILLANCE_RTSP_USERNAME
SURVEILLANCE_RTSP_PASSWORD
```

Optional per-camera overrides use names such as `CAM_01_RTSP_USERNAME` and `CAM_01_RTSP_PASSWORD`.

## Python environment

DeepStream, NVIDIA GStreamer plugins and PyGObject/GI are system dependencies and must match the installed NVIDIA stack. Use the machine's working CUDA-compatible PyTorch installation rather than blindly replacing it.

Application Python dependencies are listed in `requirements.txt`.

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Before running the wall, verify CUDA from the same environment:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

## Preflight

Run these from the repository root:

```bash
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py
```

For RTSP/NVR authentication debugging:

```bash
python scripts/probe_rtsp_server.py
```

For DeepStream plugin checks:

```bash
bash scripts/check_deepstream.sh
```

## Run

```bash
bash scripts/run_sentinel_vms.sh
```

The launcher requires an X11 display for the native `nveglglessink` video overlay used by the PySide6 wall.

## Local persistence

Sentinel local state is stored under `.runtime/sentinel/` and is gitignored:

```text
.runtime/sentinel/sentinel.db
.runtime/sentinel/people/
.runtime/sentinel/events/
```

Enrollment copies the selected images into this directory and writes the profile to SQLite. Events are displayed only when real runtime code records them; the UI no longer manufactures demo people or demo event history.

## Main code ownership

```text
services/camera_v2/config.py                 camera/deepstream config
services/camera_v2/detection.py              YOLO worker + detector metadata
services/camera_v2/person_tracking.py        NvDCF tracking layer
services/camera_v2/person_tracking_final.py  final tracking/display behavior
services/camera_v2/person_tracking_heatmap.py camera-space heatmap layer
services/camera_v2/sentinel_video_pro.py     production pipeline process/controller
services/camera_v2/sentinel_ui*.py           PySide6 application/pages
services/camera_v2/sentinel_store.py         SQLite/profile/event persistence
services/camera_v2/native_*.c                 active DeepStream native helpers
```

Do not add a second API/frontend/camera pipeline unless there is a concrete requirement that the production runtime cannot satisfy. A second implementation without a migration plan increases maintenance cost and makes bug reports ambiguous.
