# AI Surveillance — Sentinel Camera V2

Current stabilization branch: `agent/rfdetr-s-core-final`.

## Canonical production path

The deployment machine uses a GTX 1050 Ti. DeepStream 7.1 does not list Pascal in
its validated x86 dGPU matrix, and the hardware smoke logs showed NvDCF stopping
downstream after the first tracker batch while RTSP, NVDEC, nvstreammux and
RF-DETR continued to work. The production launcher therefore does **not** insert
`gst-nvtracker` on this machine.

```text
6 RTSP cameras
    ↓
nvurisrcbin / NVDEC
    ↓
per-camera latest-frame tee
    ├── sparse RF-DETR-S side capture (672x384, micro-batch 1)
    └── GPU display path
            ↓
        nvstreammux (2560x1440)
            ↓
        bounded motion-predictor bbox metadata
            ↓
        nvmultistreamtiler (2 columns x 3 rows)
            ↓
        nvvideoconvert -> RGBA NVMM
            ↓
        nvdsosd
            ↓
        nveglglessink
            ↓
        one persistent Qt/X11 native QWidget XID
```

The active UI is intentionally camera-only. It does not instantiate the legacy
Sentinel dashboard, People/Events/Rooms pages, MJPEG frontend or old Qt runtime.

## Active files

- `scripts/run_sentinel_vms.sh` — production launcher and hardware profile.
- `services/camera_v2/sentinel_ui.py` — minimal maximized Qt shell.
- `services/camera_v2/sentinel_ui_monitoring_native.py` — one native video host.
- `services/camera_v2/camera_wall_runtime.py` — process-isolated XID/GStreamer controller.
- `services/camera_v2/pascal_safe_pipeline.py` — no-NvDCF RF-DETR runtime.
- `services/camera_v2/rfdetr_backend.py` — RF-DETR-S CUDA worker.
- `services/camera_v2/detection.py` — side-capture scheduler and motion predictor.
- `services/camera_v2/dynamic_wall.py` / `main.py` / `secure.py` — RTSP/NVDEC/mux/tiler/EGL core.
- `config/cameras.yaml` — authoritative six-camera topology.

Legacy ReID, NvDCF, historical Sentinel UI and MJPEG service modules remain in the
repository for later migration/cleanup, but they are not part of the production
launcher above.

## Camera topology

`config/cameras.yaml` is authoritative:

- `Devs`: CAM-01 + CAM-04
- `Entrance`: CAM-02 + CAM-05
- `Main Rooms`: CAM-03 + CAM-06

RTSP usernames/passwords are loaded from `.env`, normally through:

```text
SURVEILLANCE_RTSP_USERNAME=...
SURVEILLANCE_RTSP_PASSWORD=...
```

Do not commit real credentials.

## Why NvDCF is disabled on this deployment

The observed failure was not an RTSP or detector failure. All six streams reached
~20 FPS, RF-DETR produced detections, and external inference metadata was marked,
but NvDCF stopped advancing while the EGL sink remained at zero rendered frames.

For the GTX 1050 Ti stabilization path, temporal continuity is therefore provided
by the existing bounded `SmoothBoxManager` motion predictor. It does not use pixel
features or DeepStream's low-level tracker library. This is a display/local-motion
fallback, not cross-camera ReID.

## Preflight

The launcher runs these checks automatically:

```bash
python scripts/preflight_rfdetr_core.py
python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py
```

Important runtime diagnostics after startup:

```text
CAMERA_INFER_LAYOUT ... stride=... tight=...
CAMERA_PASCAL_SAFE mux_batches=... wall_frames=... rendered=...
```

`mux_batches` and `wall_frames` must continually increase. If they increase while
`rendered` stays at zero, the remaining problem is isolated to EGL/X11 presentation
rather than RTSP, detector or tracking.

## Run

```bash
source .venv/bin/activate
bash scripts/run_sentinel_vms.sh 2>&1 | tee /tmp/camera-direct.log
```

Expected startup markers include:

```text
RFDETR_PREFLIGHT=PASS
PASCAL_SAFE_PREFLIGHT=PASS
SENTINEL_UI_PREFLIGHT=PASS
CAMERA_PREFLIGHT=PASS
CAMERA_PASCAL_SAFE ready backend=RF-DETR-S tracker=motion-predictor nvtracker=disabled
```

The production wall is fixed at six cameras in a 2x3 layout.
