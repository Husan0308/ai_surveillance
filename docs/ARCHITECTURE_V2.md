# Sentinel Camera V2 — Production Architecture

Base branch: `rebuild/gpu-v2-clean`

## Design rule

There is one production camera path. Experimental API, MJPEG frontend, reference UI, ReID/Qwen and pose forks are not kept beside production unless they are actively integrated and tested.

The non-negotiable priorities are:

1. RTSP ingest and visible camera motion must remain responsive.
2. One slow or reconnecting source must not freeze the others.
3. The display hot path remains NVIDIA/GPU-native.
4. Detector work consumes fresh frames only and cannot accumulate stale queues.
5. NvDCF owns local per-camera track continuity.
6. UI statistics must come from runtime state, never demo fixtures.
7. Persistence must acknowledge only writes that actually succeeded.

## Active runtime

```text
config/cameras.yaml
        |
        v
nvurisrcbin per camera
  RTSP auth / reconnect / NVDEC
        |
        v
queue(max-size-buffers=1, leaky)
        |
        v
nvstreammux
        |
        +---------------- detector sampling ----------------+
        |                                                   |
        |                                   YOLO person-only worker
        |                                                   |
        |                                    detector metadata/result
        |                                                   |
        v                                                   v
NvDCF tracker <---------------------------------------------+
        |
        v
native metadata / bbox labels
        |
        v
camera-space heatmap metadata/filter
        |
        v
nvmultistreamtiler
        |
        v
nveglglessink -> native Qt/X11 wall
```

The visible wall is controlled by `services/camera_v2/sentinel_video_pro.py`. The production process constructs `CameraPersonTrackingHeatmap`; the older ReID runtime is not a fallback.

## Main modules

- `config.py`: camera and DeepStream configuration.
- `main.py`, `dynamic_wall.py`, `secure.py`: RTSP/GStreamer wall foundation.
- `detection.py`: bounded YOLO worker and detector results.
- `detector_latency.py`: fresh-result scheduling/latency support.
- `person_tracking.py`: NvDCF integration.
- `person_tracking_final.py`: final tracker/display metadata behavior.
- `person_tracking_heatmap.py`: active camera-space heatmap layer.
- `native_bridge.py` + active `native_*.c`: DeepStream metadata helpers.
- `sentinel_video.py`: common process/controller and Qt wall primitives.
- `sentinel_video_pro.py`: only production pipeline controller.
- `sentinel_video_wall_ui.py`: production native video-wall widget behavior.
- `sentinel_ui*.py`: application shell and pages.
- `sentinel_store.py`: local SQLite/profile/event persistence.

## Geometry and freshness contracts

Current preflight enforces the production profile used by the launcher:

```text
source-preserving mux: 2560x1440
monitoring wall:       1600x1350 (2 columns x 3 rows)
fullscreen focus:      1920x1080
detector input:        736x416
tracker input:         512x288
```

These are deployment contracts, not universal optimum values. Change them only with measured camera smoothness, detector latency and GPU memory/utilization results.

Queues on the live path are deliberately latest-only where possible. A higher average detector FPS is not an improvement if it increases frame age or makes camera motion less responsive.

## Identity status

Cross-camera ReID, Qwen visual verification and face recognition are currently **not part of the production runtime**. Monitoring therefore does not infer a known identity from an enrollment profile.

Enrollment currently provides real persistence only:

```text
10 selected images
    -> validate
    -> copy to .runtime/sentinel/people/<person-id>/
    -> SQLite profile row
```

A future identity subsystem must be added behind a bounded asynchronous interface, with explicit matching thresholds, conflict rules, rollback behavior and end-to-end tests. It must not own or block RTSP ingest, YOLO, NvDCF or display.

## Events and pages

`PeoplePage` reads real local profiles from `SentinelStore`.

`EventsPage` reads real persisted events. If runtime code has not recorded events, the page stays empty; no synthetic event history is generated.

`RoomsPage` groups the actual enabled cameras from `config/cameras.yaml`; it does not invent room occupancy or FPS.

## Startup validation

Production launcher:

```bash
bash scripts/run_sentinel_vms.sh
```

Preflights:

```bash
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py
```

The core preflight checks important source contracts and builds the native metadata/heatmap helpers against the target DeepStream installation. This means the final GPU/DeepStream validation must run on the deployment machine; a generic GitHub runner cannot prove that the NVIDIA runtime, RTSP network or X11/EGL stack works.

## Repository rule

A file stays in production only when at least one of these is true:

- it is on the active import/runtime path;
- it is required configuration or documentation for that path;
- it is an explicit preflight/test/tool used to validate that path.

Historical implementations belong in Git history, not as parallel source files in the working tree.
