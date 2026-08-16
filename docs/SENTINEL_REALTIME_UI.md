# Sentinel VMS realtime UI

This branch wires the supplied Sentinel PySide6 design to the real Core v1 surveillance stack.

## Runtime data flow

```text
6 RTSP cameras
  -> CameraManager
  -> YOLO26m person detection (CUDA)
  -> camera-local ownership-locked ByteTrack/Hungarian tracking
  -> room-consensus OSNet ReID
  -> low-rate InsightFace recognition
  -> ownership-tracking latest-only mmap publishers
  -> six decode-free SmoothMmapFrameReader consumers
  -> Sentinel PySide6 UI
```

The Sentinel camera wall does not use the legacy `/video/{camera}` MJPEG path. Annotated BGR presentation frames are published locally through the existing SIGBUS-safe double-buffer mmap transport, so the UI hot path avoids JPEG encode, HTTP delivery and JPEG decode.

The UI does not create a second RTSP connection for fullscreen. Normal tiles, single-camera fullscreen dialogs and the all-camera fullscreen view share the same per-camera reader/feed objects.

## Realtime pages

- **Monitoring**: six live feeds, online/FPS state, known/unknown totals, recent identities.
- **People**: live + enrolled identities from `/tracks` and `/faces`; Face DB deletion is real.
- **Events**: realtime entry/exit/transition/face/camera-edge events derived from live tracker and service state.
- **Rooms**: realtime unique identity occupancy using the configured camera-to-room grouping.
- **Enrollment**: exactly ten selected image files; all ten are quality-gated by InsightFace and checked for embedding consistency before being persisted.
- **Reports**: charts and CSV/PDF exports are generated from the current realtime session instead of demo arrays.

## Camera hover controls

Every camera tile has two actions that are hidden until the pointer is over that camera:

- `Heatmap`: toggles only that camera's movement heatmap overlay. Accumulation continues from live track bottom-centers while the overlay is hidden.
- `Fullscreen`: opens only that camera fullscreen, sharing the same live feed.

Double-clicking a camera also opens its fullscreen view. The existing all-camera fullscreen action remains available.

## Start

Install the existing split requirements if needed:

```bash
pip install -r requirements/frontend.txt
pip install -r requirements/ml.txt
```

Run the preflight:

```bash
python scripts/preflight_sentinel_ui.py
```

Terminal 1:

```bash
PYTHONFAULTHANDLER=1 python -m services.ml_service.core_v1.main
```

Terminal 2:

```bash
PYTHONFAULTHANDLER=1 python -m services.frontend.core_v1.main
```

The first full-stack backend start can take longer when pinned ReID/InsightFace model assets are not already present.
