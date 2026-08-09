# Project Architecture

## Production data path

```text
6 × RTSP
  → GStreamer appsink + NVIDIA nvv4l2decoder/NVDEC
  → capacity-one latest-frame buffers
  → event-driven fresh-frame BatchScheduler
  → optimized batched YOLO/PyTorch detector
  → custom_motion_iou camera-local tracker
  → asynchronous OSNet ReID and InsightFace
  → GlobalIdentityManager
  → per-camera occupancy heatmaps + canonical domain events
  → FastAPI SQLite/WebSocket/MJPEG
  → PySide6 frontend
```

The production tracker is `custom_motion_iou`, not ByteTrack. The isolated ByteTrack adapter exists only for the completed P2 comparison. DeepStream/GStreamer provides hardware video ingestion and decode; detection remains the optimized YOLO/PyTorch path rather than `nvinfer`. Pose is disabled. Heatmaps are camera-space and are not floorplan/world-coordinate maps.

## Service ownership

- `services/ml_service`: RTSP readers, latest buffers, batching, YOLO, local tracking, asynchronous ReID/face work, global identity, unknown snapshots, heatmaps, ML metrics, MJPEG source frames, and the internal ML control API.
- `services/api_service`: public REST API, canonical SQLite storage, camera/person/enrollment/event/settings CRUD, ML control client, WebSocket fan-out, and persistence.
- `services/frontend`: PySide6 widgets, asynchronous REST client, WebSocket client, MJPEG rendering, synchronized overlays, and user interaction. It does not import ML or camera-reader modules.
- `shared`: service contracts, configuration, topology validation, event taxonomy, logging, and controlled enrollment staging.

Raw realtime frames are not transported through Redis or WebSocket. Video uses the ML MJPEG endpoint; WebSocket carries bounded metadata.

## Persistent state

`data/surveillance.db` is the runtime authority. SQLite WAL is enabled. `api_resources` stores cameras, persons, settings, enrollment sessions, heatmaps, and events. `api_face_embeddings` stores normalized float32 embeddings with dimension, model version, quality, metadata, and enabled state.

Unknown representative crops are stored under `data/snapshots/` with a bounded identity count, retention window, server-generated filenames, and quality replacement. Enrollment imports are copied by the same-host desktop client into `data/enrollment_staging/`; the API rejects paths outside that root and validates regular-file status, size, extension, and actual image decoding.

## Identity and topology

Local track IDs are camera-local. Global IDs are independent and bounded in memory. Appearance embeddings, camera history, and track history have explicit limits. A strong known-face assignment is locked against weak evidence; contradictory known evidence is recorded as a conflict.

Physical topology is configured only in `config/topology.yaml`. It remains `verified: false` until an on-site inspection confirms every room, overlap, adjacency, and travel time. Unverified cross-camera layout is never treated as a hard identity fact. Run `python scripts/topology_calibration.py` for a credential-masked inventory.

## Events and heatmaps

Persistent canonical events are `camera.online`, `camera.offline`, `person.identified`, `identity.conflict`, `enrollment.completed`, and `enrollment.failed`. Existing historical types are preserved and marked legacy by the API. `frame.metadata` is realtime transport data and is not a persistent business event.

Heatmaps use bbox bottom-center at a controlled interval and keep independent grids per camera. Live decay and bounded minute/hour/day accumulators prevent detector-rate or infinite accumulation.

## Lifecycle

Supported startup is `scripts/run_all.sh`. It starts API, waits for readiness, starts ML, waits for control readiness, then starts the frontend. SIGINT/SIGTERM are forwarded and child processes are joined. Individual entry points remain:

```bash
python -m services.api_service.app
python -m services.ml_service.app
python -m services.frontend.main
```

Use `python scripts/check_environment.py` for read-only prerequisite validation and `python scripts/backup_database.py` for SQLite's online backup API.
