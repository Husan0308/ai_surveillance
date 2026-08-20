# AI Surveillance Architecture Contract

This repository has one canonical production architecture. Cleanup, refactors, UI work, model work, and performance work must preserve these service boundaries unless an explicit architecture migration is approved first.

## 1. ml_service — port 8001

Owns all camera and ML runtime work:

- RTSP ingest and reconnect handling
- DeepStream / NVDEC / GStreamer
- detector inference
- per-camera tracking
- cross-camera ReID and global identity
- face recognition
- heatmap analytics
- camera runtime metrics
- live video/JPEG publishing

Canonical entrypoint: `services.ml_service.app.main`.

`ml_service` must remain independently startable and independently health-checkable.

## 2. api_service — port 8000

Owns application/backend state and the public application API:

- REST API
- WebSocket/event delivery
- SQLite persistence
- people, events, enrollment and application state
- commands and queries to `ml_service`
- health aggregation

Canonical entrypoint: `services.api_service.app.main`.

`api_service` must not own RTSP decode or model inference.

## 3. frontend — PySide6 desktop client

Owns presentation only:

- Monitoring UI
- People / Events / Enrollment / Settings pages
- camera grid/fullscreen interactions
- API/WebSocket client behavior
- displaying video produced by `ml_service`

Canonical entrypoint: `services.frontend.app.main`.

The frontend must not instantiate DeepStream, YOLO, trackers, ReID, or face-recognition runtimes itself.

## Data flow

```text
RTSP cameras
    |
    v
ml_service :8001
    |  metrics / identity / video
    v
api_service :8000  <-> SQLite
    |
    v
frontend (PySide6)
```

For live video, the desktop frontend may consume the ML video endpoint directly to avoid an unnecessary API relay/copy. Control state, persisted data, health, people, events, enrollment, and commands belong behind `api_service`.

## camera_v2 rule

`services/camera_v2` is not a fourth production service. Useful code from it may be migrated into `ml_service` or retained as an internal implementation/experimental package, but the production application must still launch and operate through the three canonical services above.

## Non-negotiable guard

A cleanup PR must never delete or replace any of these directories:

- `services/ml_service`
- `services/api_service`
- `services/frontend`

A change that intentionally alters these boundaries is an architecture migration, not cleanup, and must be reviewed as such.
