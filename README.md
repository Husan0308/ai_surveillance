# AI Surveillance System

A comprehensive AI-powered video surveillance platform with real-time person detection, face recognition, cross-camera Re-ID (Re-Identification), pose estimation, multi-camera tracking, alerting, analytics, and a rich PySide6-based UI.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Module Reference](#module-reference)
- [Data Flow](#data-flow)
- [Configuration Reference](#configuration-reference)
- [Database Schema](#database-schema)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Overview

The AI Surveillance System is a desktop application (PySide6/Qt) that turns one or more cameras (RTSP/IP/webcam) into an intelligent monitoring platform. It performs:

- Real-time object detection (YOLO / RTMPose)
- Face detection & 128D recognition (InsightFace / ArcFace embeddings)
- **Cross-camera Person Re-Identification (Re-ID)** with OSNet Market-1501
- **Unknown person tracking** with persistent unknown_id across cameras
- Pose estimation (fall / posture detection)
- Multi-object tracking (ByteTrack-style)
- Event logging with **camera-based deduplication**
- Alerts, snapshots, and video recording
- Analytics dashboards (heatmaps, counts, dwell time, stay_total)
- Enrollment of known persons with face galleries
- **Online status tracking** (person management)

The system is modular, config-driven (`config/*.yaml`), and uses a service-manager pattern for lifecycle control.

---

## Key Features

| Category | Capabilities |
|----------|--------------|
| **Detection** | YOLO-based person/object detection, configurable confidence thresholds |
| **Face Recognition** | ArcFace 128D embeddings, cosine-similarity matching, gallery enrollment |
| **Cross-camera Re-ID** | OSNet Market-1501 deep features, 2-pass matching (bir odam = bir track), SharedReIDGallery |
| **Unknown Tracking** | Unknown persons get persistent `unknown_id`, cross-camera matching via unknown_gallery |
| **Tracking** | Multi-object tracking with track IDs, lost-track recovery, zona-based tracking |
| **Pose** | Keypoint estimation, fall/posture detection |
| **Cameras** | Multi-camera RTSP/IP/webcam, health monitoring, auto-reconnect, frame buffering |
| **Events** | Entry/exit, appearance, loitering, fall, unknown-person events with **camera-based dedup** |
| **Alerts** | Configurable alert rules, sound alerts (`shutter.wav`) |
| **Storage** | Snapshots, video recordings (MP4), CSV/JSON exports, auto-cleanup |
| **Analytics** | Heatmaps, people counts, dwell time, room occupancy, **stay_total per person** |
| **UI** | PySide6 dashboard, live grids, event log, enrollment wizard, settings, **online status** |
| **Identity** | Person pool, global identity cache, unknown registry, room manager, **cross-camera identity** |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        UI Layer (PySide6)                     │
│   ui.py  ◄──►  backend/bridge/ (widgets, patches, sound)     │
└───────────────────────────┬──────────────────────────────────┘
                            │  Qt signals / event bus
┌───────────────────────────▼──────────────────────────────────┐
│                     Core Services Layer                        │
│  service_manager · event_bus · global_registry · logger       │
│  config · performance_monitor · system_monitor · person_pool  │
└───────────────────────────┬──────────────────────────────────┘
                            │
   ┌────────────────────────┼────────────────────────┐
   ▼                        ▼                        ▼
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Camera Layer│     │   AI Engine     │     │  Features Layer  │
│ manager,     │     │ detector, face, │     │ events, alerts,  │
│ worker,      │────▶│ reid, tracker,  │────▶│ analytics, enroll│
│ health, buf  │     │ pose, identity  │     │ heatmap, rooms   │
└─────────────┘     └────────┬────────┘     └────────┬─────────┘
                             │                        │
                    ┌────────▼────────┐               │
                    │  ReID Gallery    │               │
                    │ SharedReIDGallery │               │
                    │  (cross-camera)  │               │
                    └────────┬────────┘               │
                             │                        │
            ┌────────────────▼────────────────────────▼──────────┐
            │              Storage / DB Layer                     │
            │  database · db_writer · snapshots                  │
            │  recordings · exports · cleanup                     │
            │  visits (stay_total) · unknown_faces               │
            └────────────────────────────────────────────────────┘
```

**Design patterns used:**
- **Service Manager** — central lifecycle control for all services
- **Event Bus** — decoupled pub/sub between AI, cameras, features, UI
- **Global Registry** — singleton-style access to shared services
- **Worker threads** — `QThread`-based camera and AI workers
- **Signal/Slot** — Qt signals bridge backend → UI
- **2-pass ReID matching** — bir odam = bir track (eng yuqori score)
- **Camera-based event dedup** — bir kamera + bir person = 30s ichida bir event

---

## Project Structure

```
ai_surveillance/
├── main.py                      # Application entry point
├── ui.py                        # Main PySide6 UI (dashboard)
├── requirements.txt             # Python dependencies
├── .gitignore
├── assets/
│   └── sounds/
│       └── shutter.wav         # Alert sound
├── config/
│   ├── project.yaml            # Global project settings
│   └── cameras.yaml            # Camera definitions
├── data/                       # Runtime data (DB, galleries)
├── exports/                    # CSV/JSON exports
├── recordings/                  # Video recordings
├── tools/
│   ├── check_backend.py        # Backend health checker
│   └── pack_project.py         # Project packager
└── backend/
    ├── ai/                     # AI engines
    │   ├── ai_worker.py       # Central AI processing worker
    │   ├── detector.py         # YOLO object detection
    │   ├── face_engine.py      # Face detection + ArcFace recognition
    │   ├── reid_engine.py     # HybridReIDEngine (HSV + Deep)
    │   ├── deep_reid.py        # OSNet Market-1501 / ResNet50 ImageNet
    │   ├── reid_gallery.py    # SharedReIDGallery (cross-camera)
    │   ├── identity_cache.py  # Per-camera identity cache
    │   ├── tracker.py          # ByteTrack multi-object tracking
    │   └── pose_engine.py     # Pose estimation
    ├── bridge/                 # UI integration
    │   ├── system_bridge.py   # ServiceManager ↔ ui.py bridge
    │   ├── ui_patches.py       # Detection overlays
    │   ├── widgets.py          # Reusable PySide6 widgets
    │   └── sound_player.py     # Alert sound player
    ├── cameras/                # Camera management
    │   ├── camera_manager.py  # Multi-camera manager
    │   ├── camera_worker.py   # Per-camera capture thread
    │   ├── frame_buffer.py    # Thread-safe ring buffer
    │   ├── camera_health.py   # Health monitoring
    │   ├── connection_test.py # RTSP connectivity test
    │   └── utils.py            # Camera helpers
    ├── core/                   # Core services
    │   ├── config.py           # YAML config loader
    │   ├── service_manager.py # Lifecycle manager
    │   ├── global_registry.py  # Singleton registry
    │   ├── event_bus.py        # Pub/sub event system
    │   ├── logger.py           # Centralized logging
    │   ├── log_service.py      # Log viewer service
    │   ├── performance_monitor.py # FPS/latency tracking
    │   ├── system_monitor.py   # CPU/memory/GPU monitoring
    │   └── person_pool.py     # In-memory person pool
    ├── db/                     # Database & persistence
    │   ├── database.py         # SQLite connection & schema
    │   └── db_writer.py        # Async DB writer (QThread)
    ├── features/               # Business logic
    │   ├── events_service.py   # Event generation & persistence
    │   ├── alerts_service.py   # Alert rule evaluation
    │   ├── analytics_service.py # Metrics aggregation
    │   ├── enrollment.py       # Person enrollment wizard
    │   ├── person_service.py   # Person CRUD
    │   ├── identity_manager.py # Identity resolution (face + ReID)
    │   ├── heatmap.py          # Occupancy heatmaps
    │   ├── room_manager.py    # Zone/room management
    │   ├── settings_service.py # Runtime settings
    │   ├── unknown_service.py  # Unknown person management
    │   ├── unknown_registry.py # Unknown identity registry
    │   ├── global_identity_cache.py # Identity lookup cache
    │   └── snapshot_service.py # Feature-level snapshots
    └── storage/                # Media storage
        ├── snapshot_service.py # Frame snapshots
        ├── recording_service.py # Video recordings
        ├── export_service.py   # CSV/JSON exports
        └── cleanup_service.py  # Retention cleanup
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **UI** | PySide6 (Qt for Python) |
| **AI / CV** | OpenCV, NumPy, Ultralytics YOLO, InsightFace (ArcFace), ONNX |
| **Re-ID** | **OSNet Market-1501** (torchreid), ResNet50 ImageNet (fallback), HSV histogram (HybridReIDEngine) |
| **Pose** | Pose estimation models (RTMPose / YOLOv8-pose) |
| **Database** | SQLite (via `sqlite3`) |
| **Config** | YAML (`PyYAML`) |
| **Threading** | Python `threading`, `QThread`, `concurrent.futures` |
| **Media** | OpenCV VideoWriter (MP4), image snapshots |
| **Audio** | Qt Multimedia / QSound for alerts |

---

## Installation

### Prerequisites
- Python 3.9+
- A Linux/macOS/Windows environment with a display (for the UI)
- (Optional) GPU with CUDA for accelerated inference

### Steps

```bash
# 1. Clone the repository
git clone git@github.com:Husan0308/ai_surveillance.git
cd ai_surveillance

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install torchreid (for OSNet ReID)
pip install torchreid

# 5. (Optional) Verify backend health
python tools/check_backend.py
```

---

## Configuration

All configuration lives in `config/` as YAML files.

### `config/project.yaml`
Global project settings — paths, AI model thresholds, recording options, alert rules, analytics intervals, ReID settings, etc.

### `config/cameras.yaml`
Camera definitions — name, source (RTSP URL / webcam index), enabled flag, resolution, FPS, and per-camera overrides.

See the [Configuration Reference](#configuration-reference) section for details.

---

## Running the Application

```bash
# From the project root
python main.py
```

This launches the PySide6 UI, initializes the service manager, starts camera workers, loads AI models, and begins real-time processing.

### Health Check
```bash
python tools/check_backend.py
```

### Packaging
```bash
python tools/pack_project.py
```

---

## Module Reference

### Entry Points

#### `main.py`
- Application bootstrap.
- Initializes `QApplication`, loads `config/project.yaml` and `config/cameras.yaml`.
- Starts the `ServiceManager` (which starts all backend services).
- Constructs and shows the main `ui.py` window.
- Handles clean shutdown: stops services, releases cameras, closes DB.

#### `ui.py`
- Main PySide6 dashboard window.
- Live camera grid (multi-view), event log panel, analytics widgets.
- Enrollment wizard for adding known persons.
- Settings panel (runtime config editing).
- Person management panel with online status, stay_total, rec_count.
- Receives updates via Qt signals from `backend/bridge/`.

---

### AI Engine (`backend/ai/`)

The AI layer performs detection → tracking → face recognition → Re-ID → pose, producing structured detections and events.

#### `ai_worker.py` — `AIWorker`
- Central AI processing worker (runs in a `QThread`).
- Consumes frames from camera frame buffers.
- Pipeline: **detect → track → face recognize → reid match → pose → build persons**.
- Emits results via Qt signals (`result_ready`, `event_detected`).
- Coordinates `Detector`, `FaceEngine`, `ReIDEngine`, `Tracker`, `PoseEngine`, `IdentityManager`.
- Handles per-camera AI state and track lifecycles.
- Camera-based event deduplication (bir kamera + bir person = 30s ichida bir event).
- Zona-based tracking (`_recent_recognized`) — unknown box i tanilgan odam zonasiga tushsa, unknown bekor qilinadi.
- Unknown ReID tracking (`unknown_gallery`) — unknown persons get persistent unknown_id, cross-camera matching.

#### `detector.py` — `Detector`
- Wraps YOLO object detection.
- Loads model from path in config.
- Returns bounding boxes, classes, confidences.
- Configurable confidence & NMS thresholds, input size.

#### `face_engine.py` — `FaceEngine`
- Face detection + 128D ArcFace embedding extraction (InsightFace).
- Embedding comparison via cosine similarity.
- Maintains known-person gallery (name → embeddings).
- Returns recognized identity + confidence per face.

#### `reid_engine.py` — `HybridReIDEngine`
- Hybrid Re-ID combining HSV histogram + deep features.
- Extracts Re-ID feature vectors for person crops.
- Matches against `SharedReIDGallery` for cross-camera identity.
- Configurable similarity threshold.

#### `deep_reid.py` — `DeepReIDEngine`
- OSNet Market-1501 (torchreid) — primary ReID model.
- ResNet50 ImageNet — fallback model.
- Preprocessing (resize, normalize) and feature extraction.
- 512D feature vectors (OSNet) / 2048D (ResNet50).

#### `reid_gallery.py` — `SharedReIDGallery`
- Cross-camera Re-ID feature gallery (stores feature vectors).
- Add/query/remove identities.
- 2-pass matching — bir odam = bir track (eng yuqori score).
- Backed by disk (`data/`) for persistence across restarts.

#### `identity_cache.py` — `IdentityCache`
- Per-camera identity cache (track_id → person_id).
- Stores face/ReID/zone identity with timestamps.
- Automatic cleanup of stale tracks.

#### `tracker.py` — `Tracker`
- Multi-object tracking (ByteTrack-style assignment).
- Assigns persistent track IDs to detections.
- Handles track birth, update, and death (lost-track timeout).
- Provides track history for trajectory/analytics.

#### `pose_engine.py` — `PoseEngine`
- Keypoint pose estimation for persons.
- Detects falls / abnormal postures from keypoint geometry.
- Emits pose-based events (e.g., fall detected).

---

### Camera Layer (`backend/cameras/`)

#### `camera_manager.py` — `CameraManager`
- Manages all cameras defined in `config/cameras.yaml`.
- Creates and starts `CameraWorker` instances.
- Provides unified access to live frames and camera status.
- Handles add/remove/reconfigure of cameras at runtime.

#### `camera_worker.py` — `CameraWorker`
- Per-camera capture thread (`QThread`).
- Opens RTSP/IP/webcam stream via OpenCV.
- Writes frames into `FrameBuffer` (ring buffer).
- Auto-reconnect on stream failure.
- Emits frame-ready signals and health updates.

#### `frame_buffer.py` — `FrameBuffer`
- Thread-safe ring buffer for recent frames per camera.
- Decouples capture FPS from processing FPS.
- Provides latest-frame access for UI and AI.

#### `camera_health.py` — `CameraHealthMonitor`
- Monitors camera connectivity and FPS.
- Detects stale/dead streams.
- Triggers reconnect via `CameraManager`.

#### `connection_test.py`
- Utility to test RTSP/IP camera connectivity before adding.
- Reports latency and stream validity.

#### `utils.py`
- Camera helper functions (URL parsing, RTSP transport, frame conversion).

---

### Core Services (`backend/core/`)

#### `config.py` — `Config`
- Loads and merges `config/project.yaml` + `config/cameras.yaml`.
- Provides typed access to settings (paths, thresholds, intervals).
- Supports runtime reload and defaults.

#### `service_manager.py` — `ServiceManager`
- Central lifecycle manager for all backend services.
- `start()` / `stop()` / `restart()` for cameras, AI, features, storage.
- Ensures ordered startup and graceful shutdown.
- Holds references in `GlobalRegistry`.

#### `global_registry.py` — `GlobalRegistry`
- Singleton registry of shared service instances.
- Provides access to `Config`, `Database`, `EventBus`, `CameraManager`, `AIWorker`, etc.
- Avoids circular imports via lazy access.

#### `event_bus.py` — `EventBus`
- Decoupled publish/subscribe event system.
- Topics: `person_detected`, `face_recognized`, `reid_match`, `event_logged`, `alert_triggered`, `camera_status`, etc.
- Thread-safe; used by AI, features, storage, and UI bridge.

#### `logger.py` — `Logger`
- Centralized logging configuration.
- File + console handlers with rotation.
- Used across all modules.

#### `log_service.py` — `LogService`
- Application-level log viewer/persistence service.
- Stores recent log entries for UI display.

#### `performance_monitor.py` — `PerformanceMonitor`
- Tracks FPS, inference latency, queue depths.
- Reports performance metrics for UI dashboards.

#### `system_monitor.py` — `SystemMonitor`
- Monitors CPU, memory, GPU usage.
- Warns on resource exhaustion.

#### `person_pool.py` — `PersonPool`
- In-memory pool of currently tracked persons (across cameras).
- Holds current identity, last seen camera, track ID, timestamps.
- Source of truth for "who is where right now".

---

### Features Layer (`backend/features/`)

Business logic built on top of AI + core services.

#### `events_service.py` — `EventsService`
- Generates and persists surveillance events.
- Event types: entry, exit, appearance, disappearance, loitering, fall, unknown.
- Writes to DB and publishes on `EventBus`.

#### `alerts_service.py` — `AlertsService`
- Evaluates alert rules against events.
- Triggers sound alerts (`shutter.wav`), UI notifications.
- Configurable rules (per zone, per identity, per event type).

#### `analytics_service.py` — `AnalyticsService`
- Aggregates metrics: people count, dwell time, peak hours.
- Feeds analytics dashboard widgets.

#### `enrollment.py` — `EnrollmentService`
- Wizard logic for enrolling known persons.
- Captures face samples, computes embeddings, stores gallery.
- Links face identity to Re-ID gallery.

#### `person_service.py` — `PersonService`
- CRUD for known persons (name, metadata, embeddings).
- Persists to DB and face gallery.

#### `identity_manager.py` — `IdentityManager`
- Per-camera identity state (`CameraIdentityState`).
- Tracks active tracks, visit open/close, stay_total calculation.
- Heatmap generation (occupancy/trajectory).
- Emits `persons_online` signal for UI online status.
- Coordinates face + ReID + unknown registry.

#### `heatmap.py` — `HeatmapService`
- Generates occupancy/trajectory heatmaps from track positions.
- Renders overlay images for UI.

#### `room_manager.py` — `RoomManager`
- Defines zones/rooms in camera views.
- Tracks occupancy per room (entry/exit counting).

#### `settings_service.py` — `SettingsService`
- Runtime settings read/write (persisted to `config/`).
- Bridges UI settings panel to config files.

#### `unknown_service.py` — `UnknownService`
- Manages unknown persons detected.
- Auto-assigns IDs (`Unknown_001`, ...).
- Tracks repeat unknowns for potential enrollment.

#### `unknown_registry.py` — `UnknownRegistry`
- Persistent registry of unknown identities.
- Deduplicates unknowns via Re-ID features.

#### `global_identity_cache.py` — `GlobalIdentityCache`
- Caches identity lookups (face + Re-ID) for performance.
- Shared across cameras and AI workers.

#### `snapshot_service.py` — `SnapshotService` (features)
- Feature-level snapshot trigger (e.g., on event).
- Coordinates with `storage/snapshot_service.py`.

---

### Storage Layer (`backend/storage/`)

#### `snapshot_service.py` — `SnapshotService`
- Captures and saves frame snapshots (JPEG/PNG) on events.
- Organized by date/camera/event.
- Used for alerts and evidence.

#### `recording_service.py` — `RecordingService`
- Records video segments (MP4 via OpenCV VideoWriter).
- Triggered by events or manual record.
- Configurable retention and codec.

#### `export_service.py` — `ExportService`
- Exports events/logs to CSV/JSON (see `exports/`).
- Date-range and filter support.

#### `cleanup_service.py` — `CleanupService`
- Periodic cleanup of old snapshots/recordings.
- Enforces retention policy from config.

---

### Bridge Layer (UI Integration) (`backend/bridge/`)

The bridge connects backend services to the PySide6 UI without tight coupling.

#### `system_bridge.py` — `SystemBridge`
- Main bridge between `ServiceManager` and `ui.py`.
- Wires Qt signals from workers to UI slots.
- Translates backend events into UI updates.

#### `ui_patches.py`
- Dynamic UI patches/overlays (bounding boxes, labels, track IDs).
- Draws detection and identity overlays on live camera views.

#### `widgets.py`
- Reusable PySide6 widgets (camera view, event list, analytics charts).
- Custom Qt widgets used by `ui.py`.

#### `sound_player.py` — `SoundPlayer`
- Plays alert sounds (`assets/sounds/shutter.wav`).
- Non-blocking audio via Qt Multimedia.

---

### Database Layer (`backend/db/`)

#### `database.py` — `Database`
- SQLite connection management.
- Schema creation/migration.
- Provides query helpers.
- stay_total calculation via `close_visit_by_track`.

#### `db_writer.py` — `DBWriter`
- Writes events, persons, alerts, and metadata to SQLite.
- Batched writes for performance (QThread-based queue).
- Used by `EventsService`, `AlertsService`, etc.

---

### Tools (`tools/`)

#### `check_backend.py`
- Verifies backend health: config load, DB, model paths, camera connectivity.
- Run before launch to catch misconfiguration.

#### `pack_project.py`
- Packages the project (zip/archive) for distribution.
- Excludes runtime data/recordings per `.gitignore`.

---

## Data Flow

```
Camera (RTSP/webcam)
   │
   ▼
CameraWorker ──► FrameBuffer ──► AIWorker
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
       Detector                Tracker                 FaceEngine
     (YOLO bboxes)         (track IDs)             (ArcFace 128D)
            │                       │                       │
            └──────────► HybridReIDEngine ◄────────────────┘
                      (HSV + OSNet Market-1501)
                              │
                              ▼
                      SharedReIDGallery
                 (cross-camera 2-pass matching)
                              │
                              ▼
                      IdentityManager
                 (face + reid → person ID)
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      EventsService     AlertsService      AnalyticsService
      (camera-based      (sound, UI)       (stay_total, heatmap)
       dedup)
            │                 │                 │
            ▼                 ▼                 ▼
        DBWriter          SoundPlayer       Heatmap/UI
            │
            ▼
        EventBus ──► SystemBridge ──► ui.py (dashboard)
```

---

## Configuration Reference

### `config/project.yaml` (key fields)

| Section | Description |
|---------|-------------|
| `paths` | Data, recordings, exports, snapshots directories |
| `ai.detection` | Model path, confidence, NMS, input size |
| `ai.face` | ArcFace model, recognition threshold, gallery path |
| `ai.reid` | Re-ID model (OSNet Market-1501), feature dim (512), match threshold (0.70) |
| `ai.pose` | Pose model, fall-detection thresholds |
| `ai.tracker` | Track lifetime, match threshold, max lost frames |
| `recording` | Codec, FPS, segment length, retention days |
| `snapshots` | Format, quality, retention days |
| `alerts` | Rules (event type → action), sound on/off |
| `analytics` | Heatmap interval, dwell threshold, count window |
| `database` | SQLite path |
| `logging` | Level, file path, rotation |

### `config/cameras.yaml`

```yaml
cameras:
  - name: "Front Door"
    source: "rtsp://user:pass@192.168.1.10:554/stream"
    enabled: true
    width: 1280
    height: 720
    fps: 15
  - name: "Webcam"
    source: 0          # local webcam index
    enabled: false
```

| Field | Description |
|-------|-------------|
| `name` | Display name |
| `source` | RTSP URL or webcam integer index |
| `enabled` | Whether to start on launch |
| `width`/`height` | Capture resolution |
| `fps` | Target capture FPS |

---

## Database Schema

The SQLite database (`data/surveillance.db`) stores:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `persons` | Known persons | id, name, department, employee_id, status, avatar, created_at, updated_at, last_seen, rec_count, stay_total |
| `face_embeddings` | Face embeddings | id, person_id, embedding (BLOB), image (BLOB), quality, created_at |
| `unknown_faces` | Unknown person registry | id, track_id, camera_id, image, embedding, first_seen, last_seen, count, converted_person_id |
| `events` | Surveillance events | id, time, camera_id, person_id, person_name, type, level, confidence, snapshot_path, ack, extra |
| `visits` | Visit tracking (stay_total) | id, person_id, camera_id, track_id, entered_at, left_at, duration_sec |
| `analytics_hourly` | Hourly analytics | date, hour, camera_id, occupancy_sum, known_count, unknown_count, detection_count, recognition_count |
| `camera_config` | Camera configuration | id, name, location, source, username, password, online, ai_enabled, heatmap_enabled, recording_enabled |
| `settings` | Runtime key-value settings | key, value, updated_at |

Schema is created/migrated automatically by `database.py` on startup.

### stay_total Calculation

When a visit is closed (`close_visit_by_track`), the duration is calculated and added to the person's `stay_total`:

```sql
UPDATE persons SET stay_total = stay_total + ? WHERE id = ?
```

---

## Development

### Running Tests / Health Checks
```bash
python tools/check_backend.py
```

### Code Style
- Python modules under `backend/` are organized by domain (`ai/`, `cameras/`, `core/`, `db/`, `features/`, `storage/`, `bridge/`).
- Each package has an `__init__.py`.
- Services follow a singleton/registry pattern via `GlobalRegistry`.
- Inter-service communication uses `EventBus` (pub/sub) and Qt signals.

### Adding a New Camera
1. Edit `config/cameras.yaml` — add a new entry.
2. (Optional) Test with `python -c "from backend.cameras.connection_test import *; test('rtsp://...')"` or use `tools/check_backend.py`.
3. Restart the app, or use the UI settings panel to add at runtime.

### Enrolling a Known Person
1. Open the UI → Enrollment wizard.
2. Provide a name.
3. Capture face samples (the `EnrollmentService` computes and stores ArcFace embeddings).
4. The person is now recognized in live feeds and linked to Re-ID gallery.

### Adding a New AI Model
1. Place model weights in the path specified by `config/project.yaml`.
2. Update the model path / threshold in config.
3. Restart. The `Detector`/`FaceEngine`/`ReIDEngine`/`PoseEngine` load from config.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Camera won't connect** | Verify RTSP URL/credentials; run `tools/check_backend.py`; check `CameraHealthMonitor` logs. |
| **No detections** | Check `ai.detection.confidence` threshold in config; verify model path exists. |
| **Face not recognized** | Re-enroll with more samples; lower `ai.face.threshold`; check gallery path. |
| **ReID not working** | Verify torchreid installed; check OSNet model path; verify `ai.reid.threshold`. |
| **Low FPS** | Reduce input size in config; disable unused cameras; check `PerformanceMonitor`; use GPU. |
| **DB locked** | Ensure single instance running; `DBWriter` uses batched writes to minimize contention. |
| **No sound on alert** | Verify `assets/sounds/shutter.wav` exists; check `alerts.sound` enabled in config. |
| **Recordings missing** | Check `recording` retention; verify `recordings/` is writable; inspect `CleanupService`. |
| **High memory** | Reduce `FrameBuffer` size; check `PersonPool`/`GlobalIdentityCache` eviction. |
| **Events duplicated** | Camera-based dedup active (30s window); check `_emitted` dict in `ai_worker.py`. |
| **stay_total not updating** | Verify `visits` table exists; check `close_visit_by_track` in `database.py`. |

---

## License

This project is proprietary. See repository for details.

---

*Generated from source code analysis of the `ai_surveillance` codebase.*