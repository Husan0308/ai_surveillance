# Current Project Architecture

This document describes the repository's current runtime architecture only.

## 1. Important project tree

```text
ai_surveillance/
├── main.py                         # production entry point and Qt lifecycle
├── ui.py                           # PySide6 widgets, pages, window, paint code
├── config/
│   ├── project.yaml                # AI, DB, storage, UI, DeepStream settings
│   └── cameras.yaml                # six RTSP camera definitions
├── backend/
│   ├── core/
│   │   ├── service_manager.py      # composition root and lifecycle owner
│   │   ├── batch_scheduler.py      # strict six-camera detector batching
│   │   ├── config.py               # YAML configuration access
│   │   ├── event_bus.py            # QObject signal event bus
│   │   └── performance_monitor.py  # process/camera/worker metrics
│   ├── cameras/
│   │   ├── camera_manager.py       # camera configuration and workers
│   │   ├── camera_worker.py        # RTSP/GStreamer/NVDEC acquisition
│   │   ├── frame_buffer.py         # latest-frame buffer
│   │   └── camera_health.py        # source health metrics
│   ├── ai/
│   │   ├── model_manager.py        # model construction/warm-up/shutdown
│   │   ├── detector.py             # batched YOLO person detector
│   │   ├── ai_worker.py            # per-camera post-detection pipeline
│   │   ├── tracker.py              # ByteTracker/Track
│   │   ├── reid_engine.py          # asynchronous hybrid ReID facade
│   │   ├── deep_reid.py            # OSNet implementation
│   │   ├── reid_gallery.py         # shared cross-camera gallery
│   │   ├── secondary_scheduler.py  # bounded asynchronous Pose/Face scheduler
│   │   ├── pose_engine.py          # YOLO pose model
│   │   ├── pose_service.py         # track-crop pose scheduling/fusion
│   │   ├── face_engine.py          # InsightFace detection/recognition
│   │   └── face_service.py         # track-crop face scheduling/fusion
│   ├── features/
│   │   ├── identity_manager.py     # per-camera person/visit/heatmap state
│   │   ├── events_service.py       # event normalization, persistence, signals
│   │   ├── analytics_service.py    # occupancy/system analytics
│   │   ├── person_service.py       # person records and face gallery changes
│   │   ├── enrollment.py           # enrollment workflow
│   │   └── settings_service.py     # persistent runtime settings
│   ├── bridge/
│   │   ├── system_bridge.py        # backend-to-ui QObject adapters
│   │   ├── ui_patches.py           # runtime bindings from ui.py to services
│   │   └── widgets.py              # backend-aware dialogs/widgets
│   ├── db/
│   │   ├── database.py             # SQLite schema and synchronized access
│   │   └── db_writer.py            # asynchronous DB write queue
│   └── storage/
│       ├── recording_service.py    # queued video writer
│       ├── cleanup_service.py      # retention timer
│       └── export_service.py       # data export
├── models/                         # model weights (runtime files)
└── data/surveillance.db            # SQLite database
```

## 2. Startup flow

1. `main.py:main()` sets GPU environment, creates one `QApplication`, applies theme/style, and runs `ui.SplashScreen`.
2. `ServiceManager.__init__()` creates configuration, database, model, camera, AI, feature, storage, and monitoring services in the same process.
3. `ServiceManager.start()` warms models, starts `SecondaryAIScheduler`, loads six cameras, creates one `AIWorker`/`ByteTracker` per online camera, and starts `BatchScheduler`.
4. `build_real_system()` creates `RealSystem`, `RealCameraSim` adapters, and signal connections.
5. `apply_ui_patches()` replaces `ui.System` with a factory returning that `RealSystem` and patches selected UI callbacks to call backend services.
6. `ui.MainWindow()` builds the pages and gets the already-created `RealSystem` through the patched `System()` symbol.
7. `QApplication.exec()` runs the main Qt event loop. `aboutToQuit` invokes `ServiceManager.shutdown()`.

## 3. Frontend

- `ui.py` contains the PySide6 frontend: `MainWindow`, `LivePage`, `CameraCard`, `VideoSurface`, `RightPanel`, `EventsPage`, `PersonManagementPage`, `EnrollmentPage`, `SettingsPage`, dialogs, and visual helpers.
- `MainWindow` owns a `QStackedWidget` with live, people, enrollment, events, and settings pages; dashboard and analytics pages are currently not added.
- `backend/bridge/system_bridge.py:RealSystem` implements the data contract expected by `ui.py`.
- `RealCameraSim.update_from_ai_result()` receives `AIWorker.result_ready`, converts `AIResult.frame` from BGR `numpy.ndarray` to a copied RGB `QImage` through `bgr_to_qimage()`, and maps `DetectedPerson` objects to `RealPersonUI`.
- `VideoSurface.paintEvent()` paints `RealCameraSim.frame`, then draws boxes from `RealCameraSim.people`; it does not read cameras or invoke inference.
- `RealSystem._connect_signals()` receives events, analytics, person changes, and settings changes from backend QObjects. `RealSystem.new_event` feeds `MainWindow.on_event()`.
- Direct backend access occurs through `RealSystem.sm` and runtime patches: camera start/stop, heatmaps, snapshots, enrollment, people, events, settings, recording, and camera health.
- `ui.py` directly imports no HTTP/API client. `system_bridge.py` imports `ui.make_avatar`; `ui_patches.py` receives the imported `ui` module and patches it.
- Main UI timers: `MainWindow` 40 ms render/update tick and 1000 ms stats tick; `CameraCard` 600 ms recording blink; `PulsingDot` animation timer; enrollment capture timer; splash timer; person-presence timer. All are Qt event-loop timers.

## 4. Main components

| Component | File | Thread/Process | Input | Output | Called By |
|---|---|---|---|---|---|
| `main()` | `main.py` | Main process / Qt GUI thread | CLI, config | Running application | Python entry point |
| `ServiceManager` | `backend/core/service_manager.py` | Main Qt thread | Configuration | Wired services/workers | `main()` |
| `ConfigService` | `backend/core/config.py` | Caller thread | YAML files | Dotted configuration values | `ServiceManager`, services |
| `CameraManager` | `backend/cameras/camera_manager.py` | Main Qt thread | Camera config/DB | `CameraWorker` objects/signals | `ServiceManager` |
| `CameraWorker` | `backend/cameras/camera_worker.py` | One `QThread` per camera | RTSP stream | BGR frames, health signals | `CameraManager` |
| `GstVideoCapture` | `backend/cameras/camera_worker.py` | Camera thread; GStreamer internal threads | Pipeline description | Latest mapped BGR frame | `CameraWorker` |
| `FrameBuffer` | `backend/cameras/frame_buffer.py` | Shared, condition-protected | BGR frame | Latest frame/id/time | `CameraWorker`, non-batch fallback |
| `BatchScheduler` | `backend/core/batch_scheduler.py` | One `QThread` + dispatch thread | Six per-camera FIFO streams | Six detector results/callbacks | Camera callbacks / `ServiceManager` |
| `Detector` | `backend/ai/detector.py` | Batch scheduler thread | Six BGR frames | Person boxes/confidences | `BatchScheduler.run()` |
| `AIWorker` | `backend/ai/ai_worker.py` | One `QThread` per camera | Batched detections + source frame | `AIResult`, AI events | `BatchScheduler` callback |
| `ByteTracker` / `Track` | `backend/ai/tracker.py` | Owning `AIWorker` thread | Person detections/ReID embedding | Active stable tracks | `AIWorker._run_after_detection()` |
| `HybridReIDEngine` | `backend/ai/reid_engine.py` | `OSNetReIDBatcher` thread | Selected body crops | Embeddings via callback | `AIWorker` |
| `SecondaryAIScheduler` | `backend/ai/secondary_scheduler.py` | One daemon Python thread | Pose/face tasks | Async callback results | `PoseService`, `FaceService` |
| `PoseService` | `backend/ai/pose_service.py` | Submitter + secondary thread | Selected track crops | Pose state/keypoints/ankle | `AIWorker._queue_secondary_tasks()` |
| `FaceService` | `backend/ai/face_service.py` | Submitter + secondary thread | Quality-gated upper-body crops | Identity/embedding | `AIWorker._queue_secondary_tasks()` |
| `IdentityManager` | `backend/features/identity_manager.py` | Main Qt thread via signal delivery | `AIResult` | Camera/person/visit/heatmap state | `AIWorker.result_ready` |
| `EventsService` | `backend/features/events_service.py` | Main Qt thread | AI/status/enrollment events | Memory event, DB task, Qt signal | Workers/services/EventBus |
| `DBWriter` | `backend/db/db_writer.py` | One `QThread` | Named DB operations | SQLite writes/result signals | Feature services |
| `Database` | `backend/db/database.py` | Shared, lock-protected | Queries/write methods | SQLite rows | Services / `DBWriter` |
| `RecordingService` | `backend/storage/recording_service.py` | One daemon writer thread | Camera BGR frames | Segmented MP4 files | `CameraWorker.frame_bgr_ready` |
| `RealSystem` | `backend/bridge/system_bridge.py` | Main Qt thread | Backend signals/services | UI-facing state/signals | `build_real_system()` |
| `RealCameraSim` | `backend/bridge/system_bridge.py` | Main Qt thread | `AIResult`, health | `QImage`, UI person list | `RealSystem` |
| `MainWindow` / `VideoSurface` | `ui.py` | Main Qt thread | `RealSystem` state | Painted UI | `main()` / Qt |

## 5. Complete camera frame flow

`cameras.yaml RTSP URL`
→ `CameraManager.add_camera()`
→ `CameraWorker.run()`
→ `GstVideoCapture.read()`
→ `rtspsrc` + RTP depayloader + `nvv4l2decoder`
→ `nvvideoconvert` to configured size
→ `videoconvert` BGR + dropping `appsink`
→ `FrameBuffer.put()` and `CameraWorker.ai_frame_callback`
→ `BatchScheduler.submit()` per-camera FIFO
→ `_prepare_ready_locked()` selects one frame from each of six cameras
→ `BatchScheduler.run()` calls `Detector.detect_batch(frames)` once
→ result dispatch calls each `AIWorker.enqueue_batch_result()`
→ `AIWorker._run_batch_results()`
→ `_run_after_detection()`
→ `ByteTracker.update()`
→ event-driven asynchronous ReID/Pose/Face submission
→ identity cache and `DetectedPerson` construction
→ `AIWorker.result_ready`
→ `IdentityManager.process_result()` and `RealCameraSim.update_from_ai_result()`
→ events/visits/analytics/heatmap state and `QImage`
→ `VideoSurface.paintEvent()`.

## 6. Camera and batch system

- Six enabled cameras are defined as `CAM-01` through `CAM-06`; all configured output sizes are `800x448`.
- `CameraWorker` has one `FrameBuffer`, which stores only the latest frame and uses `threading.Condition` for `wait_for_new()`.
- The detector path bypasses polling that buffer: `set_ai_frame_callback()` directly submits each captured frame to `BatchScheduler`.
- `BatchScheduler.batch_size` is capped at 6. It registers camera IDs in insertion order and uses the first six.
- Each camera has a `deque`; configured hard capacity is 5. `fresh_queue_target` is 2, so `submit()` drops oldest queued items before normal occupancy exceeds two; the hard five-item check remains.
- Additional drops occur for duplicate/non-increasing frame IDs, age above `max_frame_age_ms=120`, and replacement by a fresher queued frame at GPU launch.
- `_ready_batch` is a single prepared next-batch buffer. Producers call `_prepare_ready_locked()` while the current inference runs.
- `_cond` wakes the scheduler when data first arrives or a full batch becomes ready; maintenance waits are for expiry/debug deadlines.
- The scheduler performs exactly one `Detector.detect_batch()` call for each complete six-camera batch. No `nvstreammux` element is used in the current Python batching path.
- Detector-result callbacks run on `batch-result-dispatch`; per-camera post/tracker queues are condition-driven and bounded to 5.

## 7. AI models

| Model | File/Class | Current model/config | Device | Input | Enabled | Execution / call site |
|---|---|---|---|---|---|---|
| Person YOLO | `backend/ai/detector.py:Detector` | `models/yolo26n.pt`, class 0 only | `auto` → CUDA when available | `448x800`, fixed runtime batch 6 | Yes | Synchronous in `BatchScheduler.run()`; shared lock |
| Tracker | `backend/ai/tracker.py:ByteTracker` | ByteTrack-style IoU/motion + optional ReID | CPU | Person boxes | Yes | Synchronous per camera in `AIWorker._run_after_detection()` |
| Person ReID | `backend/ai/reid_engine.py:HybridReIDEngine`, `deep_reid.py` | `osnet_x0_25`, Market1501 checkpoint; HSV fallback | CPU | Body crops; deep model uses `256x128` | Yes | Event-driven async, queue 96, batch up to 6, 5 ms gather |
| Face | `backend/ai/face_engine.py:FaceEngine` | InsightFace `buffalo_l` | CPU / ONNX Runtime | detector `320x320`; upper-body crop | Yes | Async through `FaceService` and `SecondaryAIScheduler`; quality/cooldown gated |
| Pose | `backend/ai/pose_engine.py:PoseEngine` | `models/yolo26n-pose.pt` | CPU | Track crop, `imgsz=320`, batch up to 6 | Yes | Async through `PoseService` and `SecondaryAIScheduler` |

`ModelManager` constructs all five logical stages, warms detector/Pose/Face in `ServiceManager.start()`, and shuts optional engines down.

## 8. Video and memory flow

- Native RTSP source resolutions are not queried or stored by the hardware-decode path: **UNKNOWN**.
- Configured camera output/UI frame size is `800x448` for all six cameras.
- Hardware path: compressed RTSP → depay → `nvv4l2decoder` (NVDEC/NVMM) → `nvvideoconvert` → BGRx raw → CPU `videoconvert` to BGR → `appsink`.
- The appsink is `sync=false max-buffers=1 drop=true`; `GstVideoCapture.read()` also drains pending samples and keeps the newest.
- Mapping `Gst.Buffer` and `np.ascontiguousarray()` creates a CPU BGR array. NVMM/zero-copy does not continue past the appsink boundary.
- `CameraWorker` resizes with `cv2.resize()` only if the received frame differs from `target_size`.
- `Detector` receives six CPU BGR `800x448` arrays. Its direct path packs BGR→RGB/CHW into reusable pinned host memory, copies to CUDA, and normalizes; detector input shape is `448x800`.
- Detector boxes are copied back to CPU. Tracker, identity state, events, and UI overlays operate on CPU data.
- `AIResult.frame` is the same configured-size CPU BGR frame. `bgr_to_qimage()` performs BGR→RGB and `QImage.copy()`; UI paints that `800x448` image scaled to widget size.
- Pose/Face/ReID copy selected CPU crops into their bounded asynchronous queues.

## 9. Threads, loops, queues, callbacks, and synchronization

- One OS process contains the GUI and backend; there is no service subprocess.
- Main Qt thread: UI, `ServiceManager`, service QObjects, signal receivers, and all QTimers.
- Six `CameraWorker.run()` loops: open/read/reconnect/health; each has source-rate sleep and reconnect wait.
- One `BatchScheduler.run()` loop protected by `_cond`; one `batch-result-dispatch` callback thread using `queue.SimpleQueue`.
- Six `AIWorker` QThreads; in batch mode each runs `_run_batch_results()` and waits on `_batch_result_cond`.
- One `OSNetReIDBatcher` loop using `queue.Queue(maxsize=96)`; callbacks apply embeddings to trackers/galleries.
- One `secondary-ai-scheduler` loop using `threading.Condition`, bounded list queue 36, task deduplication, priority, expiry, and Pose batching.
- One `DBWriter.run()` loop using `queue.Queue(maxsize=10000)`.
- One recording loop using `queue.Queue(maxsize=300)`; recording is globally disabled in current config.
- Important locks: detector/model locks; `ByteTracker.lock`; database lock; frame-buffer condition; scheduler condition; AI result condition; identity/cache/gallery/unknown-registry locks; face/pose service locks.
- Important QTimers: `IdentityManager` 1 s sampling/5 s DB flush; `AnalyticsService` 1 s; `PerformanceMonitor` 1 s; `EventsService` and `CleanupService` hourly; enrollment detect/countdown timers; UI timers listed in section 3.

## 10. State, events, and database

- `AIWorker.IdentityCache` stores per-track person ID, name, embedding, confidence, source, and last-seen frame. `_pose_state` stores asynchronous Pose results.
- `ByteTracker.tracks` owns active/lost `Track` state, misses, velocities, confirmation, cached ReID embedding, and recall/debug counters.
- `IdentityManager.states[camera_id]` contains `CameraIdentityState`: active tracks, occupancy, known/unknown totals, recognition totals, visits, and interval counters.
- `IdentityManager.process_result()` updates heatmaps, opens/closes visits through `DBWriter`, emits `persons_online`, `metrics_updated`, and `identity_updated`.
- `AIWorker.event_detected` is connected to `EventsService.publish_event()`. Camera status and enrollment also produce events.
- `EventsService` normalizes events, applies cooldowns, queues `Database.add_event`, stores a bounded in-memory list, and emits `event_added`.
- `RealSystem._on_event_added()` converts the event to UI format and emits `new_event`; `MainWindow.on_event()` updates the right panel, event page, and notifications.
- SQLite tables include persons, face embeddings, unknown faces, events, visits, hourly analytics, camera configuration, and settings. `Database` uses `check_same_thread=False` plus a lock; writes normally go through `DBWriter`.

## 11. Frontend/backend relationship

- Frontend and backend run in the **same Python process**.
- They are **directly imported and instantiated**, not separate deployable services.
- Communication uses direct method calls for commands/queries and PySide6 signals for frames/results, events, status, analytics, and person changes.
- There is no HTTP, WebSocket, RPC, message broker, or inter-process IPC boundary.
- `ui.py` initially defines simulated `System`/camera classes, but production startup replaces `ui.System` with the `RealSystem` factory through `apply_ui_patches()` before `MainWindow` is created.

## 12. Configuration and current runtime settings

- `config/project.yaml`: central settings for AI, tracker, ReID, Pose, Face, batching, DeepStream, DB, events, storage, heatmap, security, and performance.
- `config/cameras.yaml`: six RTSP definitions, codec, configured size/FPS, latency, online/AI/recording flags, and reconnect limits.
- Current detector: YOLO26n, person only, confidence `0.05`, IoU `0.45`, `max_det=50`, `448x800`, batch 6.
- Current queues: detector per camera hard 5/fresh target 2; AI post-result 5; secondary 36; ReID 96; DB 10000; recording 300.
- Current decode: DeepStream path enabled, maximum 16 decode streams, OpenCV fallback disabled, per-camera appsink buffering 1.
- Current storage: SQLite `data/surveillance.db`; recordings disabled; event retention 30 days; file retention 14 days.

## 13. Current architecture diagram

```text
                         ONE PYTHON PROCESS
┌───────────────────────────────────────────────────────────────────────┐
│ Qt GUI thread                                                        │
│ main() → ServiceManager → RealSystem → MainWindow/VideoSurface       │
│                 ↑ signals: AIResult, events, health, analytics       │
│                 │ direct calls: settings/camera/snapshot/people      │
│                                                                       │
│  6× RTSP                                                              │
│    ↓                                                                  │
│  6× CameraWorker QThread                                              │
│  rtspsrc → depay → NVDEC/NVMM → convert/resize → CPU BGR appsink     │
│    ├─→ latest FrameBuffer                                             │
│    ├─→ RecordingService queue                                         │
│    └─→ BatchScheduler per-camera FIFO (hard 5, fresh target 2)        │
│             ↓ one fresh frame × six                                  │
│        READY batch 6 → YOLO26n CUDA → result dispatch                │
│             ↓                                                         │
│        6× AIWorker QThread → ByteTracker                              │
│             ├─→ async OSNet ReID batcher                             │
│             ├─→ SecondaryAIScheduler → Pose CPU / Face CPU           │
│             ├─→ IdentityManager → heatmap/visits/analytics           │
│             ├─→ EventsService → DBWriter QThread → SQLite            │
│             └─→ RealCameraSim → BGR-to-QImage → VideoSurface paint   │
└───────────────────────────────────────────────────────────────────────┘
```
