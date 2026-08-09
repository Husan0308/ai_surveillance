# AI Surveillance

A six-camera local surveillance application with NVIDIA hardware decode, batched person detection, camera-local tracking, asynchronous biometric/appearance processing, persistent configuration, and a PySide6 operator UI.

## Current production pipeline

```text
RTSP × 6 → GStreamer/NVDEC → latest buffers → BatchScheduler
→ YOLO/PyTorch → custom_motion_iou → async OSNet ReID/InsightFace
→ GlobalIdentityManager → camera heatmaps/events
→ FastAPI WebSocket/MJPEG → PySide6
```

Production tracking is `custom_motion_iou`; ByteTrack is not used in production. DeepStream/GStreamer is used for hardware ingestion/NVDEC, not `nvinfer`. Pose is currently disabled. Heatmaps are per-camera occupancy maps, not a floorplan. Physical topology remains manual and unverified.

## Start

Use the repository virtual environment or set `SURVEILLANCE_PYTHON`:

```bash
scripts/check_environment.py
scripts/run_all.sh
```

Individual services:

```bash
python -m services.api_service.app
python -m services.ml_service.app
python -m services.frontend.main
```

Ctrl+C on `run_all.sh` forwards termination to all children and waits for clean shutdown.

## Configuration and credentials

- `config/project.yaml`: ML/runtime settings.
- `config/cameras.yaml`: secret-free camera bootstrap records.
- `config/cameras.local.yaml`: optional ignored local overrides.
- `config/topology.yaml`: manually confirmed physical topology.
- `data/surveillance.db`: canonical runtime camera/person/event storage.

Provide secrets through environment variables such as `SURVEILLANCE_RTSP_USERNAME` and `SURVEILLANCE_RTSP_PASSWORD`. Public API responses mask embedded RTSP credentials. Do not commit local camera overrides.

Topology must not be inferred from camera/channel numbers. Run:

```bash
python scripts/topology_calibration.py
```

Then fill `config/topology.yaml` only after physical inspection.

## Person enrollment

The same-host PySide6 client accepts 10–30 image files. It copies them to the controlled `data/enrollment_staging/` directory before calling the API. The API rejects paths outside staging and validates file type, size, regular-file status, and image decoding. ML then requires a usable single face and applies size, blur, pose/quality, and embedding checks. The best ten normalized embeddings are persisted with their InsightFace model version.

Path-based enrollment assumes a trusted same-host desktop. A future remote client must use an authenticated upload endpoint instead.

## Persistence and maintenance

SQLite uses WAL. Make a consistent online backup with:

```bash
python scripts/backup_database.py
```

Do not copy the live database and WAL independently while services are writing. Historical legacy events are preserved; future persistent writes use the canonical event taxonomy documented in [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md).

Unknown representative crops use bounded quality replacement under `data/snapshots/`. Raw crops are not included in frequent WebSocket messages.

## Validation baseline

The established six-camera baseline is approximately:

- Detector p50: 92–93 ms
- Fast path p50: 97 ms
- Capture-to-metadata p50: 161 ms
- Batch cadence: 9–10 batches/s

Use the test suite and a labelled face-calibration dataset before changing biometric thresholds. Do not claim biometric accuracy from synthetic embeddings or one enrolled person.
