# AI Surveillance — Stable Detection/UI Baseline

This branch intentionally reduces the runtime to the smallest production path that must work first:

```text
6 x RTSP cameras
  -> DeepStream / GStreamer capture
  -> LatestFrameStore per camera
  -> detector-only spawned YOLO26m CUDA process
  -> visual tracker
  -> JPEG publishers
  -> plain PySide6 dashboard
```

Optional analytics are kept in the repository but are disabled by default until the baseline passes real-machine acceptance:

- ReID: disabled
- YOLO26m-pose: disabled
- floor heatmap: disabled
- custom ROI second pass: disabled
- camera exclusion masks: disabled
- UI polish monkey-patch: opt-in
- Heatmap UI monkey-patch: opt-in

The backup branch `backup/local-43c0763` preserves the previous local experiment that embedded YOLO26 pose keypoints directly into the detector/tracker path. It is not merged into this baseline because it duplicates the newer pose path and can destabilize detection.

## Why this baseline exists

Detection must not depend on Pose, Heatmap, ReID, or UI extensions. A failure in any optional analytics feature must never prevent camera frames or YOLO person detections from being published.

The stable detector also fixes model startup behavior. If a configured project-local model file is missing, the worker falls back to a plain Ultralytics checkpoint name such as `yolo26m.pt` instead of failing before the CUDA process starts.

## Install

```bash
cd ~/ai_surveillance
source venv/bin/activate
pip install -r requirements/ml.txt
pip install -r requirements/frontend.txt
```

## 1. Run tests

Use unittest discovery so an unrelated installed Python package named `tests` cannot shadow this repository's `tests/` directory.

```bash
python -m unittest discover -s tests -p "test_stable_detector.py" -v
python -m unittest discover -s tests -p "test_core_v1_visual_tracker.py" -v
python -m unittest discover -s tests -p "test_core_v1_jpeg_publisher.py" -v
```

## 2. Run ML only

```bash
PYTHONFAULTHANDLER=1 python -m services.ml_service.core_v1.main
```

The detector may download `yolo26m.pt` on first run if no local checkpoint exists.

## 3. Verify camera and detector health

In a second terminal:

```bash
curl -s http://127.0.0.1:8001/health | python -m json.tool
```

Acceptance requirements:

- `online` should reach the expected camera count.
- `detector.ready` must become `true`.
- `detector.process_alive` must be `true`.
- `detector.cuda_topology` must be `detector_only_spawned_process`.
- `detector.pose_in_hot_path` must be `false`.
- `detector.last_error` must be empty.
- camera input counters must increase.

Then verify actual person results:

```bash
curl -s http://127.0.0.1:8001/detections | python -m json.tool
```

At least cameras currently seeing people should contain non-empty `boxes`.

## 4. Verify frames before opening the UI

```bash
curl -o /tmp/cam01.jpg http://127.0.0.1:8001/frame/CAM-01
file /tmp/cam01.jpg
```

Repeat for any camera that appears offline or stale.

## 5. Run the plain frontend

```bash
python -m services.frontend.core_v1.main
```

The baseline does not install UI monkey-patches by default.

Optional presentation polish can be tested later with:

```bash
AI_SURVEILLANCE_UI_POLISH=1 python -m services.frontend.core_v1.main
```

Heatmap UI is intentionally not part of baseline acceptance. It can only be tested later with:

```bash
AI_SURVEILLANCE_UI_HEATMAP=1 python -m services.frontend.core_v1.main
```

## Recovery order

Do not enable features out of order.

1. Six camera capture must be stable.
2. YOLO26m detection must be stable.
3. Visual tracker and overlays must be stable.
4. Plain UI must be stable.
5. Re-enable ReID and soak-test it.
6. Add YOLO26m-pose as a truly optional side path.
7. Add floor heatmap only after Pose and homography calibration are verified.
8. Re-enable UI polish/heatmap extensions last.

## Run commands

ML:

```bash
python -m services.ml_service.core_v1.main
```

API:

```bash
python -m services.api_service.core_v1.main
```

Frontend:

```bash
python -m services.frontend.core_v1.main
```

Machine-local models, databases, captures, `.runtime/`, logs and `cameras.local.yaml` must not be committed.
