# AI Surveillance — Full Features Safe Profile

The production hot path stays simple and authoritative:

```text
6 x RTSP cameras
  -> DeepStream / GStreamer capture
  -> LatestFrameStore per camera
  -> detector-only spawned YOLO26m CUDA process
  -> visual tracker
  -> JPEG publishers
  -> PySide6 dashboard
```

All analytics are enabled again, but they are isolated from detection:

```text
YOLO26m detections
  ├─> CPU ReID side path -> OSNet -> Global ID / room fusion
  └─> sparse CPU YOLO26m-pose side path -> ankle keypoints -> floor heatmap
```

The detector is still the only PyTorch CUDA analytics process. `yolo26m-pose.pt` intentionally runs sparsely on CPU so Pose cannot create a second CUDA context or gate person detection. ReID is also CPU-only. If either optional side path fails, camera frames and YOLO26m detections keep running.

UI polish and Heatmap UI are enabled by default, but their installers are fail-safe: an extension import/install error falls back to the plain dashboard instead of preventing startup.

Custom ROI second-pass detection and camera exclusion masks remain off because the current full-frame detector is the verified baseline. They are tuning features, not required system features.

## Install

```bash
cd ~/ai_surveillance
source venv/bin/activate
pip install -r requirements/ml.txt
pip install -r requirements/frontend.txt
```

## Tests

Use unittest discovery so an unrelated installed package named `tests` cannot shadow this repository's tests.

```bash
python -m unittest discover -s tests -p "test_stable_detector.py" -v
python -m unittest discover -s tests -p "test_core_v1_visual_tracker.py" -v
python -m unittest discover -s tests -p "test_core_v1_jpeg_publisher.py" -v
python -m unittest discover -s tests -p "test_core_v1_pose.py" -v
python -m unittest discover -s tests -p "test_floor_heatmap.py" -v
python -m unittest discover -s tests -p "test_full_feature_profile.py" -v
```

## Run ML

```bash
PYTHONFAULTHANDLER=1 python -m services.ml_service.core_v1.main
```

On first use, Ultralytics may download `yolo26m.pt` and `yolo26m-pose.pt`. ReID may also download its configured OSNet checkpoint if it is not present locally.

## Verify health

```bash
curl -s http://127.0.0.1:8001/health | python -m json.tool
```

Primary acceptance requirements:

- `status` is `ok`.
- `online` reaches the expected camera count.
- `detector.ready` is `true`.
- `detector.process_alive` is `true`.
- `detector.cuda_topology` is `detector_only_spawned_process`.
- `detector.pose_in_hot_path` is `false`.
- `detector.last_error` is empty.

Optional feature checks are non-gating:

- `reid.enabled` should be true and `reid.ready` should become true after its checkpoint is available.
- `pose.enabled` should be true, `pose.model` should be `yolo26m-pose.pt`, and `pose.device` should be `cpu`.
- `pose.isolation` should be `detector_independent_sidepath`.
- `heatmap.enabled` should be true. Heat samples require valid room homography calibration.

## Verify endpoints

```bash
curl -s http://127.0.0.1:8001/detections | python -m json.tool
curl -s http://127.0.0.1:8001/reid | python -m json.tool
curl -s http://127.0.0.1:8001/poses | python -m json.tool
curl -s http://127.0.0.1:8001/heatmap | python -m json.tool
curl -s http://127.0.0.1:8001/room-mapping | python -m json.tool
```

For Heatmap to accumulate real ankle samples, each camera used for room-floor projection needs a valid camera-to-floor homography in `config/room_mapping.yaml` or through the Room Map calibration UI.

## Frontend

```bash
python -m services.frontend.core_v1.main
```

UI polish and Heatmap UI are on by default. Either can be disabled for isolation without editing code:

```bash
AI_SURVEILLANCE_UI_POLISH=0 python -m services.frontend.core_v1.main
AI_SURVEILLANCE_UI_HEATMAP=0 python -m services.frontend.core_v1.main
```

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

Machine-local models, databases, captures, `.runtime/`, logs and `cameras.local.yaml` must not be committed. The backup branch `backup/local-43c0763` remains available for the earlier experimental detector-pose changes.
