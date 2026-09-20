# AI Surveillance

Clean production-oriented branch for the validated six-camera DeepStream 9.1
pipeline.

## Current validated pipeline

```text
6 x RTSP
  -> nvurisrcbin / NVDEC / NVMM
  -> nvstreammux (batch=6, live, sync-inputs=false)
  -> nvinfer YOLO26m raw one-to-many / TensorRT FP16
  -> DeepStream NMS (confidence=0.25, IoU=0.45)
  -> nvtracker / NvDCF_stable_person (960x544, ReID disabled)
  -> nvmultistreamtiler (3x2)
  -> nvdsosd
  -> NVENC
  -> MKV + live ffplay preview
```

Validated platform:

- NVIDIA RTX 3060 12 GB
- NVIDIA driver 595.91.07
- DeepStream 9.1.0
- CUDA runtime 13.2
- TensorRT 10.16.1.11
- RTSP TCP, 100 ms source latency
- pinned DeepStream container from `config/deepstream-platform.json`

## Passed gates

- six-camera camera-only 660-second soak: PASS
- six-camera camera-only source isolation/recovery: PASS
- YOLO26m 60-second detector gate: PASS
- YOLO26m 660-second detector soak: PASS
- YOLO26m detector source isolation/recovery: PASS
- NvDCF 60-second tracker gate: PASS
- NvDCF 660-second tracker soak: PASS
- process-specific GPU memory stability: PASS
- detector duplicate-box / NMS validation: PASS
- visual tracker ID-stability review: PASS

Current NvDCF profile:

```text
config/deepstream/config_tracker_NvDCF_stable_person.yml
```

ReID remains disabled in the current tracker stage.



## Clean-tree runtime regression — PASS

After removing 520 legacy/experimental files (609 tracked files reduced to 89),
the retained production-oriented tree passed both static/unit checks and a real
60-second six-camera YOLO26m + NvDCF smoke run.

Smoke evidence:

- all six RTSP sources remained near 20 FPS;
- no >1000 ms source-gap events or correlated gap clusters;
- YOLO26m inference/parser errors remained zero;
- detector NMS overlap validation passed;
- NvDCF reported zero untracked objects and zero duplicate per-frame IDs;
- process-specific GPU memory was flat at 2602 MiB after warmup;
- the separate visual review passed for advancing tiles, aligned boxes, stable
  IDs and absence of obvious ID switches or duplicate boxes.

This confirms that the cleanup did not remove a runtime dependency required by
the validated detector/tracker pipeline.

## Current next gate

Run source isolation/recovery with YOLO26m + NvDCF enabled for CAM-01 through
CAM-06. The interrupted source may receive a new local tracker ID after reset;
healthy peer streams must continue advancing and tracking.

Example:

```bash
python3 scripts/validate_yolo26m_tracker.py \
  --duration 90 \
  --interrupt-camera CAM-01 \
  --interrupt-at 20 \
  --interrupt-seconds 12 \
  --out .runtime/nvdcf-isolation-cam01

python3 scripts/yolo26m_tracker/check_isolation.py \
  .runtime/nvdcf-isolation-cam01
```

Do not begin cross-camera ReID until all six tracker isolation/recovery tests
pass.

## Active source tree

Core launchers:

- `scripts/validate_cam_six.py`
- `scripts/validate_yolo26m_person.py`
- `scripts/validate_yolo26m_tracker.py`

Native DeepStream graph:

- `scripts/cam_six_validation/main.cpp`
- `scripts/yolo26m_person/detection.hpp`
- `scripts/yolo26m_tracker/tracker.hpp`

Detector:

- `scripts/yolo26m_person/export_raw.py`
- `scripts/yolo26m_person/build.py`
- `scripts/yolo26m_person/parser_raw.cpp`
- `scripts/yolo26m_person/runtime.py`
- `scripts/yolo26m_person/check_gate.py`
- `scripts/yolo26m_person/check_isolation.py`

Tracker:

- `scripts/yolo26m_tracker/check_gate.py`
- `scripts/yolo26m_tracker/check_isolation.py`
- `scripts/yolo26m_tracker/mark_visual_review.py`

Active configs:

- `config/cameras.yaml`
- `config/deepstream-platform.json`
- `config/deepstream/config_infer_primary_yolo26m_raw_otm.txt`
- `config/deepstream/config_tracker_NvDCF_stable_person.yml`

Future project layers retained intentionally:

- `config/reid.yaml` and `config/reid/`
- `services/api_service/`
- `services/ml_service/`
- `services/frontend/`

Runtime-generated engines, ONNX files, recordings, logs and validation evidence
belong under `.runtime/` and are not committed.

## Validation commands

Static/unit checks:

```bash
python3 -m unittest \
  tests.test_deepstream91_preflight \
  tests.test_cam_six_stability \
  tests.test_yolo26m_gate \
  tests.test_yolo26m_tracker_gate -v
```

Camera probe:

```bash
python3 scripts/probe_cameras.py
```

60-second current tracker smoke test:

```bash
python3 scripts/validate_yolo26m_tracker.py \
  --duration 60 \
  --out .runtime/clean-tree-tracker-smoke

python3 scripts/yolo26m_tracker/check_gate.py \
  .runtime/clean-tree-tracker-smoke
```

The first tracker checker run will remain blocked until a separate visual review
is recorded for that exact recording.

## Documentation

- `docs/DEEPSTREAM91_MIGRATION_AUDIT.md`
- `docs/CAM01_CAM02_CAM03_CAM04_CAM05_CAM06_DEEPSTREAM91_VALIDATION.md`
- `docs/YOLO26M_DEEPSTREAM91_PERSON.md`
- `docs/NVDCF_TRACKER_STAGE.md`
