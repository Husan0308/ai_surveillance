from pathlib import Path

from services.ml_service.app.detector import _compatible_arches

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_ml_has_one_latest_only_person_detector() -> None:
    detector = source("services/ml_service/app/detector.py")
    pipeline = source("services/ml_service/app/deepstream/pipeline.py")

    assert "class PersonDetector" in detector
    assert '"classes": [0]' in detector
    assert "LatestFrameStore" in detector
    assert "model.predict(source=frames" in detector
    assert 'mp.get_context("spawn")' in detector
    assert 'name="person-detector-cuda"' in detector
    assert "self.detector = PersonDetector(settings.detection, self.stores)" in pipeline
    assert "PersonDetector(" not in source("services/ml_service/app/camera_worker.py")


def test_cuda_cubin_minor_compatibility() -> None:
    # NVIDIA desktop binary compatibility permits sm_60 cubin on sm_61 GPU.
    assert _compatible_arches((6, 1), ("sm_50", "sm_60", "sm_70")) == ("sm_60",)
    assert _compatible_arches((7, 5), ("sm_70", "sm_75", "sm_80")) == ("sm_70", "sm_75")
    assert _compatible_arches((6, 0), ("sm_61", "sm_70")) == ()


def test_detection_stage_does_not_own_tracking_reid_or_face() -> None:
    detector = source("services/ml_service/app/detector.py")
    forbidden_imports = (
        "import insightface",
        "from insightface",
        "import fastreid",
        "from fastreid",
        "import torchreid",
        "from torchreid",
        "GlobalIdentity",
        "ByteTrack",
        "BYTETracker",
    )
    for text in forbidden_imports:
        assert text not in detector


def test_detection_config_is_gpu_batched_and_overlayed() -> None:
    config = source("config/cameras.yaml")
    runtime_config = source("services/ml_service/app/config.py")
    publisher = source("services/ml_service/app/jpeg_publisher.py")

    for text in (
        "detection:",
        "model: yolo26m.pt",
        "batch_size: 2",
        "width: 512",
        "height: 288",
        "target_fps_per_camera: 4.0",
    ):
        assert text in config
    assert "class DetectionConfig" in runtime_config
    assert "DetectionStore" in publisher
    assert 'text = f"Person {detection.confidence:.2f}"' in publisher


def test_ml_exposes_detection_health_and_results() -> None:
    main = source("services/ml_service/app/main.py")
    pipeline = source("services/ml_service/app/deepstream/pipeline.py")

    assert '"detector": current.detector_metrics()' in main
    assert '@app.get("/detections/{camera_id}")' in main
    assert 'runtime: Any | None = None' in main
    assert '"people": int(detection.get("people", 0))' in pipeline
