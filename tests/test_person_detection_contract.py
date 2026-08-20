from pathlib import Path

from services.ml_service.app.detector import _compatible_arches

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_ml_has_one_latest_only_person_detector() -> None:
    detector = source("services/ml_service/app/detector.py")
    legacy = source("services/ml_service/app/legacy_latest_detector.py")
    pipeline = source("services/ml_service/app/deepstream/pipeline.py")

    assert "class PersonDetector" in detector
    assert "class LegacyLatestPersonDetector" in legacy
    assert '"classes": [0]' in legacy
    assert "model.predict(source=frames" in legacy
    assert 'name="person-detector-cuda"' in legacy
    assert "legacy-latest-only-uncapped" in legacy
    assert "cv2.resize" in legacy
    assert '"source_w"' in legacy
    assert '"source_h"' in legacy
    assert "LegacyLatestPersonDetector(settings.detection, self.stores)" in pipeline
    assert "PersonDetector(" not in source("services/ml_service/app/camera_worker.py")


def test_cuda_cubin_minor_compatibility() -> None:
    assert _compatible_arches((6, 1), ("sm_50", "sm_60", "sm_70")) == ("sm_60",)
    assert _compatible_arches((7, 5), ("sm_70", "sm_75", "sm_80")) == ("sm_70", "sm_75")
    assert _compatible_arches((6, 0), ("sm_61", "sm_70")) == ()


def test_detection_stage_does_not_own_tracking_reid_or_face() -> None:
    detector = source("services/ml_service/app/detector.py")
    legacy = source("services/ml_service/app/legacy_latest_detector.py")
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
        assert text not in legacy


def test_detection_config_is_gpu_batched_and_fp32() -> None:
    config = source("config/cameras.yaml")
    runtime_config = source("services/ml_service/app/config.py")
    detector = source("services/ml_service/app/detector.py")
    legacy = source("services/ml_service/app/legacy_latest_detector.py")

    for text in (
        "detection:",
        "model: yolo26m.pt",
        "batch_size: 2",
        "width: 736",
        "height: 416",
        "confidence: 0.08",
        "iou: 0.70",
        "max_detections: 50",
        "target_fps_per_camera: 4.0",
    ):
        assert text in config
    assert "class DetectionConfig" in runtime_config
    assert '"classes": [0]' in legacy
    assert '"half": bool(config["half"])' not in detector
    assert '"half": bool(config["half"])' not in legacy


def test_legacy_scheduler_does_not_apply_four_fps_deadline_gate() -> None:
    legacy = source("services/ml_service/app/legacy_latest_detector.py")
    pipeline = source("services/ml_service/app/deepstream/pipeline.py")

    assert "next_due" not in legacy
    assert "max_submit_age_ms = 260.0" in legacy
    assert "max_result_age_ms = 700.0" in legacy
    assert "nominal_detector_fps = max(10.0" in pipeline


def test_ml_exposes_detection_health_and_results() -> None:
    main = source("services/ml_service/app/main.py")
    pipeline = source("services/ml_service/app/deepstream/pipeline.py")

    assert '"detector": current.detector_metrics()' in main
    assert '@app.get("/detections/{camera_id}")' in main
    assert 'runtime: Any | None = None' in main
    assert '"detection": detection' in pipeline
