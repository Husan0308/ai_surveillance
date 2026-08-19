from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_tracking_is_separate_from_detection_and_cross_camera_identity() -> None:
    tracking = source("services/ml_service/app/tracking.py")
    detector = source("services/ml_service/app/detector.py")

    assert "class PersonTracker" in tracking
    assert "BYTETracker" in tracking
    assert '"scope": "per-camera"' in tracking
    assert '"reid": False' in tracking
    assert '"face": False' in tracking
    assert "BYTETracker" not in detector

    for forbidden in (
        "insightface",
        "fastreid",
        "torchreid",
        "GlobalIdentity",
        "cross_camera",
    ):
        assert forbidden not in tracking.lower()


def test_tracking_configuration_is_time_based_for_low_detector_fps() -> None:
    config = source("config/cameras.yaml")
    runtime_config = source("services/ml_service/app/config.py")
    tracking = source("services/ml_service/app/tracking.py")

    assert "tracking:" in config
    assert "track_buffer_seconds: 2.5" in config
    assert "class TrackingConfig" in runtime_config
    assert "track_buffer_seconds" in runtime_config
    assert "buffer_frames" in tracking
    assert "self.detector_fps" in tracking


def test_tracking_is_exposed_but_not_global_identity() -> None:
    main = source("services/ml_service/app/main.py")
    pipeline = source("services/ml_service/app/deepstream/pipeline.py")
    publisher = source("services/ml_service/app/jpeg_publisher.py")

    assert '"tracker": current.tracker_metrics()' in main
    assert '@app.get("/tracks/{camera_id}")' in main
    assert "self.tracker = PersonTracker(" in pipeline
    assert 'text = f"T{int(track_id)} {detection.confidence:.2f}"' in publisher


def test_tracking_exposes_churn_diagnostics() -> None:
    tracking = source("services/ml_service/app/tracking.py")
    stability = source("scripts/smoke_person_tracking_stability.py")

    assert "created_tracks: int = 0" in tracking
    assert '"created_tracks": created' in tracking
    assert "self._metrics.created_tracks += 1" in tracking
    assert "same_count_id_changes" in stability
    assert "created_delta" in stability
    assert "PERSON_TRACK_STABILITY=PASS" in stability
