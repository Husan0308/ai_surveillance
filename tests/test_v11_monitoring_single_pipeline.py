from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_monitoring_consumes_existing_ml_frames_only() -> None:
    source = (ROOT / "services/ml_service/app/v11_monitoring.py").read_text(encoding="utf-8")
    assert "V11Step3TrackingV2" not in source
    assert "camera_runtime.stores" in source
    assert "TRT86DetectorClient" in source
    assert "extra_camera_pipeline=0" in source


def test_ml_main_injects_canonical_runtime_into_monitoring() -> None:
    source = (ROOT / "services/ml_service/app/main.py").read_text(encoding="utf-8")
    assert "runtime = DeepStreamRuntime(settings)" in source
    assert "V11MonitoringTrackerService(runtime)" in source
    assert "V11MonitoringTrackerService()" not in source
