from collections import deque
from pathlib import Path
from types import SimpleNamespace

from services.ml_service.app.config import CameraConfig, load_settings
from services.ml_service.app.deepstream.capture import (
    DeepStreamCapture,
    _decoder_extra_surfaces_for_camera,
)
from services.camera_v11.preview_only_runtime import CameraStats, _decoder_low_latency_enabled


def test_project_camera_latencies_are_parsed():
    settings = load_settings(Path("config/cameras.yaml"))
    cameras = {camera.camera_id: camera for camera in settings.cameras}

    assert cameras["CAM-01"].effective_latency_ms(settings.deepstream.latency_ms) == 20
    assert cameras["CAM-02"].effective_latency_ms(settings.deepstream.latency_ms) == 20
    assert cameras["CAM-03"].effective_latency_ms(settings.deepstream.latency_ms) == 20
    assert cameras["CAM-04"].effective_latency_ms(settings.deepstream.latency_ms) == 80
    assert cameras["CAM-05"].effective_latency_ms(settings.deepstream.latency_ms) == 80
    assert cameras["CAM-06"].effective_latency_ms(settings.deepstream.latency_ms) == 80


def test_camera_latency_falls_back_to_global():
    camera = CameraConfig(
        camera_id="CAM-X",
        uri="rtsp://example.invalid/stream",
        username="",
        password="",
        latency_ms=None,
    )
    assert camera.effective_latency_ms(150) == 150


def test_decoder_low_latency_mode_uses_validated_camera_profile():
    settings = load_settings()
    cameras = {camera.camera_id: camera for camera in settings.cameras}
    assert all(cameras[camera_id].decoder_low_latency_mode for camera_id in cameras)
    assert cameras["CAM-04"].decoder_extra_surfaces == 8
    assert cameras["CAM-01"].decoder_extra_surfaces is None

    def pipeline(low_latency: bool) -> str:
        capture = DeepStreamCapture.__new__(DeepStreamCapture)
        capture.camera_id = "CAM-02"
        capture.uri = "rtsp://camera.invalid/live"
        capture.codec = "h264"
        capture.config = settings.deepstream
        capture.transport = "tcp"
        capture.username = ""
        capture.password = ""
        capture.output_bgrx = True
        capture.latency_ms = 20
        capture.low_latency_mode = low_latency
        capture.decoder_extra_surfaces = settings.deepstream.decoder_extra_surfaces
        return capture._build_pipeline()

    assert "low-latency-mode=true" not in pipeline(False)
    assert "low-latency-mode=true" in pipeline(True)


def test_preview_low_latency_defaults_to_camera_profile_without_override(monkeypatch):
    monkeypatch.delenv("MV3DT_TEST_DECODER_LOW_LATENCY_CAMERAS", raising=False)
    monkeypatch.delenv("MV3DT_TEST_DECODER_LOW_LATENCY", raising=False)
    assert _decoder_low_latency_enabled("CAM-02", True) is True
    assert _decoder_low_latency_enabled("CAM-02", False) is False


def test_decoder_extra_surfaces_override_is_camera_scoped():
    overrides = "CAM-02=8,CAM-03=8,CAM-04=8"
    assert _decoder_extra_surfaces_for_camera("CAM-02", 4, overrides) == 8
    assert _decoder_extra_surfaces_for_camera("CAM-04", 1, overrides) == 8
    assert _decoder_extra_surfaces_for_camera("CAM-05", 4, overrides) == 4


def test_missing_decoder_input_timestamp_is_not_zero_latency(monkeypatch):
    from services.ml_service.app.deepstream import capture as module

    capture = DeepStreamCapture.__new__(DeepStreamCapture)
    capture._decoder_input_times = {}
    capture._decoder_output_times = {}
    capture._decoder_output_order = deque(maxlen=256)
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 500)
    info = SimpleNamespace(
        get_buffer=lambda: SimpleNamespace(pts=123, dts=module.Gst.CLOCK_TIME_NONE)
    )
    capture._decoder_output_probe(None, info)
    assert capture._decoder_output_times[123] == (0, 500, 0)


def test_recovery_preserves_redacted_failure_evidence():
    stats = CameraStats("CAM-05")
    stats.record_failure(RuntimeError('rtsp://example:secret@camera.invalid/live user-pw="secret"'))
    assert stats.failures == 1
    assert stats.state == "DISCONNECTED"
    stats.state = "LIVE"
    stats.last_error = ""
    snapshot = stats.snapshot()
    assert snapshot["last_error"] == ""
    assert snapshot["last_failure_monotonic_ns"] > 0
    assert "secret" not in snapshot["last_failure_error"]
    assert "example:" not in snapshot["last_failure_error"]
    assert "camera.invalid/live" in snapshot["last_failure_error"]
