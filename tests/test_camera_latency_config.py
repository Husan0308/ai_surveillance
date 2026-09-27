from pathlib import Path
from collections import deque
from types import SimpleNamespace

from services.ml_service.app.config import CameraConfig, load_settings
from services.ml_service.app.deepstream.capture import (
    DeepStreamCapture,
    _decoder_extra_surfaces_for_camera,
)
from services.camera_v11.preview_only_runtime import (
    _test_decoder_low_latency_enabled,
    _test_rtsp_transport_for_camera,
)


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


def test_decoder_low_latency_mode_is_explicitly_opt_in():
    settings = load_settings()

    def pipeline(low_latency: bool) -> str:
        capture = DeepStreamCapture.__new__(DeepStreamCapture)
        capture.camera_id = "CAM-01"
        capture.uri = "rtsp://camera.invalid/live"
        capture.codec = "h264"
        capture.config = settings.deepstream
        capture.transport = "tcp"
        capture.username = ""
        capture.password = ""
        capture.output_bgrx = True
        capture.latency_ms = 20
        capture.low_latency_mode = low_latency
        return capture._build_pipeline()

    assert "low-latency-mode=true" not in pipeline(False)
    assert "low-latency-mode=true" in pipeline(True)


def test_preview_low_latency_allowlist_keeps_unlisted_cameras_on_default(monkeypatch):
    monkeypatch.setenv("MV3DT_TEST_DECODER_LOW_LATENCY", "1")
    monkeypatch.setenv(
        "MV3DT_TEST_DECODER_LOW_LATENCY_CAMERAS", "CAM-01,CAM-04,CAM-05,CAM-06"
    )

    assert _test_decoder_low_latency_enabled("CAM-01") is True
    assert _test_decoder_low_latency_enabled("CAM-02") is False
    assert _test_decoder_low_latency_enabled("CAM-03") is False
    assert _test_decoder_low_latency_enabled("CAM-04") is True


def test_preview_low_latency_legacy_switch_is_unchanged_without_allowlist(monkeypatch):
    monkeypatch.delenv("MV3DT_TEST_DECODER_LOW_LATENCY_CAMERAS", raising=False)
    monkeypatch.setenv("MV3DT_TEST_DECODER_LOW_LATENCY", "1")
    assert _test_decoder_low_latency_enabled("CAM-02") is True

    monkeypatch.setenv("MV3DT_TEST_DECODER_LOW_LATENCY", "0")
    assert _test_decoder_low_latency_enabled("CAM-02") is False


def test_rtsp_transport_override_is_test_only_and_camera_scoped(monkeypatch):
    monkeypatch.delenv("MV3DT_TEST_RTSP_TRANSPORT_BY_CAMERA", raising=False)
    assert _test_rtsp_transport_for_camera("CAM-02", "tcp") == "tcp"
    monkeypatch.setenv("MV3DT_TEST_RTSP_TRANSPORT_BY_CAMERA", "CAM-02=udp")
    assert _test_rtsp_transport_for_camera("CAM-02", "tcp") == "udp"
    assert _test_rtsp_transport_for_camera("CAM-03", "tcp") == "tcp"


def test_rtsp_transport_override_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("MV3DT_TEST_RTSP_TRANSPORT_BY_CAMERA", "CAM-02=bad")
    try:
        _test_rtsp_transport_for_camera("CAM-02", "tcp")
    except ValueError as exc:
        assert "tcp, udp, or auto" in str(exc)
    else:
        raise AssertionError("unsupported RTSP transport was accepted")


def test_decoder_extra_surfaces_override_is_camera_scoped():
    overrides = "CAM-02=8,CAM-03=8,CAM-04=8"
    assert _decoder_extra_surfaces_for_camera("CAM-02", 4, overrides) == 8
    assert _decoder_extra_surfaces_for_camera("CAM-04", 1, overrides) == 8
    assert _decoder_extra_surfaces_for_camera("CAM-05", 4, overrides) == 4
    assert _decoder_extra_surfaces_for_camera("CAM-02", 4, "CAM-02=0") == 0


def test_decoder_extra_surfaces_override_rejects_unsupported_values():
    try:
        _decoder_extra_surfaces_for_camera("CAM-02", 4, "CAM-02=25")
    except ValueError as exc:
        assert "0..24" in str(exc)
    else:
        raise AssertionError("out-of-range decoder surface override was accepted")


def test_missing_decoder_input_timestamp_is_not_zero_latency(monkeypatch):
    from services.ml_service.app.deepstream import capture as module

    capture = DeepStreamCapture.__new__(DeepStreamCapture)
    capture._decoder_timing_started_ns = 0
    capture._decoder_output_buffers = 0
    capture._last_decoder_output_mono_ns = 0
    capture._last_decoder_output_pts = -1
    capture._decoder_input_times = {}
    capture._decoder_output_times = {}
    capture._decoder_output_order = deque(maxlen=256)
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 500)
    info = SimpleNamespace(get_buffer=lambda: SimpleNamespace(pts=123))
    capture._decoder_output_probe(None, info)
    assert capture._decoder_output_times[123] == (0, 500, 0)
