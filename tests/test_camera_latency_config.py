from pathlib import Path

from services.ml_service.app.config import CameraConfig, load_settings


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
