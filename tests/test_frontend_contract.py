from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frontend_uses_api_for_metadata_and_mmap_for_local_video() -> None:
    main = source("services/frontend/app/main.py")
    api_client = source("services/frontend/app/api_client.py")
    wall = source("services/frontend/app/camera_wall.py")
    mmap_reader = source("services/frontend/app/mmap_frame_reader.py")
    fallback = source("services/frontend/app/mjpeg_reader.py")

    assert "ApiClient(self.settings.api_base_url" in main
    assert "CameraWall(self.settings)" in main
    assert "self.camera_wall.set_cameras(cameras)" in main
    assert '"/api/v1/ml/health"' in api_client
    assert '"/api/v1/cameras"' in api_client
    assert '"/api/v1/tracks"' in api_client
    assert "SmoothMmapFrameReader" in wall
    assert "MmapFrameReader" in mmap_reader
    assert "mapping_is_current" in mmap_reader
    assert 'f"/video/{self.camera_id}"' in fallback

    for forbidden in (
        "DeepStreamRuntime",
        "nvurisrcbin",
        "ultralytics",
        "BYTETracker",
        "YOLO(",
    ):
        assert forbidden not in main
        assert forbidden not in api_client
        assert forbidden not in wall
        assert forbidden not in mmap_reader
        assert forbidden not in fallback


def test_mmap_wall_repaints_only_new_frames_and_keeps_mjpeg_fallback() -> None:
    main = source("services/frontend/app/main.py")
    wall = source("services/frontend/app/camera_wall.py")

    assert "MmapVideoCanvas" in wall
    assert "int(version) == self._version" in wall
    assert "WA_OpaquePaintEvent" in wall
    assert "SmoothPixmapTransform" in wall  # comment documents it is deliberately OFF
    assert "SmoothMjpegReader" in wall
    assert "_start_fallback" in wall
    assert "Qt.TimerType.PreciseTimer" in main
    assert "frame_refresh_interval_ms" in main


def test_tracking_overlay_is_baked_by_ml_not_duplicated_in_qt() -> None:
    wall = source("services/frontend/app/camera_wall.py")
    mmap_publisher = source("services/ml_service/app/mmap_publisher.py")
    jpeg_publisher = source("services/ml_service/app/jpeg_publisher.py")

    assert "self._image_for_encode(frame)" in mmap_publisher
    assert "track_id = getattr(detection, \"track_id\", None)" in jpeg_publisher
    assert "would duplicate boxes" in wall


def test_frontend_camera_wall_is_two_columns() -> None:
    wall = source("services/frontend/app/camera_wall.py")
    assert "COLUMNS = 2" in wall
    assert "divmod(index, self.COLUMNS)" in wall
    assert "for row in range(3)" in wall


def test_frontend_launcher_preflight_and_mmap_smoke_exist() -> None:
    launcher = source("scripts/run_frontend.sh")
    preflight = source("scripts/preflight_frontend.py")
    integration = source("scripts/smoke_frontend_integration.py")
    mmap_smoke = source("scripts/smoke_mmap_video.py")

    assert "python scripts/preflight_frontend.py" in launcher
    assert "services.frontend.app.main" in launcher
    assert "FRONTEND_PREFLIGHT=PASS" in preflight
    assert "FRONTEND_MMAP" in preflight
    assert '"/api/v1/cameras"' in integration
    assert "FRONTEND_SMOKE=PASS" in integration
    assert "MMAP_VIDEO_SMOKE=PASS" in mmap_smoke
    assert "960" not in mmap_smoke  # geometry comes from canonical frontend settings
