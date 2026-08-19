from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frontend_uses_api_for_metadata_and_mmap_for_local_video() -> None:
    main = source("services/frontend/app/main.py")
    operator = source("services/frontend/app/operator_window.py")
    api_client = source("services/frontend/app/api_client.py")
    wall = source("services/frontend/app/camera_wall.py")
    mmap_reader = source("services/frontend/app/mmap_frame_reader.py")
    fallback = source("services/frontend/app/mjpeg_reader.py")

    assert "OperatorWindow" in main
    assert "ApiClient(self.settings.api_base_url" in operator
    assert "CameraWall(settings" in operator
    assert "self.monitoring.camera_wall.set_cameras(cameras)" in operator
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
        for text in (main, operator, api_client, wall, mmap_reader, fallback):
            assert forbidden not in text


def test_operator_ui_restores_dark_monitoring_layout() -> None:
    operator = source("services/frontend/app/operator_window.py")
    wall = source("services/frontend/app/camera_wall.py")

    for text in (
        "Apsidal",
        "Monitoring",
        "People",
        "Events",
        "Enrollment",
        "Settings",
        "People in Building",
        "Recent Views",
        "KNOWN",
        "UNKNOWN",
    ):
        assert text in operator
    assert '"#071018"' in operator
    assert "fullscreenRequested" in wall
    assert "toggle_focus" in wall
    assert 'self.status.setText("● LIVE")' in wall


def test_mmap_wall_repaints_only_new_frames_and_keeps_mjpeg_fallback() -> None:
    operator = source("services/frontend/app/operator_window.py")
    wall = source("services/frontend/app/camera_wall.py")

    assert "MmapVideoCanvas" in wall
    assert "int(version) == self._version" in wall
    assert "WA_OpaquePaintEvent" in wall
    assert "SmoothMjpegReader" in wall
    assert "_start_fallback" in wall
    assert "Qt.TimerType.PreciseTimer" in operator
    assert "frame_refresh_interval_ms" in operator


def test_focused_camera_gets_hq_scaling_without_six_feed_copy_load() -> None:
    wall = source("services/frontend/app/camera_wall.py")
    mmap_reader = source("services/frontend/app/mmap_frame_reader.py")

    # Normal wall stays on the cheap painter path; only the single focused tile
    # enables Qt's higher-quality scaling filter.
    assert "set_smooth_scaling" in wall
    assert "QPainter.RenderHint.SmoothPixmapTransform" in wall
    assert "set_presentation_mode" in wall
    assert "_apply_presentation_policy" in wall

    # Hidden tiles stop converting every mmap packet into a QImage while one
    # camera is focused, then immediately jump to the newest sequence on resume.
    assert "def set_active" in mmap_reader
    assert "if not self._active.is_set()" in mmap_reader
    assert "self.mmap_reader.set_active(bool(active))" in wall
    assert "if self._focused_camera:" in wall
    assert "self.tiles[self._focused_camera].refresh()" in wall


def test_tracking_overlay_uses_bytetrack_ids_plus_presentation_smoother() -> None:
    wall = source("services/frontend/app/camera_wall.py")
    mmap_publisher = source("services/ml_service/app/mmap_publisher.py")
    smoother = source("services/ml_service/app/presentation_smoother.py")

    assert "PresentationSmoother" in mmap_publisher
    assert 'label = f"Person T{int(track.track_id)}' in mmap_publisher
    assert "ByteTrack remains the only identity/association owner" in smoother
    assert "self.smoother.visible" in mmap_publisher
    assert "update_tracks" in wall


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
    assert "960" not in mmap_smoke
