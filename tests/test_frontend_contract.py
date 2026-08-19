from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frontend_uses_api_for_metadata_and_ml_only_for_video_transport() -> None:
    main = source("services/frontend/app/main.py")
    api_client = source("services/frontend/app/api_client.py")
    wall = source("services/frontend/app/camera_wall.py")
    native = source("services/frontend/app/native_video.py")
    reader = source("services/frontend/app/mjpeg_reader.py")

    assert "ApiClient(self.settings.api_base_url" in main
    assert "CameraWall(self.settings)" in main
    assert "self.camera_wall.set_cameras(cameras)" in main
    assert '"/api/v1/ml/health"' in api_client
    assert '"/api/v1/cameras"' in api_client
    assert '"/api/v1/tracks"' in api_client
    assert "NativeShmRenderer" in wall
    assert "shmsrc" in native
    assert "nveglglessink" in native
    assert 'f"/video/{self.camera_id}"' in reader

    # Frontend may render ML-owned frames, but it must never own RTSP decode,
    # inference or tracking runtime.
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
        assert forbidden not in native
        assert forbidden not in reader


def test_native_video_uses_negotiated_render_profile_and_low_latency_sink() -> None:
    native = source("services/frontend/app/native_video.py")
    wall = source("services/frontend/app/camera_wall.py")
    capture = source("services/ml_service/app/deepstream/capture.py")

    assert '"wait-for-connection=false"' in capture
    assert '"leaky=downstream"' in capture
    assert '"shmsink"' in capture
    assert '"video/x-raw,format=NV12"' in capture
    assert "render_width" in wall
    assert "render_height" in wall
    assert "render_format" in wall
    assert "pixel_format=pixel_format" in wall
    assert '"sync=false"' in native
    assert '"qos=false"' in native
    assert '"max-size-buffers=1"' in native
    assert "SmoothMjpegReader" in wall
    assert "_start_fallback" in wall


def test_track_boxes_are_vector_overlay_not_baked_into_native_video() -> None:
    main = source("services/frontend/app/main.py")
    wall = source("services/frontend/app/camera_wall.py")

    assert "TrackOverlay" in wall
    assert "QPainter" in wall
    assert 'text = f"Person T{track_id}' in wall
    assert "track_refresh_interval_ms" in main
    assert "self.api.refresh_tracks" in main
    assert "self.camera_wall.update_tracks" in main


def test_frontend_camera_wall_is_two_columns() -> None:
    wall = source("services/frontend/app/camera_wall.py")
    assert "COLUMNS = 2" in wall
    assert "divmod(index, self.COLUMNS)" in wall
    assert "for row in range(3)" in wall


def test_frontend_launcher_preflight_and_smokes_exist() -> None:
    launcher = source("scripts/run_frontend.sh")
    preflight = source("scripts/preflight_frontend.py")
    smoke = source("scripts/smoke_frontend_integration.py")
    shm_smoke = source("scripts/smoke_shm_video.py")

    assert "python scripts/preflight_frontend.py" in launcher
    assert "services.frontend.app.main" in launcher
    assert "FRONTEND_PREFLIGHT=PASS" in preflight
    assert "nveglglessink" in preflight
    assert '"/api/v1/cameras"' in smoke
    assert 'f"/video/{camera_id}"' in smoke
    assert "FRONTEND_SMOKE=PASS" in smoke
    assert "SHM_VIDEO_SMOKE=PASS" in shm_smoke
    assert "native=" in shm_smoke
    assert "analysis=" in shm_smoke
