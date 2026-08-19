from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frontend_uses_api_for_metadata_and_ml_only_for_video() -> None:
    main = source("services/frontend/app/main.py")
    api_client = source("services/frontend/app/api_client.py")
    wall = source("services/frontend/app/camera_wall.py")
    reader = source("services/frontend/app/mjpeg_reader.py")

    assert "ApiClient(self.settings.api_base_url" in main
    assert "CameraWall(ml_video_base_url=self.settings.ml_video_base_url)" in main
    assert '"/api/v1/ml/health"' in api_client
    assert '"/api/v1/cameras"' in api_client
    assert 'f"/video/{self.camera_id}"' in reader

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
        assert forbidden not in reader


def test_frontend_camera_wall_is_two_columns() -> None:
    wall = source("services/frontend/app/camera_wall.py")
    assert "COLUMNS = 2" in wall
    assert "divmod(index, self.COLUMNS)" in wall
    assert "for row in range(3)" in wall


def test_frontend_launcher_preflight_and_smoke_exist() -> None:
    launcher = source("scripts/run_frontend.sh")
    preflight = source("scripts/preflight_frontend.py")
    smoke = source("scripts/smoke_frontend_integration.py")

    assert "python scripts/preflight_frontend.py" in launcher
    assert "services.frontend.app.main" in launcher
    assert "FRONTEND_PREFLIGHT=PASS" in preflight
    assert '"/api/v1/cameras"' in smoke
    assert 'f"/video/{camera_id}"' in smoke
    assert "FRONTEND_SMOKE=PASS" in smoke
