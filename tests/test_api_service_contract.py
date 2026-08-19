from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_api_service_proxies_ml_application_state_without_owning_ml_runtime() -> None:
    main = source("services/api_service/app/main.py")
    client = source("services/api_service/app/ml_client.py")

    assert "MLServiceClient" in main
    assert '@app.get("/api/v1/ml/health")' in main
    assert '@app.get("/api/v1/cameras")' in main
    assert '@app.get("/api/v1/cameras/{camera_id}/detections")' in main
    assert '@app.get("/api/v1/cameras/{camera_id}/tracks")' in main
    assert "async def detections" in client
    assert "async def tracks" in client

    for forbidden in (
        "DeepStreamRuntime",
        "nvurisrcbin",
        "ultralytics",
        "BYTETracker",
        "YOLO(",
    ):
        assert forbidden not in main
        assert forbidden not in client


def test_api_preserves_ml_404_and_maps_transport_failures_to_503() -> None:
    main = source("services/api_service/app/main.py")
    client = source("services/api_service/app/ml_client.py")

    assert "class MLServiceNotFound" in client
    assert "response.status_code == 404" in client
    assert "status.HTTP_404_NOT_FOUND" in main
    assert "status.HTTP_503_SERVICE_UNAVAILABLE" in main


def test_api_launcher_and_smoke_exist() -> None:
    launcher = source("scripts/run_api_service.sh")
    smoke = source("scripts/smoke_api_service.py")

    assert "services.api_service.app.main" in launcher
    assert '"/api/v1/cameras/{camera_id}/detections"' not in smoke
    assert "/api/v1/cameras/" in smoke
    assert "API_SMOKE=PASS" in smoke
