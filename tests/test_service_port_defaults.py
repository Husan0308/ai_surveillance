from __future__ import annotations

import importlib

from services.api_service.app.config import load_settings as load_api_settings
from services.frontend.app.config import load_settings as load_frontend_settings


_PORT_ENV = (
    "API_PORT",
    "ML_PORT",
    "ML_SERVICE_URL",
    "FRONTEND_API_BASE_URL",
    "FRONTEND_ML_VIDEO_BASE_URL",
)


def _clear_port_environment(monkeypatch) -> None:
    for name in _PORT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("V11_ENV_FILE", "/dev/null")


def _run_ml_main_port(monkeypatch) -> int:
    ml_main = importlib.import_module("services.ml_service.app.main")
    assert ml_main.ML_PORT_DEFAULT == 8101
    captured: dict = {}

    def fake_run(app, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(ml_main.uvicorn, "run", fake_run)
    ml_main.main()
    return int(captured["port"])


def test_service_defaults_use_canonical_8100_8101_ports(monkeypatch) -> None:
    _clear_port_environment(monkeypatch)

    frontend = load_frontend_settings()
    api = load_api_settings()

    assert frontend.api_base_url == "http://127.0.0.1:8100"
    assert frontend.ml_video_base_url == "http://127.0.0.1:8101"
    assert api.port == 8100
    assert api.ml_base_url == "http://127.0.0.1:8101"
    assert _run_ml_main_port(monkeypatch) == 8101


def test_service_port_environment_overrides_still_work(monkeypatch) -> None:
    monkeypatch.setenv("API_PORT", "18200")
    monkeypatch.setenv("ML_PORT", "18201")
    monkeypatch.setenv("ML_SERVICE_URL", "http://ml.internal:18201/")
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "http://api.internal:18200/")
    monkeypatch.setenv("FRONTEND_ML_VIDEO_BASE_URL", "http://ml.internal:18201/")
    monkeypatch.setenv("V11_ENV_FILE", "/dev/null")

    frontend = load_frontend_settings()
    api = load_api_settings()

    assert frontend.api_base_url == "http://api.internal:18200"
    assert frontend.ml_video_base_url == "http://ml.internal:18201"
    assert api.port == 18200
    assert api.ml_base_url == "http://ml.internal:18201"
    assert _run_ml_main_port(monkeypatch) == 18201
