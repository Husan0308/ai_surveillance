"""Service ports must not collide with Auto Magic Calibration on 8000."""
import importlib

from services.api_service.app.config import load_settings as api_settings
from services.frontend.app.config import load_settings as frontend_settings


def ml_port(monkeypatch):
    main = importlib.import_module("services.ml_service.app.main")
    captured = {}
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kw: captured.update(kw))
    main.main()
    return captured["port"]


def test_default_ports_preserve_calibration(monkeypatch):
    for name in ("API_PORT", "ML_PORT", "ML_SERVICE_URL", "FRONTEND_API_BASE_URL",
                 "FRONTEND_ML_VIDEO_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("V11_ENV_FILE", "/dev/null")
    assert api_settings().port == 8100
    assert api_settings().ml_base_url == "http://127.0.0.1:8101"
    assert frontend_settings().api_base_url == "http://127.0.0.1:8100"
    assert frontend_settings().ml_video_base_url == "http://127.0.0.1:8101"
    assert ml_port(monkeypatch) == 8101


def test_environment_overrides_remain_supported(monkeypatch):
    monkeypatch.setenv("V11_ENV_FILE", "/dev/null")
    monkeypatch.setenv("API_PORT", "18100")
    monkeypatch.setenv("ML_PORT", "18101")
    monkeypatch.setenv("ML_SERVICE_URL", "http://ml:18101/")
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "http://api:18100/")
    monkeypatch.setenv("FRONTEND_ML_VIDEO_BASE_URL", "http://ml:18101/")
    assert api_settings().port == 18100
    assert api_settings().ml_base_url == "http://ml:18101"
    assert frontend_settings().api_base_url == "http://api:18100"
    assert frontend_settings().ml_video_base_url == "http://ml:18101"
    assert ml_port(monkeypatch) == 18101
