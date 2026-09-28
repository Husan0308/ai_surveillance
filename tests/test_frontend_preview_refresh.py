from services.frontend.app.config import load_settings
from services.frontend.app.camera_wall import _fast_preview_scale_enabled


def test_production_preview_poll_interval_defaults_to_eight_ms(monkeypatch):
    monkeypatch.delenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", raising=False)

    assert load_settings().frame_refresh_interval_ms == 8


def test_preview_poll_interval_remains_operator_configurable(monkeypatch):
    monkeypatch.setenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "12")

    assert load_settings().frame_refresh_interval_ms == 12


def test_fast_tile_scaling_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("FRONTEND_PREVIEW_FAST_SCALE", raising=False)
    assert _fast_preview_scale_enabled()

    monkeypatch.setenv("FRONTEND_PREVIEW_FAST_SCALE", "0")
    assert not _fast_preview_scale_enabled()
