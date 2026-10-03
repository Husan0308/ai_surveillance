"""One surveillance deployment contract; port 8000 belongs to AutoMagicCalib."""
from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlsplit, urlunsplit


def _port(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be in 1..65535")
    return value


def _url(name: str, default: str) -> str:
    value = os.getenv(name, default).rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError(f"{name} must be an HTTP(S) URL without credentials")
    return value


@dataclass(frozen=True)
class Deployment:
    api_host: str
    api_port: int
    ml_host: str
    ml_port: int
    ml_service_url: str
    frontend_api_base_url: str
    frontend_ml_video_base_url: str

    @property
    def monitoring_ws_url(self) -> str:
        parts = urlsplit(self.frontend_api_base_url)
        return urlunsplit(("wss" if parts.scheme == "https" else "ws", parts.netloc,
                          parts.path + "/ws/v1/monitoring", "", ""))

    def environment(self) -> dict[str, str]:
        return {"API_HOST": self.api_host, "API_PORT": str(self.api_port),
                "ML_HOST": self.ml_host, "ML_PORT": str(self.ml_port),
                "ML_SERVICE_URL": self.ml_service_url,
                "FRONTEND_API_BASE_URL": self.frontend_api_base_url,
                "FRONTEND_ML_VIDEO_BASE_URL": self.frontend_ml_video_base_url}


def load_deployment() -> Deployment:
    api, ml = _port("API_PORT", 8100), _port("ML_PORT", 8101)
    if api == ml:
        raise ValueError("API_PORT and ML_PORT must differ")
    return Deployment(os.getenv("API_HOST", "0.0.0.0"), api,
                      os.getenv("ML_HOST", "0.0.0.0"), ml,
                      _url("ML_SERVICE_URL", f"http://127.0.0.1:{ml}"),
                      _url("FRONTEND_API_BASE_URL", f"http://127.0.0.1:{api}"),
                      _url("FRONTEND_ML_VIDEO_BASE_URL", f"http://127.0.0.1:{ml}"))


def analytics_enabled() -> bool:
    return os.getenv("SURVEILLANCE_ANALYTICS_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
