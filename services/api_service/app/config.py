from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    ml_base_url: str
    ml_timeout_seconds: float


def load_settings() -> Settings:
    ml_base_url = os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")

    return Settings(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        ml_base_url=ml_base_url,
        ml_timeout_seconds=float(os.getenv("ML_SERVICE_TIMEOUT_SECONDS", "2.0")),
    )
