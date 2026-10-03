from __future__ import annotations

import os
from dataclasses import dataclass
from services.shared.deployment import load_deployment


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    ml_base_url: str
    ml_timeout_seconds: float


def load_settings() -> Settings:
    deployment = load_deployment()
    return Settings(
        host=deployment.api_host,
        port=deployment.api_port,
        ml_base_url=deployment.ml_service_url,
        ml_timeout_seconds=float(os.getenv("ML_SERVICE_TIMEOUT_SECONDS", "4.0")),
    )
