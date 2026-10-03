from __future__ import annotations

import os
from dataclasses import dataclass
from services.shared.deployment import load_deployment


@dataclass(frozen=True)
class FrontendSettings:
    api_base_url: str
    ml_video_base_url: str
    refresh_interval_ms: int
    frame_refresh_interval_ms: int


def load_settings() -> FrontendSettings:
    deployment = load_deployment()
    return FrontendSettings(
        api_base_url=deployment.frontend_api_base_url,
        ml_video_base_url=deployment.frontend_ml_video_base_url,
        refresh_interval_ms=int(os.getenv("FRONTEND_REFRESH_INTERVAL_MS", "200")),
        frame_refresh_interval_ms=int(os.getenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "16")),
    )
