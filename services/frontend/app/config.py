from __future__ import annotations

import os
from dataclasses import dataclass

from services.runtime_ports import API_BASE_URL_DEFAULT, ML_BASE_URL_DEFAULT


@dataclass(frozen=True)
class FrontendSettings:
    api_base_url: str
    ml_video_base_url: str
    refresh_interval_ms: int
    frame_refresh_interval_ms: int


def load_settings() -> FrontendSettings:
    return FrontendSettings(
        api_base_url=os.getenv("FRONTEND_API_BASE_URL", API_BASE_URL_DEFAULT).rstrip("/"),
        ml_video_base_url=os.getenv("FRONTEND_ML_VIDEO_BASE_URL", ML_BASE_URL_DEFAULT).rstrip("/"),
        refresh_interval_ms=int(os.getenv("FRONTEND_REFRESH_INTERVAL_MS", "200")),
        frame_refresh_interval_ms=int(os.getenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "16")),
    )
