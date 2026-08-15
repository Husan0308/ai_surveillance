from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendSettings:
    api_base_url: str
    refresh_interval_ms: int
    frame_bus_dir: str
    frame_refresh_interval_ms: int
    frame_stale_after_ms: int


def load_settings() -> FrontendSettings:
    return FrontendSettings(api_base_url=os.getenv("FRONTEND_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"), refresh_interval_ms=int(os.getenv("FRONTEND_REFRESH_INTERVAL_MS", "2000")), frame_bus_dir=os.getenv("FRAME_BUS_DIR", "/dev/shm/ai_surveillance"), frame_refresh_interval_ms=int(os.getenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "66")), frame_stale_after_ms=int(os.getenv("FRONTEND_FRAME_STALE_AFTER_MS", "2000")))
