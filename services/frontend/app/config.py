from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendSettings:
    api_base_url: str
    ml_video_base_url: str
    refresh_interval_ms: int
    frame_refresh_interval_ms: int
    track_refresh_interval_ms: int
    video_transport: str
    source_width: int
    source_height: int
    source_fps: int


def load_settings() -> FrontendSettings:
    video_transport = os.getenv("FRONTEND_VIDEO_TRANSPORT", "mmap").strip().lower()
    if video_transport not in {"mmap", "mjpeg"}:
        raise ValueError("FRONTEND_VIDEO_TRANSPORT must be mmap or mjpeg")
    return FrontendSettings(
        api_base_url=os.getenv("FRONTEND_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        ml_video_base_url=os.getenv("FRONTEND_ML_VIDEO_BASE_URL", "http://127.0.0.1:8001").rstrip("/"),
        refresh_interval_ms=int(os.getenv("FRONTEND_REFRESH_INTERVAL_MS", "2000")),
        # Old proven wall polled faster than the 20 FPS source but repainted only
        # cameras whose mmap sequence changed.
        frame_refresh_interval_ms=int(os.getenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "12")),
        track_refresh_interval_ms=int(os.getenv("FRONTEND_TRACK_REFRESH_INTERVAL_MS", "250")),
        video_transport=video_transport,
        source_width=int(os.getenv("FRONTEND_SOURCE_WIDTH", "960")),
        source_height=int(os.getenv("FRONTEND_SOURCE_HEIGHT", "540")),
        source_fps=int(os.getenv("FRONTEND_SOURCE_FPS", "20")),
    )
