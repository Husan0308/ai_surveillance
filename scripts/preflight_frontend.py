from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.frontend.app.config import load_settings


def fail(message: str) -> int:
    print(f"FRONTEND_PREFLIGHT=FAIL {message}", flush=True)
    return 1


def _validate_http_url(name: str, value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be an http(s) URL, got {value!r}")


def main() -> int:
    try:
        import PySide6
        from PySide6.QtGui import QImage  # noqa: F401
    except Exception as exc:
        return fail(f"PySide6 import failed: {type(exc).__name__}: {exc}")

    try:
        settings = load_settings()
        _validate_http_url("FRONTEND_API_BASE_URL", settings.api_base_url)
        _validate_http_url("FRONTEND_ML_VIDEO_BASE_URL", settings.ml_video_base_url)
        if settings.refresh_interval_ms < 250:
            raise ValueError("FRONTEND_REFRESH_INTERVAL_MS must be >= 250")
        if not 4 <= settings.frame_refresh_interval_ms <= 1000:
            raise ValueError("FRONTEND_FRAME_REFRESH_INTERVAL_MS must be 4..1000")
        if not 100 <= settings.track_refresh_interval_ms <= 2000:
            raise ValueError("FRONTEND_TRACK_REFRESH_INTERVAL_MS must be 100..2000")
        if settings.source_width <= 0 or settings.source_height <= 0 or settings.source_fps <= 0:
            raise ValueError("frontend source geometry/fps must be positive")

        if settings.video_transport == "mmap":
            from shared.mmap_frame import frame_directory

            frame_dir = frame_directory()
            if not frame_dir.exists() or not frame_dir.is_dir():
                raise ValueError(f"mmap frame directory unavailable: {frame_dir}")
    except Exception as exc:
        return fail(f"configuration invalid: {type(exc).__name__}: {exc}")

    print(
        f"FRONTEND_CONFIG api={settings.api_base_url} video={settings.ml_video_base_url} "
        f"transport={settings.video_transport} metadata_ms={settings.refresh_interval_ms} "
        f"frame_poll_ms={settings.frame_refresh_interval_ms} track_ms={settings.track_refresh_interval_ms} "
        f"source={settings.source_width}x{settings.source_height}@{settings.source_fps}",
        flush=True,
    )
    if settings.video_transport == "mmap":
        from shared.mmap_frame import frame_directory

        print(f"FRONTEND_MMAP frame_dir={frame_directory()} latest_only=PASS", flush=True)
    print(f"FRONTEND_LIB PySide6={PySide6.__version__}", flush=True)
    print("FRONTEND_PREFLIGHT=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
