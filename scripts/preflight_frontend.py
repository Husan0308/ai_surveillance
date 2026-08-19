from __future__ import annotations

import os
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
        if not 20 <= settings.frame_refresh_interval_ms <= 1000:
            raise ValueError("FRONTEND_FRAME_REFRESH_INTERVAL_MS must be 20..1000")
        if not 100 <= settings.track_refresh_interval_ms <= 2000:
            raise ValueError("FRONTEND_TRACK_REFRESH_INTERVAL_MS must be 100..2000")
        if settings.source_width <= 0 or settings.source_height <= 0 or settings.source_fps <= 0:
            raise ValueError("frontend source geometry/fps must be positive")

        if settings.video_transport == "shm":
            session = os.getenv("XDG_SESSION_TYPE", "").strip().lower()
            display = os.getenv("DISPLAY", "").strip()
            if session != "x11":
                raise ValueError(f"native nveglglessink mode requires X11, got {session or 'unknown'}")
            if not display:
                raise ValueError("native nveglglessink mode requires DISPLAY")

            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            required = ("shmsrc", "nvvideoconvert", "nveglglessink")
            missing = [name for name in required if Gst.ElementFactory.find(name) is None]
            if missing:
                raise ValueError(f"missing native video plugins: {','.join(missing)}")
    except Exception as exc:
        return fail(f"configuration invalid: {type(exc).__name__}: {exc}")

    print(
        f"FRONTEND_CONFIG api={settings.api_base_url} video={settings.ml_video_base_url} "
        f"transport={settings.video_transport} metadata_ms={settings.refresh_interval_ms} "
        f"track_ms={settings.track_refresh_interval_ms} source="
        f"{settings.source_width}x{settings.source_height}@{settings.source_fps}",
        flush=True,
    )
    if settings.video_transport == "shm":
        print(
            f"FRONTEND_NATIVE plugins=PASS session={os.getenv('XDG_SESSION_TYPE')} "
            f"display={os.getenv('DISPLAY')} shm_dir={settings.shm_video_dir}",
            flush=True,
        )
    print(f"FRONTEND_LIB PySide6={PySide6.__version__}", flush=True)
    print("FRONTEND_PREFLIGHT=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
