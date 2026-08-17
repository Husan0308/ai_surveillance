#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UI_FILE = ROOT / "services" / "camera_v2" / "monitor_ui.py"


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    print("MONITOR_UI_PREFLIGHT=FAIL")
    return 1


def main() -> int:
    if not UI_FILE.exists():
        return fail(f"missing {UI_FILE}")

    try:
        ast.parse(UI_FILE.read_text(encoding="utf-8"), filename=str(UI_FILE))
        print("python_syntax=OK")
    except Exception as exc:
        return fail(f"monitor_ui.py syntax: {exc}")

    try:
        import PySide6
        print(f"PySide6=OK version={PySide6.__version__}")
    except Exception as exc:
        return fail(f"PySide6 unavailable: {exc}")

    display = os.environ.get("DISPLAY", "")
    if not display:
        return fail("DISPLAY is empty; run from the graphical desktop/AnyDesk session")
    print(f"DISPLAY=OK {display}")

    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        Gst.init(None)
        _ = GstVideo.VideoOverlay
        print("GstVideoOverlay=OK")
    except Exception as exc:
        return fail(f"GStreamer video overlay unavailable: {exc}")

    required = (
        "nvurisrcbin",
        "nvstreammux",
        "nvtracker",
        "nvmultistreamtiler",
        "nvvideoconvert",
        "nvdsosd",
        "nveglglessink",
        "queue",
    )
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        return fail("missing plugins: " + ", ".join(missing))
    print("deepstream_plugins=OK")

    try:
        from services.ml_service.app.config import load_settings
        settings = load_settings()
        cameras = list(settings.cameras)
    except Exception as exc:
        return fail(f"camera config could not load: {exc}")

    if len(cameras) != 6:
        return fail(f"expected 6 cameras, found {len(cameras)}")
    print("camera_count=OK 6")

    from services.camera_v2 import monitor_ui

    checks = {
        "columns": monitor_ui.GRID_COLUMNS == 2,
        "rows": monitor_ui.GRID_ROWS == 3,
        "tile_width": monitor_ui.TILE_WIDTH == 640,
        "tile_height": monitor_ui.TILE_HEIGHT == 360,
        "wall_width": monitor_ui.WALL_WIDTH == 1280,
        "wall_height": monitor_ui.WALL_HEIGHT == 1080,
        "same_pixel_budget": monitor_ui.WALL_WIDTH * monitor_ui.WALL_HEIGHT == 1920 * 720,
    }
    for name, ok in checks.items():
        print(f"contract {name}={'OK' if ok else 'FAIL'}")
        if not ok:
            return fail(f"layout contract failed: {name}")

    print("layout=2 columns x 3 rows")
    print("camera_tile=640x360")
    print("wall=1280x1080")
    print("pixel_budget=UNCHANGED_FROM_1920x720")
    print("MONITOR_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
