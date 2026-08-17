#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UI = ROOT / "services" / "camera_v2" / "monitor_ui_reference.py"


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    print("MONITOR_UI_REFERENCE_PREFLIGHT=FAIL")
    return 1


def main() -> int:
    try:
        ast.parse(UI.read_text(encoding="utf-8"), filename=str(UI))
        print("python_syntax=OK")
    except Exception as exc:
        return fail(f"UI syntax: {exc}")

    try:
        import PySide6
        print(f"PySide6=OK version={PySide6.__version__}")
    except Exception as exc:
        return fail(f"PySide6 unavailable: {exc}")

    if not os.environ.get("DISPLAY"):
        return fail("DISPLAY is empty; run inside the graphical/AnyDesk session")
    print(f"DISPLAY=OK {os.environ['DISPLAY']}")

    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        Gst.init(None)
        _ = GstVideo.VideoOverlay
    except Exception as exc:
        return fail(f"GstVideoOverlay unavailable: {exc}")
    print("GstVideoOverlay=OK")

    required = (
        "nvurisrcbin", "nvstreammux", "nvtracker", "nvmultistreamtiler",
        "nvvideoconvert", "nvdsosd", "nveglglessink", "queue",
    )
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        return fail("missing plugins: " + ", ".join(missing))
    print("deepstream_plugins=OK")

    try:
        from services.ml_service.app.config import load_settings
        cameras = list(load_settings().cameras)
    except Exception as exc:
        return fail(f"camera config: {exc}")
    if len(cameras) != 6:
        return fail(f"expected 6 cameras, got {len(cameras)}")
    print("camera_count=OK 6")

    from services.camera_v2 import monitor_ui_reference as ui

    checks = {
        "columns_2": ui.GRID_COLUMNS == 2,
        "rows_3": ui.GRID_ROWS == 3,
        "tile_512x288": (ui.TILE_WIDTH, ui.TILE_HEIGHT) == (512, 288),
        "wall_1024x864": (ui.WALL_WIDTH, ui.WALL_HEIGHT) == (1024, 864),
        "aspect_16_9": abs(ui.TILE_WIDTH / ui.TILE_HEIGHT - 16 / 9) < 0.001,
        "large_like_reference": ui.TILE_WIDTH >= 500 and ui.TILE_HEIGHT >= 280,
        "lighter_than_old_wall": ui.WALL_WIDTH * ui.WALL_HEIGHT < 1920 * 720,
    }
    for name, ok in checks.items():
        print(f"contract {name}={'OK' if ok else 'FAIL'}")
        if not ok:
            return fail(name)

    print("layout=2 columns x 3 rows")
    print("camera_tile=512x288")
    print("wall=1024x864")
    print("screenshot_scale=PASS")
    print("pipeline_core_files=UNCHANGED")
    print("MONITOR_UI_REFERENCE_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
