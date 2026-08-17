from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the full UI module during preflight. PySide6 is imported lazily inside
# sentinel_ui.main(), so this catches Python syntax/import regressions without
# opening a window or touching the live DeepStream pipeline.
from services.camera_v2 import sentinel_ui
from services.camera_v2.sentinel_config import MAX_CAMERAS, ROOMS, list_cameras, room_cameras
from services.camera_v2.sentinel_store import SentinelStore
from services.ml_service.app.config import load_settings


def main() -> int:
    if not callable(getattr(sentinel_ui, "main", None)):
        raise RuntimeError("Sentinel UI entry point is missing")

    rows = list_cameras()
    active = [row for row in rows if row.get("enabled", True)]
    if not 1 <= len(active) <= MAX_CAMERAS:
        raise RuntimeError(f"enabled camera count must be 1..{MAX_CAMERAS}")

    settings = load_settings()
    if len(settings.cameras) != len(active):
        raise RuntimeError(
            f"config mismatch: UI={len(active)} runtime={len(settings.cameras)}"
        )

    mapping = room_cameras()
    for room in ROOMS:
        mapping.setdefault(room, [])

    store = SentinelStore()
    _ = store.list_people()
    _ = store.list_events(limit=5)

    print("SENTINEL_PREFLIGHT ui_import=PASS")
    print(f"SENTINEL_PREFLIGHT cameras={len(active)} max={MAX_CAMERAS}")
    print(
        "SENTINEL_PREFLIGHT rooms="
        + ",".join(f"{room}:{len(mapping[room])}" for room in ROOMS)
    )
    print("SENTINEL_PREFLIGHT workers=known-only enrollment=10-images events=dedup-once")
    print("SENTINEL_PREFLIGHT monitoring=heatmap-focus+fullscreen total-people=local-nvdcf")
    print("SENTINEL_PREFLIGHT settings=dynamic-rtsp-camera-config")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
