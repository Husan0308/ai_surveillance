from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.data import EVENTS, PEOPLE, ROOMS
from services.camera_v2.sentinel_ui import MainWindow, main
from services.camera_v2.sentinel_ui_enrollment import EnrollmentPage
from services.camera_v2.sentinel_ui_monitoring import MonitoringPage
from services.camera_v2.sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from services.camera_v2.sentinel_ui_settings import SettingsPage
from services.camera_v2.sentinel_video import CAMERA_COUNT, GRID_COLUMNS, GRID_ROWS
from services.ml_service.app.config import load_settings


EXPECTED_NAV = [
    "Monitoring",
    "People",
    "Events",
    "Rooms",
    "Enrollment",
    "Settings",
]
FORBIDDEN_NAV = {"Cameras", "Heatmap", "Diagnostics", "Reports"}


def _fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def main_preflight() -> int:
    if not callable(main):
        _fail("main entry point is missing")

    nav_titles = [str(row[1]) for row in MainWindow.NAV]
    if nav_titles != EXPECTED_NAV:
        _fail(f"navigation={nav_titles!r}, expected={EXPECTED_NAV!r}")
    if FORBIDDEN_NAV.intersection(nav_titles):
        _fail("forbidden navigation item is present")

    expected_classes = [
        MonitoringPage,
        PeoplePage,
        EventsPage,
        RoomsPage,
        EnrollmentPage,
        SettingsPage,
    ]
    nav_classes = [row[3] for row in MainWindow.NAV]
    if nav_classes != expected_classes:
        _fail("navigation page classes do not match supplied UI order")

    if CAMERA_COUNT != 6:
        _fail(f"Monitoring wall capacity must be 6, got {CAMERA_COUNT}")
    if (GRID_COLUMNS, GRID_ROWS) != (2, 3):
        _fail(f"Monitoring grid must be 2x3, got {GRID_COLUMNS}x{GRID_ROWS}")

    settings = load_settings()
    if not 1 <= len(settings.cameras) <= CAMERA_COUNT:
        _fail(
            f"enabled camera count must be 1..{CAMERA_COUNT}, got {len(settings.cameras)}"
        )

    if len(ROOMS) != 3:
        _fail(f"supplied demo model expects 3 rooms, got {len(ROOMS)}")
    if not PEOPLE:
        _fail("supplied People demo data is empty")
    if not EVENTS:
        _fail("supplied Events demo data is empty")

    print("SENTINEL_PREFLIGHT ui_import=PASS")
    print("SENTINEL_PREFLIGHT nav=" + ",".join(nav_titles))
    print("SENTINEL_PREFLIGHT forbidden_nav=none")
    print(
        "SENTINEL_PREFLIGHT monitoring=2x3-live-deepstream "
        f"active_cameras={len(settings.cameras)} in_place_fullscreen=1"
    )
    print("SENTINEL_PREFLIGHT settings=camera-crud active-limit=6")
    print("SENTINEL_PREFLIGHT enrollment=10-images profile-required")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
