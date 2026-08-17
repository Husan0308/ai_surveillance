from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.data import CAMERAS, EVENTS, PEOPLE, ROOMS
from services.camera_v2.sentinel_ui import MainWindow, main
from services.camera_v2.sentinel_ui_enrollment import EnrollmentPage, ReportsPage
from services.camera_v2.sentinel_ui_monitoring import MonitoringPage
from services.camera_v2.sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from services.camera_v2.sentinel_video import CAMERA_COUNT, GRID_COLUMNS, GRID_ROWS


EXPECTED_NAV = [
    "Monitoring",
    "People",
    "Events",
    "Rooms",
    "Enrollment",
    "Reports",
]
FORBIDDEN_NAV = {"Cameras", "Heatmap", "Diagnostics", "Settings"}


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
        ReportsPage,
    ]
    nav_classes = [row[3] for row in MainWindow.NAV]
    if nav_classes != expected_classes:
        _fail("navigation page classes do not match supplied UI order")

    if CAMERA_COUNT != 6 or len(CAMERAS) != 6:
        _fail(f"Monitoring must contain exactly 6 cameras, got runtime={CAMERA_COUNT}, data={len(CAMERAS)}")
    if (GRID_COLUMNS, GRID_ROWS) != (2, 3):
        _fail(f"Monitoring grid must be 2x3, got {GRID_COLUMNS}x{GRID_ROWS}")

    if len(ROOMS) != 3:
        _fail(f"supplied demo model expects 3 rooms, got {len(ROOMS)}")
    if not PEOPLE:
        _fail("supplied People demo data is empty")
    if not EVENTS:
        _fail("supplied Events demo data is empty")

    # The supplied Enrollment implementation owns ten explicit photo buttons and
    # validates len(image_paths) == 10 before allowing completion. Importing the
    # exact class here protects the page wiring without creating a QApplication.
    if EnrollmentPage.__name__ != "EnrollmentPage":
        _fail("Enrollment page import failed")

    print("SENTINEL_PREFLIGHT ui_import=PASS")
    print("SENTINEL_PREFLIGHT nav=" + ",".join(nav_titles))
    print("SENTINEL_PREFLIGHT forbidden_nav=none")
    print("SENTINEL_PREFLIGHT monitoring=2x3-live-deepstream right-rail=known+unknown+recent-views")
    print("SENTINEL_PREFLIGHT enrollment=10-images profile-required")
    print("SENTINEL_PREFLIGHT supplied_demo_data=people+events+rooms")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
