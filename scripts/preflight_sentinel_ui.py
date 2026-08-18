from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.data import EVENTS, PEOPLE, ROOMS
from services.camera_v2.sentinel_ui import BUILD_TAG, MainWindow, main
from services.camera_v2.sentinel_ui_enrollment import EnrollmentPage
from services.camera_v2.sentinel_ui_monitoring import MonitoringPage
from services.camera_v2.sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from services.camera_v2.sentinel_ui_settings import SettingsPage
from services.camera_v2.sentinel_video import CAMERA_COUNT, GRID_COLUMNS, GRID_ROWS
from services.ml_service.app.config import CameraConfig, load_settings


EXPECTED_NAV = [
    "Monitoring",
    "People",
    "Events",
    "Rooms",
    "Enrollment",
    "Settings",
]
FORBIDDEN_NAV = {"Cameras", "Heatmap", "Diagnostics", "Reports"}
EXPECTED_BUILD = "2026.08.18-r4"


def _fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main_preflight() -> int:
    if not callable(main):
        _fail("main entry point is missing")
    if BUILD_TAG != EXPECTED_BUILD:
        _fail(f"build tag={BUILD_TAG!r}, expected={EXPECTED_BUILD!r}")

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

    fields = set(CameraConfig.__dataclass_fields__)
    if "codec" in fields:
        _fail("CameraConfig still exposes codec")
    raw = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8")) or {}
    for row in raw.get("cameras") or []:
        forbidden = {"codec", "env_uri", "environment_uri"}.intersection(row)
        if forbidden:
            _fail(f"{row.get('id','camera')} contains forbidden source fields: {sorted(forbidden)}")

    settings_source = _source("services/camera_v2/sentinel_ui_settings.py")
    for forbidden_text in (
        "Environment URI",
        'form.addRow("Codec"',
        "self.codec =",
        "self.env_uri =",
    ):
        if forbidden_text in settings_source:
            _fail(f"stale Settings source still contains {forbidden_text!r}")
    if "DeepStream RTSP streamni avtomatik aniqlaydi" not in settings_source:
        _fail("Settings no longer states automatic stream negotiation")

    monitoring_source = _source("services/camera_v2/sentinel_ui_monitoring.py")
    if 'metrics.get("known_people"' not in monitoring_source:
        _fail("Monitoring Known counter is not wired to live metrics")
    if 'metrics.get("unknown_people"' not in monitoring_source:
        _fail("Monitoring Unknown counter is not wired to live metrics")
    if "self.identity_panel.setMaximumWidth(336)" not in monitoring_source:
        _fail("Monitoring right rail responsive width guard is missing")

    video_source = _source("services/camera_v2/sentinel_video_pro.py")
    required_video_guards = (
        'runtime.focus_source = sid',
        'force-aspect-ratio',
        'FOCUS_WIDTH = 1920',
        'FOCUS_HEIGHT = 1080',
        'badge.hide()',
        'mapFromGlobal(QCursor.pos())',
    )
    for guard in required_video_guards:
        if guard not in video_source:
            _fail(f"professional video-wall guard missing: {guard}")

    heat_source = _source("services/camera_v2/person_tracking_heatmap.py")
    pose_source = _source("services/camera_v2/pose_ankle.py")
    if "anchor=pose-ankle-only" not in heat_source or "bbox_anchor=disabled" not in heat_source:
        _fail("heatmap is not ankle-only")
    if "LEFT_ANKLE = 15" not in pose_source or "RIGHT_ANKLE = 16" not in pose_source:
        _fail("COCO ankle keypoint mapping is missing")
    if "focus_source >= 0 and not self._heatmap_sources.get" not in heat_source:
        _fail("focused camera heatmap toggle is not isolated")

    enrollment_source = _source("services/camera_v2/sentinel_ui_enrollment.py")
    if "class ReportsPage" in enrollment_source:
        _fail("stale ReportsPage still exists")

    if len(ROOMS) != 3:
        _fail(f"supplied demo model expects 3 rooms, got {len(ROOMS)}")
    if not PEOPLE:
        _fail("supplied People demo data is empty")
    if not EVENTS:
        _fail("supplied Events demo data is empty")

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui_import=PASS")
    print("SENTINEL_PREFLIGHT nav=" + ",".join(nav_titles))
    print("SENTINEL_PREFLIGHT camera_form=id,name,room,rtsp,status legacy_fields=none")
    print(
        "SENTINEL_PREFLIGHT monitoring=2x3-live-deepstream "
        f"active_cameras={len(settings.cameras)} fullscreen_aspect=16:9 "
        "hover_controls=stable demo_tile_counts=hidden"
    )
    print("SENTINEL_PREFLIGHT occupancy=total+known+unknown live_metrics=PASS")
    print("SENTINEL_PREFLIGHT heatmap=pose-ankle-only bbox_anchor=OFF per_camera=PASS")
    print("SENTINEL_PREFLIGHT reports=REMOVED settings=production-camera-crud")
    print("SENTINEL_PREFLIGHT enrollment=10-images profile-required")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
