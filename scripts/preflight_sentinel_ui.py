from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.config import CameraConfig, load_settings
from services.camera_v2.sentinel_ui import BUILD_TAG, MainWindow, main
from services.camera_v2.sentinel_ui_enrollment import EnrollmentPage
from services.camera_v2.sentinel_ui_monitoring import MonitoringPage
from services.camera_v2.sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from services.camera_v2.sentinel_ui_settings import SettingsPage
from services.camera_v2.sentinel_video import CAMERA_COUNT, GRID_COLUMNS, GRID_ROWS

EXPECTED_NAV = ["Monitoring", "People", "Events", "Rooms", "Enrollment", "Settings"]
EXPECTED_BUILD = "2026.08.19-r12"


def _fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _require_all(source: str, guards: tuple[str, ...], label: str) -> None:
    for guard in guards:
        if guard not in source:
            _fail(f"{label} guard missing: {guard}")


def main_preflight() -> int:
    if not callable(main):
        _fail("main entry point is missing")
    if BUILD_TAG != EXPECTED_BUILD:
        _fail(f"build tag={BUILD_TAG!r}, expected={EXPECTED_BUILD!r}")

    nav_titles = [str(row[1]) for row in MainWindow.NAV]
    if nav_titles != EXPECTED_NAV:
        _fail(f"navigation={nav_titles!r}, expected={EXPECTED_NAV!r}")

    expected_classes = [
        MonitoringPage,
        PeoplePage,
        EventsPage,
        RoomsPage,
        EnrollmentPage,
        SettingsPage,
    ]
    if [row[3] for row in MainWindow.NAV] != expected_classes:
        _fail("navigation page classes do not match")

    if CAMERA_COUNT != 6 or (GRID_COLUMNS, GRID_ROWS) != (2, 3):
        _fail("Monitoring wall must be 2 columns x 3 rows")

    settings = load_settings()
    if not 1 <= len(settings.cameras) <= CAMERA_COUNT:
        _fail(f"enabled camera count must be 1..{CAMERA_COUNT}")

    fields = set(CameraConfig.__dataclass_fields__)
    if "codec" in fields:
        _fail("CameraConfig still exposes codec")

    raw = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8")) or {}
    for row in raw.get("cameras") or []:
        forbidden = {"codec", "env_uri", "environment_uri"}.intersection(row)
        if forbidden:
            _fail(f"{row.get('id', 'camera')} contains forbidden fields: {sorted(forbidden)}")

    monitoring = _source("services/camera_v2/sentinel_ui_monitoring.py")
    base_video = _source("services/camera_v2/sentinel_video.py")
    wall = _source("services/camera_v2/sentinel_video_wall_ui.py")
    video = _source("services/camera_v2/sentinel_video_pro.py")
    launcher = _source("scripts/run_sentinel_vms.sh")

    _require_all(
        monitoring,
        (
            'QLabel("Estimated people")',
            'metrics.get("total_people"',
            'metrics.get("known_people"',
            "self.wall.nativeReady.connect(self._start_or_bind)",
            "xid = int(self.wall.winId())",
        ),
        "monitoring",
    )

    _require_all(
        base_video,
        (
            "class _GstVideoSurface(QWidget)",
            'self.setObjectName("gstVideoSurface")',
            "self.setAttribute(Qt.WA_NoSystemBackground, True)",
            "self.setAttribute(Qt.WA_NativeWindow, True)",
            "self.video_surface = _GstVideoSurface(self)",
            "def video_window_id(self)",
            "def winId(self)",
            "return self.video_surface.winId()",
            "self.video_surface.lower()",
        ),
        "dedicated EGL surface",
    )

    _require_all(
        wall,
        (
            "class ProPipelineController(_BaseProPipelineController)",
            'self.command_q.put_nowait(("focus", -1))',
            'header.setAttribute(Qt.WA_NativeWindow, True)',
            "cam.setParent(header)",
            "stat.setParent(header)",
            'frame.setAttribute(Qt.WA_NativeWindow, True)',
            "self.fullscreen_camera_label.setAttribute(Qt.WA_NativeWindow, True)",
            "left + width - actions.width() - 8",
            "top + 34",
            "self.video_surface.lower()",
        ),
        "native wall chrome",
    )
    if "top + height - actions.height()" in wall:
        _fail("camera action controls returned to the bottom of the tile")

    _require_all(
        video,
        (
            "GstVideo.VideoOverlay.set_window_handle",
            'runtime.tiler.set_property("rows", GRID_ROWS)',
            'runtime.tiler.set_property("columns", GRID_COLUMNS)',
            'runtime.tiler.set_property("show-source", -1)',
            "runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)",
        ),
        "GStreamer grid",
    )

    _require_all(
        launcher,
        (
            "export QT_QPA_PLATFORM=xcb",
            "export CAMERA_V2_TILER_COLUMNS=2",
            "export CAMERA_V2_WALL_WIDTH=1600",
            "export CAMERA_V2_WALL_HEIGHT=1350",
            "python scripts/preflight_camera_v2_core.py",
            "exec python -m services.camera_v2.monitor_ui",
        ),
        "launcher",
    )

    for removed in (
        "services/api_service",
        "services/frontend",
        "services/ml_service",
        "services/camera_v2/sentinel_app.py",
        "services/camera_v2/monitor_ui_reference.py",
        "services/camera_v2/qwen_reid.py",
        "services/camera_v2/reid_runtime.py",
    ):
        if (ROOT / removed).exists():
            _fail(f"legacy path returned: {removed}")

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT grid=2x3 startup-grid-prime=PASS")
    print("SENTINEL_PREFLIGHT video_surface=dedicated-native-xid PASS")
    print("SENTINEL_PREFLIGHT chrome=native-sibling-no-egl-overpaint PASS")
    print("SENTINEL_PREFLIGHT actions=top-right PASS")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
