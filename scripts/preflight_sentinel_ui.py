from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.sentinel_ui import BUILD_TAG, MainWindow, main
from services.camera_v2.sentinel_ui_enrollment import EnrollmentPage
from services.camera_v2.sentinel_ui_monitoring_native import MonitoringPage
from services.camera_v2.sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from services.camera_v2.sentinel_ui_settings import SettingsPage
from services.camera_v2.sentinel_video import CAMERA_COUNT, GRID_COLUMNS, GRID_ROWS
from services.ml_service.app.config import CameraConfig, load_settings

EXPECTED_NAV = ["Monitoring", "People", "Events", "Rooms", "Enrollment", "Settings"]
EXPECTED_BUILD = "2026.08.20-r15-monitoring-recovery"


def _fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _require_all(source: str, guards: tuple[str, ...], label: str) -> None:
    for guard in guards:
        if guard not in source:
            _fail(f"{label} guard missing: {guard}")


def _called_attribute_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(str(node.func.attr))
    return names


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
        _fail("Monitoring must keep a fixed 2x3 / 6-camera wall")

    settings = load_settings()
    if not 1 <= len(settings.cameras) <= CAMERA_COUNT:
        _fail(f"enabled camera count must be 1..{CAMERA_COUNT}")

    fields = set(CameraConfig.__dataclass_fields__)
    if "codec" in fields:
        _fail("CameraConfig still exposes codec")

    raw = yaml.safe_load(
        (ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8")
    ) or {}
    for row in raw.get("cameras") or []:
        forbidden = {"codec", "env_uri", "environment_uri"}.intersection(row)
        if forbidden:
            _fail(
                f"{row.get('id', 'camera')} contains forbidden fields: {sorted(forbidden)}"
            )

    monitoring = _source("services/camera_v2/sentinel_ui_monitoring_native.py")
    _require_all(
        monitoring,
        (
            "class NativeVideoHost(QWidget)",
            "self.video_window = QWindow()",
            "QWidget.createWindowContainer(self.video_window, self)",
            "self.video_window.winId()",
            "self.video_window.installEventFilter(self)",
            "def publish_current_xid(self, *, force: bool = False)",
            "self.surface.publish_current_xid(force=True)",
            "class MonitoringPage(QWidget)",
            "self.surface = NativeVideoHost(self.wall_card)",
            "self.surface.nativeReady.connect(self._start_or_bind)",
            "ProPipelineController()",
            'QLabel("People in Building")',
            "class CameraStatusRow(QFrame)",
            'metrics.get("total_people"',
            'metrics.get("known_people"',
            "def open_fullscreen_grid",
            "def exit_fullscreen",
            "def shutdown",
        ),
        "QWindow Monitoring",
    )

    monitoring_calls = _called_attribute_names(monitoring)
    forbidden_monitoring_calls = {"focus", "set_fullscreen_mode"}.intersection(
        monitoring_calls
    )
    if forbidden_monitoring_calls:
        _fail(
            "Monitoring reintroduced dynamic tiler/focus calls: "
            + ",".join(sorted(forbidden_monitoring_calls))
        )
    if "ProLiveVideoWall(" in monitoring:
        _fail("Monitoring reintroduced the legacy native wall widget")

    shell = _source("services/camera_v2/sentinel_ui.py")
    _require_all(
        shell,
        (
            "from .sentinel_ui_monitoring_native import MonitoringPage",
            "BUILD_TAG = \"2026.08.20-r15-monitoring-recovery\"",
            "self.monitoring_host = QWidget(self.content)",
            "self.monitoring_page = MonitoringPage()",
            "monitoring_layout.addWidget(self.monitoring_page, 1)",
            "self.pages = [self.monitoring_page]",
            "for _, _, _, klass in self.NAV[1:]",
            "self.stack.hide()",
            "self.monitoring_host.hide()",
            "self.stack.setCurrentIndex(index - 1)",
            "self.monitoring_host.show()",
            "window.showMaximized()",
            "def set_monitoring_fullscreen",
        ),
        "native-safe shell",
    )

    # The native QWindow must never be inserted into QStackedWidget. Only the
    # ordinary Qt pages from NAV[1:] may be added to the stack.
    if "self.stack.addWidget(self.monitoring_page)" in shell:
        _fail("Monitoring QWindow was reinserted into QStackedWidget")
    if "for _, _, _, klass in self.NAV:" in shell:
        _fail("shell again stacks Monitoring with ordinary pages")

    shell_calls = _called_attribute_names(shell)
    forbidden_window_calls = {"showFullScreen", "showNormal"}.intersection(shell_calls)
    if forbidden_window_calls:
        _fail(
            "top-level window mode churn returned: "
            + ",".join(sorted(forbidden_window_calls))
        )

    video = _source("services/camera_v2/sentinel_video_pro.py")
    _require_all(
        video,
        (
            'runtime.tiler.set_property("rows", GRID_ROWS)',
            'runtime.tiler.set_property("columns", GRID_COLUMNS)',
            "runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)",
            "GstVideo.VideoOverlay.set_window_handle",
            "GstVideo.VideoOverlay.expose",
            "live_source_counts",
            "room_people[room_key] = max",
            "total = sum(room_people.values())",
        ),
        "fixed-grid runtime",
    )

    launcher = _source("scripts/run_sentinel_vms.sh")
    _require_all(
        launcher,
        (
            "python scripts/preflight_rfdetr_core.py",
            "python scripts/preflight_sentinel_ui.py",
            "python scripts/preflight_camera_v2_core.py",
            "exec python -m services.camera_v2.monitor_ui",
            "export QT_QPA_PLATFORM=xcb",
        ),
        "launcher",
    )

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT shell=Monitoring-outside-QStackedWidget PASS")
    print("SENTINEL_PREFLIGHT monitoring=fixed-2x3 qwindow-container right-rail=270px")
    print("SENTINEL_PREFLIGHT native_video=QWindow+createWindowContainer one-xid PASS")
    print("SENTINEL_PREFLIGHT overlays=outside-native-video PASS")
    print("SENTINEL_PREFLIGHT tiler=runtime-fixed no-focus-mutation PASS")
    print("SENTINEL_PREFLIGHT people_count=room-fused total+known+unknown wiring=PASS")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
