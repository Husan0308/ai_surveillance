from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.sentinel_ui import BUILD_TAG, MainWindow, main
from services.camera_v2.sentinel_ui_enrollment import EnrollmentPage
from services.camera_v2.sentinel_ui_monitoring import MonitoringPage
from services.camera_v2.sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from services.camera_v2.sentinel_ui_settings import SettingsPage
from services.camera_v2.sentinel_video import CAMERA_COUNT, GRID_COLUMNS, GRID_ROWS
from services.ml_service.app.config import CameraConfig, load_settings

EXPECTED_NAV = ["Monitoring", "People", "Events", "Rooms", "Enrollment", "Settings"]
FORBIDDEN_NAV = {"Cameras", "Heatmap", "Diagnostics", "Reports"}
EXPECTED_BUILD = "2026.08.19-r8"


def _fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _require_all(source: str, guards: tuple[str, ...], label: str) -> None:
    for guard in guards:
        if guard not in source:
            _fail(f"{label} guard missing: {guard}")


def main_preflight() -> int:
    """Validate only the Sentinel UI/control contract.

    Detector, NvDCF, native bridge and heatmap runtime validation intentionally
    belongs to scripts/preflight_camera_v2_core.py, which the launcher runs
    immediately after this script. Keeping the two contracts separate prevents
    harmless implementation/string changes in the tracking stack from blocking
    an otherwise valid UI build.
    """
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
    if [row[3] for row in MainWindow.NAV] != expected_classes:
        _fail("navigation page classes do not match")

    if CAMERA_COUNT != 6 or (GRID_COLUMNS, GRID_ROWS) != (2, 3):
        _fail("Monitoring wall must be 2x3 / 6-camera capacity")

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

    ui_files = {
        "shell": _source("services/camera_v2/sentinel_ui.py"),
        "monitoring": _source("services/camera_v2/sentinel_ui_monitoring.py"),
        "wall": _source("services/camera_v2/sentinel_video_wall_ui.py"),
        "settings": _source("services/camera_v2/sentinel_ui_settings.py"),
    }

    # Technical backend names must stay in terminal diagnostics, not user UI.
    for name, source in ui_files.items():
        for forbidden_text in ("NVDEC", "no pose/reid", "pose=", "reid=", "DeepStream"):
            if forbidden_text in source:
                _fail(f"{name} still exposes technical UI text: {forbidden_text}")

    settings_source = ui_files["settings"]
    for forbidden_text in (
        "Environment URI",
        'form.addRow("Codec"',
        "self.codec =",
        "self.env_uri =",
    ):
        if forbidden_text in settings_source:
            _fail(f"stale Settings field remains: {forbidden_text}")

    monitoring_source = ui_files["monitoring"]
    _require_all(
        monitoring_source,
        (
            'metrics.get("known_people"',
            'metrics.get("total_people"',
            "unknown = max(0, total - known)",
            "self.wall.set_pipeline_status(status)",
        ),
        "monitoring",
    )

    # Runtime exposes room-fused people metrics to the UI. Do not assert detector
    # or NvDCF implementation details here; core preflight owns those.
    video_source = _source("services/camera_v2/sentinel_video_pro.py")
    _require_all(
        video_source,
        (
            "live_source_counts",
            "room_people",
            "room_people[room_key] = max",
            "total = sum(room_people.values())",
            "known = 0",
            "unknown = total",
            "force-aspect-ratio",
            "FOCUS_WIDTH = 1280",
            "FOCUS_HEIGHT = 720",
            "runtime.set_wall_output_geometry(FOCUS_WIDTH, FOCUS_HEIGHT)",
            "runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)",
            'runtime.tiler.set_property("rows", 1)',
            'runtime.tiler.set_property("columns", 1)',
        ),
        "monitoring runtime contract",
    )

    dynamic_source = _source("services/camera_v2/dynamic_wall.py")
    _require_all(
        dynamic_source,
        (
            'self.wall_caps = self._make("capsfilter", "camera_v2_wall_geometry")',
            "set_wall_output_geometry",
            "pixel-aspect-ratio=1/1",
            'self._require_link(self.tiler, self.wall_caps',
        ),
        "wall aspect",
    )

    wall_source = ui_files["wall"]
    _require_all(
        wall_source,
        (
            'self.fullscreen_camera_label = QLabel("", self)',
            'self.fullscreen_fps_label = QLabel("", self)',
            "for widget in self.camera_labels:",
            "for widget in self.status_labels:",
            "widget.setVisible(not active)",
            "self.fullscreen_camera_label.setText(camera_id)",
            'self.fullscreen_fps_label.setText("LIVE")',
            "self._layout_fullscreen_hud()",
        ),
        "fullscreen fixed-HUD",
    )

    base_video_source = _source("services/camera_v2/sentinel_video.py")
    if "occupancy_label = QLabel" in base_video_source or "occupancy = len" in base_video_source:
        _fail("demo per-camera occupancy badge returned")

    enrollment_source = _source("services/camera_v2/sentinel_ui_enrollment.py")
    if "class ReportsPage" in enrollment_source:
        _fail("stale ReportsPage still exists")

    launcher = _source("scripts/run_sentinel_vms.sh")
    _require_all(
        launcher,
        (
            "python scripts/preflight_sentinel_ui.py",
            "python scripts/preflight_camera_v2_core.py",
            "exec python -m services.camera_v2.monitor_ui",
        ),
        "launcher",
    )
    for forbidden_launcher in ("setup_camera_v2_reid.py", "preflight_camera_v2_reid.py"):
        if forbidden_launcher in launcher:
            _fail(f"launcher still starts optional path: {forbidden_launcher}")

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT camera_form=id,name,room,rtsp,status")
    print("SENTINEL_PREFLIGHT monitoring=2x3 fullscreen=16:9-renegotiated fixed-hud=PASS")
    print("SENTINEL_PREFLIGHT people_count=room-fused total+known+unknown wiring=PASS")
    print("SENTINEL_PREFLIGHT runtime_validation=delegated-to-camera-v2-core")
    print("SENTINEL_PREFLIGHT ui_technical_labels=REMOVED")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
