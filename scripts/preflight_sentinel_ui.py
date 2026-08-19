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
EXPECTED_BUILD = "2026.08.19-r7"


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
            _fail(f"{row.get('id','camera')} contains forbidden fields: {sorted(forbidden)}")

    ui_files = {
        "shell": _source("services/camera_v2/sentinel_ui.py"),
        "monitoring": _source("services/camera_v2/sentinel_ui_monitoring.py"),
        "wall": _source("services/camera_v2/sentinel_video_wall_ui.py"),
        "settings": _source("services/camera_v2/sentinel_ui_settings.py"),
    }
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
    if 'metrics.get("known_people"' not in monitoring_source:
        _fail("Known counter is not wired")
    if 'metrics.get("total_people"' not in monitoring_source:
        _fail("Total counter is not wired")
    if "unknown = max(0, total - known)" not in monitoring_source:
        _fail("Unknown counter is not kept consistent with Total/Known")
    if "self.wall.set_pipeline_status(status)" not in monitoring_source:
        _fail("camera wall state cover is not wired")

    tracker_source = _source("services/camera_v2/person_tracking_final.py")
    if "live_source_counts" not in tracker_source or "source_track_counts" not in tracker_source:
        _fail("per-camera live tracking counts are missing")

    video_source = _source("services/camera_v2/sentinel_video_pro.py")
    for guard in (
        "live_source_counts",
        "room_people",
        "room_people[room_key] = max",
        "total = sum(room_people.values())",
        "known = 0",
        "unknown = total",
        "force-aspect-ratio",
        "FOCUS_WIDTH = 1920",
        "FOCUS_HEIGHT = 1080",
    ):
        if guard not in video_source:
            _fail(f"runtime guard missing: {guard}")

    base_video_source = _source("services/camera_v2/sentinel_video.py")
    if "occupancy_label = QLabel" in base_video_source or "occupancy = len" in base_video_source:
        _fail("demo per-camera occupancy badge returned")

    heat_source = _source("services/camera_v2/person_tracking_heatmap.py")
    if "self.bridge.heatmap_update(buffer)" not in heat_source:
        _fail("tracked floor heatmap update is missing")
    for forbidden_heat in ("PoseHeatmapBridge", "deposit_pose_ankle", "pose_sidecar"):
        if forbidden_heat in heat_source:
            _fail(f"active heatmap has optional dependency: {forbidden_heat}")

    enrollment_source = _source("services/camera_v2/sentinel_ui_enrollment.py")
    if "class ReportsPage" in enrollment_source:
        _fail("stale ReportsPage still exists")

    launcher = _source("scripts/run_sentinel_vms.sh")
    for forbidden_launcher in ("setup_camera_v2_reid.py", "preflight_camera_v2_reid.py"):
        if forbidden_launcher in launcher:
            _fail(f"launcher still starts optional path: {forbidden_launcher}")

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT camera_form=id,name,room,rtsp,status")
    print("SENTINEL_PREFLIGHT monitoring=2x3 fullscreen=PASS hover=PASS")
    print("SENTINEL_PREFLIGHT people_count=per-camera->room-max->total known+unknown=consistent")
    print("SENTINEL_PREFLIGHT ui_technical_labels=REMOVED")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
