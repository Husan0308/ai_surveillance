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

    tracker_source = _source("services/camera_v2/person_tracking_final.py")
    _require_all(
        tracker_source,
        (
            "live_source_counts",
            "source_track_counts",
            'CAMERA_V2_DETECT_WIDTH", "736"',
            'CAMERA_V2_DETECT_HEIGHT", "416"',
            'CAMERA_V2_MIN_DISPLAY_TRACK_CONF", "0.28"',
            '"maxShadowTrackingAge": "12"',
            '"minTrackingConfidenceDuringInactive": "0.40"',
            "Always publish the detector decision, including an empty list",
        ),
        "current NvDCF tracking",
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
        "dynamic wall aspect",
    )

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
        "runtime",
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

    heat_source = _source("services/camera_v2/person_tracking_heatmap.py")
    if "self.bridge.heatmap_update(buffer)" not in heat_source:
        _fail("tracked floor heatmap update is missing")
    for forbidden_heat in ("PoseHeatmapBridge", "deposit_pose_ankle", "pose_sidecar"):
        if forbidden_heat in heat_source:
            _fail(f"active heatmap has optional dependency: {forbidden_heat}")

    # The old downstream smoother generated held/predicted rectangles after NvDCF.
    # Production must now use the current NvDCF bbox only.
    smoother_source = _source("services/camera_v2/native_display_smoother.c")
    _require_all(
        smoother_source,
        (
            "Intentionally a no-op",
            "do not create, move, resize, or hold any metadata",
            "return buffer_ptr ? 0 : -1",
        ),
        "NvDCF-only display bbox",
    )
    for forbidden_smoother in (
        "DISPLAY_HOLD_FRAMES",
        "add_display_hold",
        "display_cx",
        "display_w",
    ):
        if forbidden_smoother in smoother_source:
            _fail(f"custom display predictor returned: {forbidden_smoother}")

    meta_source = _source("services/camera_v2/native_meta_bridge.c")
    _require_all(
        meta_source,
        (
            'CAMERA_V2_MIN_DISPLAY_TRACK_CONF", 0.28f',
            "obj->class_id = -1",
            "obj->unique_component_id == 191",
            "frame_meta->pipeline_width > 1",
        ),
        "tracker metadata filter",
    )

    label_source = _source("services/camera_v2/native_label_style.c")
    _require_all(
        label_source,
        (
            "should_style_track",
            "obj->unique_component_id == 191",
            "obj->rect_params.border_width == 0",
        ),
        "track label filter",
    )

    profile_source = _source("services/camera_v2/tracker_profile.py")
    _require_all(
        profile_source,
        (
            '"minTrackerConfidence": "0.28"',
            '"maxShadowTrackingAge": "12"',
            '"earlyTerminationAge": "1"',
            '"minTrackingConfidenceDuringInactive": "0.40"',
            '"outputShadowTracks", "0"',
        ),
        "NvDCF stale-track profile",
    )

    latency_source = _source("services/camera_v2/detector_latency.py")
    _require_all(
        latency_source,
        (
            "self.max_projection_s = 0.16",
            "self.projection_gain = 0.62",
            "geometry_stable",
            "motion_plausible",
        ),
        "conservative detector projection",
    )

    enrollment_source = _source("services/camera_v2/sentinel_ui_enrollment.py")
    if "class ReportsPage" in enrollment_source:
        _fail("stale ReportsPage still exists")

    launcher = _source("scripts/run_sentinel_vms.sh")
    for forbidden_launcher in ("setup_camera_v2_reid.py", "preflight_camera_v2_reid.py"):
        if forbidden_launcher in launcher:
            _fail(f"launcher still starts optional path: {forbidden_launcher}")

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT camera_form=id,name,room,rtsp,status")
    print("SENTINEL_PREFLIGHT monitoring=2x3 fullscreen=16:9-renegotiated fixed-hud=PASS")
    print("SENTINEL_PREFLIGHT bbox=current-nvdcf-only custom_hold=OFF tracker_conf=0.28 shadow_age=12")
    print("SENTINEL_PREFLIGHT detector=736x416 empty_results=authoritative latency_projection=conservative")
    print("SENTINEL_PREFLIGHT people_count=visible-tracks->room-max->total known+unknown=consistent")
    print("SENTINEL_PREFLIGHT ui_technical_labels=REMOVED")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
