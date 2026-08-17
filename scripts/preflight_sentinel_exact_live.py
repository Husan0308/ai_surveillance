#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FILES = (
    ROOT / "services/camera_v2/sentinel_exact.py",
    ROOT / "services/camera_v2/sentinel_app.py",
    ROOT / "services/camera_v2/safe_live_wall.py",
    ROOT / "services/camera_v2/qt_runtime.py",
    ROOT / "services/camera_v2/sentinel_live_runtime.py",
    ROOT / "services/camera_v2/ui_bridge.py",
    ROOT / "services/camera_v2/person_tracking_final.py",
)


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    print("SENTINEL_EXACT_LIVE_PREFLIGHT=FAIL")
    return 1


def require_text(source: str, needle: str, label: str) -> None:
    if needle not in source:
        raise RuntimeError(f"missing contract {label}: {needle}")
    print(f"contract {label}=OK")


def forbid_text(source: str, needle: str, label: str) -> None:
    if needle in source:
        raise RuntimeError(f"forbidden contract {label}: {needle}")
    print(f"contract {label}=OK")


def main() -> int:
    try:
        for path in FILES:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            print(f"syntax {path.name}=OK")
    except Exception as exc:
        return fail(f"Python syntax: {exc}")

    try:
        import PySide6
        version = tuple(int(part) for part in PySide6.__version__.split(".")[:2])
        if version < (6, 7) or version >= (7, 0):
            return fail(f"PySide6 version {PySide6.__version__}; required >=6.7,<7")
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
        return fail(f"GStreamer/GstVideo unavailable: {exc}")
    print("GstVideoOverlay=OK")

    required_plugins = (
        "nvurisrcbin", "nvstreammux", "nvtracker", "nvmultistreamtiler",
        "nvvideoconvert", "nvdsosd", "nveglglessink", "queue",
    )
    missing = [name for name in required_plugins if Gst.ElementFactory.find(name) is None]
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

    try:
        from services.camera_v2.ui_bridge import ensure_ui_bridge
        bridge = ensure_ui_bridge()
        print(f"ui_meta_bridge=OK {bridge}")
    except Exception as exc:
        return fail(f"native UI metadata bridge: {exc}")

    try:
        exact = FILES[0].read_text(encoding="utf-8")
        app = FILES[1].read_text(encoding="utf-8")
        safe = FILES[2].read_text(encoding="utf-8")
        runtime = FILES[3].read_text(encoding="utf-8")
        live = FILES[4].read_text(encoding="utf-8")
        tracking = FILES[6].read_text(encoding="utf-8")

        # Supplied Sentinel visual contract stays the shell of the application.
        require_text(exact, 'setFixedWidth(224)', "sidebar_224")
        require_text(exact, 'setFixedHeight(70)', "header_70")
        require_text(exact, 'self.layout.addLayout(camera_column, 3)', "camera_column_3")
        require_text(exact, 'self.layout.addLayout(self.identity_rail, 1)', "identity_rail_1")
        require_text(exact, 'StatCard("Total"', "total_metric")
        require_text(exact, 'StatCard("Known"', "known_metric")
        require_text(exact, 'StatCard("Unknown"', "unknown_metric")
        require_text(exact, 'Recent Views', "recent_views")
        require_text(exact, 'range(10)', "enrollment_10_slots")
        require_text(exact, 'profile_index', "enrollment_profile_photo")
        require_text(exact, 'self.setMinimumSize(1180, 720)', "window_minimum")

        # Native-video integration: normal child backing stores must never be
        # composited directly on the WA_PaintOnScreen GstVideoOverlay target.
        require_text(app, 'ui.LiveWall = SafeLiveWall', "safe_live_wall_installed")
        require_text(safe, 'class SafeLiveWall(QWidget)', "safe_live_wall")
        require_text(safe, 'Qt.WindowType.Tool', "separate_overlay_window")
        require_text(safe, 'WA_TranslucentBackground', "translucent_overlay")
        require_text(safe, 'WA_PaintOnScreen', "native_video_surface")
        require_text(safe, 'def paintEngine(self):', "native_paint_engine_disabled")
        require_text(safe, 'GRID_ASPECT = 1280.0 / 1080.0', "grid_aspect")
        require_text(safe, 'FOCUS_ASPECT = 16.0 / 9.0', "focus_aspect")
        require_text(safe, 'self._live_timer.setInterval(50)', "overlay_20hz")
        require_text(safe, 'self.fullscreenRequested.emit', "camera_fullscreen")
        require_text(safe, 'self._hover_source', "camera_hover")
        forbid_text(safe, 'CameraTile(', "no_child_camera_backing_store")

        # Realtime architecture and aspect-ratio contract.
        require_text(runtime, 'mp.get_context("spawn")', "process_isolation")
        require_text(runtime, 'GstVideo.VideoOverlay.set_window_handle', "native_video_overlay")
        require_text(runtime, 'WALL_WIDTH = 1280', "wall_width")
        require_text(runtime, 'WALL_HEIGHT = 1080', "wall_height")
        require_text(runtime, 'FOCUS_HEIGHT = 720', "focus_16_9")
        require_text(runtime, 'GRID_ROWS = 3', "rows_3")
        require_text(runtime, 'GRID_COLUMNS = 2', "columns_2")
        require_text(runtime, 'runtime.GLib.timeout_add(50, publish_snapshot)', "metadata_publish_20hz")
        require_text(live, 'class SentinelLiveRuntime(CameraPersonHeatmap)', "working_pipeline_inherited")
        require_text(live, 'self.ui_bridge.snapshot_tracks(buffer)', "nvdcf_realtime_metadata")
        require_text(live, 'self._set_if(self.sink, "force-aspect-ratio", True)', "no_camera_stretch")
        require_text(live, 'self._visible_grace = 0.30', "no_long_stale_bbox")
        require_text(live, 'self.stats[camera.camera_id]', "realtime_camera_metrics")

        # Detector/tracker latency contract.
        require_text(tracking, 'CAMERA_V2_DETECT_HEIGHT", "360"', "detector_16_9_input")
        require_text(tracking, 'CAMERA_V2_DETECT_TARGET_HZ", "3.8"', "detector_target_rate")
        require_text(tracking, 'CAMERA_V2_TRACKER_WIDTH", "576"', "tracker_width")
        require_text(tracking, 'if not prepared:', "skip_single_empty_detector_miss")

        # Camera frames stay zero-copy/native across the Qt boundary.
        forbidden_runtime = (
            "cv2.",
            "QImage(",
            "QPixmap(",
            "requests.get",
            "StreamingResponse",
            "imencode(",
            "mmap.mmap(",
        )
        runtime_combo = runtime + "\n" + live
        for token in forbidden_runtime:
            if token in runtime_combo:
                return fail(f"forbidden camera hot-path transport found: {token}")
        print("hot_path_frame_transport=ZERO_COPY_NATIVE")
    except Exception as exc:
        return fail(str(exc))

    print("layout=uploaded Sentinel shell + realtime native wall")
    print("monitoring=2 columns x 3 rows + Total/Known/Unknown + Recent Views")
    print("camera_tile=640x360 16:9")
    print("video=GstVideoOverlay/nveglglessink force-aspect-ratio=1")
    print("overlay=20Hz fresh NvDCF metadata; stale bbox hold capped at 300ms")
    print("detector=640x360 target~3.8Hz/camera with adaptive wall governor")
    print("SENTINEL_EXACT_LIVE_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())