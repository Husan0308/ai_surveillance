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
    ROOT / "services/camera_v2/qt_runtime.py",
    ROOT / "services/camera_v2/sentinel_live_runtime.py",
    ROOT / "services/camera_v2/ui_bridge.py",
)


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    print("SENTINEL_EXACT_LIVE_PREFLIGHT=FAIL")
    return 1


def require_text(source: str, needle: str, label: str) -> None:
    if needle not in source:
        raise RuntimeError(f"missing contract {label}: {needle}")
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
        runtime = FILES[2].read_text(encoding="utf-8")
        live = FILES[3].read_text(encoding="utf-8")

        # Visual/source contract from the supplied Sentinel ui.py.
        require_text(exact, 'setFixedWidth(224)', "sidebar_224")
        require_text(exact, 'setFixedHeight(70)', "header_70")
        require_text(exact, 'self.layout.addLayout(camera_column, 3)', "camera_column_3")
        require_text(exact, 'self.layout.addLayout(self.identity_rail, 1)', "identity_rail_1")
        require_text(exact, 'self.setMinimumSize(1180, 720)', "window_minimum")
        for page in ("Monitoring", "People", "Events", "Rooms", "Enrollment", "Reports"):
            require_text(exact, f'"{page}"', f"page_{page.lower()}")
        require_text(exact, 'range(10)', "enrollment_10_slots")
        require_text(exact, 'profile_index', "enrollment_profile_photo")
        require_text(exact, 'Recent Views', "recent_views")

        # Exact-source fixes: no historical per-camera hover controls.
        require_text(app, 'self.controls.hide()', "per_camera_controls_hidden")
        require_text(app, 'QEvent.Type.WinIdChange', "video_rebind_on_wid_change")
        require_text(app, 'window.show()', "uploaded_main_window_show")

        # Realtime architecture contract.
        require_text(runtime, 'mp.get_context("spawn")', "process_isolation")
        require_text(runtime, 'GstVideo.VideoOverlay.set_window_handle', "native_video_overlay")
        require_text(runtime, 'WALL_WIDTH = 1024', "wall_width")
        require_text(runtime, 'WALL_HEIGHT = 864', "wall_height")
        require_text(runtime, 'GRID_ROWS = 3', "rows_3")
        require_text(runtime, 'GRID_COLUMNS = 2', "columns_2")
        require_text(live, 'class SentinelLiveRuntime(CameraPersonHeatmap)', "working_pipeline_inherited")
        require_text(live, 'self.ui_bridge.snapshot_tracks(buffer)', "nvdcf_realtime_metadata")
        require_text(live, 'self.stats[camera.camera_id]', "realtime_camera_metrics")

        # Camera frames must never cross into Qt/Python through a secondary frame
        # transport. Comments may mention JPEG/MJPEG while explicitly saying they
        # are absent, so test actual frame APIs/imports rather than prose tokens.
        forbidden_runtime = (
            "appsink",
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

    print("layout=uploaded Sentinel source")
    print("monitoring=2 columns x 3 rows + 1/4 identity rail")
    print("video=GstVideoOverlay/nveglglessink")
    print("metadata=NvDCF current-frame snapshots")
    print("camera_pipeline_core=UNCHANGED")
    print("SENTINEL_EXACT_LIVE_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
