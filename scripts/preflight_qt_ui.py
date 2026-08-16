from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ok = True
    print("CAMERA_QT PREFLIGHT")
    print(f"repo_root={ROOT}")
    print(f"DISPLAY={os.environ.get('DISPLAY') or 'MISSING'}")
    if not os.environ.get("DISPLAY"):
        print("FAIL: DISPLAY is missing; nveglglessink needs a running X server")
        ok = False

    try:
        import PySide6
        print(f"PySide6={PySide6.__version__}")
    except Exception as exc:
        print(f"FAIL: PySide6 import: {exc}")
        ok = False

    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        Gst.init(None)
        print("GstVideoOverlay=OK")
        required = (
            "nvurisrcbin", "nvstreammux", "nvtracker", "nvstreamdemux",
            "nvvideoconvert", "nvdsosd", "nveglglessink", "queue", "tee", "appsink",
        )
        for name in required:
            present = Gst.ElementFactory.find(name) is not None
            print(f"plugin {name}={'OK' if present else 'MISSING'}")
            ok = ok and present
        _ = GstVideo.VideoOverlay
    except Exception as exc:
        print(f"FAIL: GStreamer/GstVideo import: {exc}")
        ok = False

    try:
        from services.camera_v2.qt_heatmap_bridge import ensure_qt_heatmap_bridge
        path = ensure_qt_heatmap_bridge()
        print(f"qt_heatmap_bridge=OK path={path}")
    except Exception as exc:
        print(f"FAIL: Qt heatmap bridge: {exc}")
        ok = False

    for rel in ("services/camera_v2/qt_runtime.py", "services/camera_v2/qt_ui.py"):
        path = ROOT / rel
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            print(f"python_syntax {rel}=OK")
        except Exception as exc:
            print(f"FAIL: python_syntax {rel}: {exc}")
            ok = False

    print("CAMERA_QT_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
