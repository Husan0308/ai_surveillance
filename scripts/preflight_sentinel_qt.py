from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ok = True
    print("SENTINEL_QT_PREFLIGHT")
    print(f"repo={ROOT}")
    print(f"DISPLAY={os.environ.get('DISPLAY') or 'MISSING'}")
    if not os.environ.get("DISPLAY"):
        print("FAIL DISPLAY is missing")
        ok = False

    try:
        import PySide6
        print(f"PySide6={PySide6.__version__}")
    except Exception as exc:
        print(f"FAIL PySide6: {exc}")
        ok = False

    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        Gst.init(None)
        for name in ("nvurisrcbin", "nvstreammux", "nvmultistreamtiler", "nvtracker", "nvvideoconvert", "nvdsosd", "nveglglessink"):
            present = Gst.ElementFactory.find(name) is not None
            print(f"plugin {name}={'OK' if present else 'MISSING'}")
            ok = ok and present
        _ = GstVideo.VideoOverlay
        print("GstVideoOverlay=OK")
    except Exception as exc:
        print(f"FAIL GStreamer: {exc}")
        ok = False

    for tool in ("gcc", "pkg-config"):
        present = shutil.which(tool) is not None
        print(f"tool {tool}={'OK' if present else 'MISSING'}")
        ok = ok and present

    for rel in (
        "services/camera_v2/qt_live.py",
        "services/camera_v2/qt_runtime.py",
        "services/camera_v2/person_heatmap.py",
        "services/camera_v2/ui_bridge.py",
    ):
        path = ROOT / rel
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            print(f"syntax {rel}=OK")
        except Exception as exc:
            print(f"FAIL syntax {rel}: {exc}")
            ok = False

    for rel in ("services/camera_v2/native_ui_bridge.c", "services/camera_v2/native_heatmap.c"):
        present = (ROOT / rel).exists()
        print(f"source {rel}={'OK' if present else 'MISSING'}")
        ok = ok and present

    print("SENTINEL_QT_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
