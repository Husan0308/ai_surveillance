from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CAMERA_V2_QT_UI", "1")


def main() -> int:
    ok = True
    print("SENTINEL_RUNTIME_PREFLIGHT")
    print(f"repo={ROOT}")
    print(f"DISPLAY={os.environ.get('DISPLAY') or 'MISSING'}")
    print(f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM') or 'auto'}")

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
        required = (
            "nvurisrcbin", "nvstreammux", "nvtracker", "nvmultistreamtiler",
            "nvvideoconvert", "nvdsosd", "nveglglessink", "queue", "tee", "appsink",
        )
        for name in required:
            present = Gst.ElementFactory.find(name) is not None
            print(f"plugin {name}={'OK' if present else 'MISSING'}")
            ok = ok and present
        _ = GstVideo.VideoOverlay
        print("GstVideoOverlay=OK")
    except Exception as exc:
        print(f"FAIL GStreamer/DeepStream: {type(exc).__name__}: {exc}")
        ok = False

    for rel in (
        "services/camera_v2/sentinel_exact.py",
        "services/camera_v2/sentinel_runtime.py",
        "services/camera_v2/qt_runtime.py",
        "services/camera_v2/person_heatmap.py",
        "services/camera_v2/native_ui_bridge.c",
        "services/camera_v2/native_heatmap.c",
    ):
        path = ROOT / rel
        if not path.exists():
            print(f"FAIL missing {rel}")
            ok = False
            continue
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
                print(f"syntax {rel}=OK")
            except Exception as exc:
                print(f"FAIL syntax {rel}: {exc}")
                ok = False
        else:
            print(f"source {rel}=OK")

    runtime = None
    try:
        from services.camera_v2.person_heatmap import CameraPersonHeatmap
        runtime = CameraPersonHeatmap()
        print(f"runtime cameras={len(runtime.cameras)}")
        print(f"runtime sources={len(runtime.sources)}")
        print(f"runtime mux={runtime.mux.get_name()}")
        print(f"runtime tracker={runtime.tracker.get_name()} {runtime.tracker_width}x{runtime.tracker_height}")
        print(f"runtime tiler={runtime.tiler.get_name()} rows={runtime.tiler.get_property('rows')} columns={runtime.tiler.get_property('columns')} width={runtime.tiler.get_property('width')} height={runtime.tiler.get_property('height')}")
        print(f"runtime osd={runtime.osd.get_name()}")
        print(f"runtime sink={runtime.sink.get_name()}")
        auth_count = sum(1 for camera in runtime.cameras if getattr(camera, "username", ""))
        print(f"runtime rtsp_auth={auth_count}/{len(runtime.cameras)}")

        checks = [
            (len(runtime.cameras) == 6, "camera_count=6"),
            (len(runtime.sources) == 6, "source_count=6"),
            (runtime.tiler.get_property("rows") == 3, "tiler_rows=3"),
            (runtime.tiler.get_property("columns") == 2, "tiler_columns=2"),
            (runtime.sink.get_factory().get_name() == "nveglglessink", "sink=nveglglessink"),
            (auth_count == 6, "rtsp_auth=6/6"),
        ]
        for passed, label in checks:
            print(f"check {label}={'OK' if passed else 'FAIL'}")
            ok = ok and passed
    except Exception as exc:
        print(f"FAIL runtime_build: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        if runtime is not None:
            try:
                runtime.pipeline.set_state(runtime.Gst.State.NULL)
            except Exception:
                pass

    print("SENTINEL_RUNTIME_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
