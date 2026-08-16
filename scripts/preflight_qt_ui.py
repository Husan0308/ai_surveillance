from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ok = True
    print("CAMERA_QT_V2 PREFLIGHT")
    print(f"repo_root={ROOT}")
    print(f"DISPLAY={os.environ.get('DISPLAY') or 'MISSING'}")
    print(f"XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE') or 'unknown'}")
    print(f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM') or 'auto'}")
    if not os.environ.get("DISPLAY"):
        print("FAIL: DISPLAY is missing; dGPU nveglglessink needs a running X server")
        ok = False

    try:
        import PySide6
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QWidget

        print(f"PySide6={PySide6.__version__}")
        app = QApplication.instance() or QApplication(["camera-qt-preflight"])
        probe = QWidget()
        probe.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        probe.createWinId()
        wid = int(probe.winId())
        print(f"qt_native_wid={'OK' if wid > 0 else 'FAIL'} value={wid}")
        if wid <= 0:
            ok = False
        probe.close()
    except Exception as exc:
        print(f"FAIL: PySide6/native widget: {exc}")
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

    for rel in (
        "services/camera_v2/qt_runtime.py",
        "services/camera_v2/qt_runtime_v2.py",
        "services/camera_v2/qt_app.py",
        "services/camera_v2/qt_ui.py",
    ):
        path = ROOT / rel
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            print(f"python_syntax {rel}=OK")
        except Exception as exc:
            print(f"FAIL: python_syntax {rel}: {exc}")
            ok = False

    # Structural smoke test: build the exact runtime graph without PLAYING it.
    # No RTSP connection is opened here, but constructor/link/pad errors are caught.
    runtime = None
    try:
        from services.camera_v2.qt_runtime_v2 import CameraQtRuntimeV2

        runtime = CameraQtRuntimeV2()
        sink_count = len(runtime.camera_sinks)
        osd_count = len(runtime.camera_osds)
        demux_pad_count = len(runtime.demux_request_pads)
        print(f"graph camera_sinks={sink_count}/6")
        print(f"graph camera_osds={osd_count}/6")
        print(f"graph demux_pads={demux_pad_count}/6")
        if sink_count != 6 or osd_count != 6 or demux_pad_count != 6:
            print("FAIL: Qt graph did not build six complete display branches")
            ok = False
        else:
            print("qt_graph_build=OK")
    except Exception as exc:
        print(f"FAIL: Qt runtime graph build: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        if runtime is not None:
            try:
                runtime.pipeline.set_state(runtime.Gst.State.NULL)
            except Exception:
                pass
            try:
                runtime.release_qt_pads()
            except Exception:
                pass
            for tee, pad in getattr(runtime, "tee_request_pads", []):
                try:
                    tee.release_request_pad(pad)
                except Exception:
                    pass
            for pad in getattr(runtime, "_request_pads", []):
                try:
                    runtime.mux.release_request_pad(pad)
                except Exception:
                    pass

    print("CAMERA_QT_V2_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
