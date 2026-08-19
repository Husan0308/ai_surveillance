from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except Exception as exc:
        print(f"ML_PREFLIGHT=FAIL reason=gstreamer-python error={type(exc).__name__}: {exc}")
        return 1

    Gst.init(None)

    required = ("nvurisrcbin", "nvvideoconvert", "appsink")
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        print(f"ML_PREFLIGHT=FAIL reason=missing-plugins plugins={','.join(missing)}")
        return 1

    try:
        from services.ml_service.app.config import load_settings

        settings = load_settings()
    except Exception as exc:
        print(f"ML_PREFLIGHT=FAIL reason=config error={type(exc).__name__}: {exc}")
        return 1

    ds = settings.deepstream
    print(
        "ML_PREFLIGHT plugins=PASS "
        f"backend=nvurisrcbin cameras={len(settings.cameras)} "
        f"gpu={ds.gpu_id} cudadec_memtype={ds.cudadec_memtype} "
        f"transport={ds.rtsp_transport} latency_ms={ds.latency_ms} "
        f"display={ds.display_width}x{ds.display_height}@{ds.display_fps}",
        flush=True,
    )
    for camera in settings.cameras:
        print(
            f"ML_PREFLIGHT camera={camera.camera_id} room={camera.room or '-'} "
            f"auth={'yes' if camera.username else 'no'} uri={camera.uri}",
            flush=True,
        )

    print("ML_PREFLIGHT=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
