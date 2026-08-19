from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except Exception as exc:
        print(f"ML_PREFLIGHT=FAIL reason=gstreamer-python error={type(exc).__name__}: {exc}")
        return 1

    Gst.init(None)

    required = ["nvurisrcbin", "nvvideoconvert", "appsink"]
    shm_enabled = _env_bool("ML_SHM_VIDEO_ENABLED", True)
    if shm_enabled:
        required.append("shmsink")
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
    shm_dir = os.getenv("ML_SHM_VIDEO_DIR", "/tmp/ai-surveillance")
    print(
        "ML_PREFLIGHT plugins=PASS "
        f"backend=nvurisrcbin cameras={len(settings.cameras)} "
        f"gpu={ds.gpu_id} cudadec_memtype={ds.cudadec_memtype} "
        f"transport={ds.rtsp_transport} latency_ms={ds.latency_ms} "
        f"display={ds.display_width}x{ds.display_height}@{ds.display_fps} "
        f"shm={'on' if shm_enabled else 'off'} shm_dir={shm_dir}",
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
