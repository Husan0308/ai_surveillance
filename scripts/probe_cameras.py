from __future__ import annotations

from pathlib import Path
import sys
import time

# When this file is executed as `python scripts/probe_cameras.py`, Python puts
# the scripts/ directory on sys.path, not the repository root. Add the project
# root explicitly so `services.*` imports work from a fresh clone without
# requiring a global PYTHONPATH setting.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.ml_service.app.config import load_settings
from services.ml_service.app.deepstream.capture import DeepStreamCapture


def main() -> int:
    settings = load_settings()
    failures = 0

    print("=== RTSP/NVDEC one-frame probe ===", flush=True)
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print("Each camera is tested sequentially so the NVR is not connection-stormed.\n", flush=True)

    for camera in settings.cameras:
        cap = None
        started = time.monotonic()
        try:
            print(
                f"[PROBE] {camera.camera_id} {camera.uri} "
                f"codec={camera.codec} transport={settings.deepstream.rtsp_transport}",
                flush=True,
            )
            cap = DeepStreamCapture(
                camera.camera_id,
                camera.uri,
                camera.codec,
                settings.deepstream,
                transport=settings.deepstream.rtsp_transport,
            )
            deadline = time.monotonic() + settings.deepstream.startup_grace_sec
            frame = None
            while time.monotonic() < deadline:
                ok, image = cap.read()
                if ok and image is not None:
                    frame = image
                    break
            if frame is None:
                raise RuntimeError("no decoded frame before startup deadline")
            h, w = frame.shape[:2]
            elapsed = time.monotonic() - started
            print(f"[PASS] {camera.camera_id} first_frame={w}x{h} in {elapsed:.2f}s", flush=True)
        except Exception as exc:
            failures += 1
            detail = {}
            if cap is not None:
                try:
                    detail = cap.debug_info()
                except Exception:
                    pass
            print(f"[FAIL] {camera.camera_id}: {exc}", flush=True)
            if detail:
                print(f"       detail={detail}", flush=True)
        finally:
            if cap is not None:
                try:
                    cap.close()
                except Exception:
                    pass
            time.sleep(0.5)

    passed = len(settings.cameras) - failures
    print(f"\nRESULT: {passed}/{len(settings.cameras)} cameras passed", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
