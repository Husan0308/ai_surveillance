from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "services" / "frontend" / "core_v1"
ML = ROOT / "services" / "ml_service" / "core_v1"
EXPECTED_SHA256 = "65487172d3e63f96dbd59539c6da1cf050002b77a43f60799ed651b5fd65518e"


def main() -> int:
    ok = True
    print("SENTINEL_REALTIME_UI PREFLIGHT")
    print(f"repo_root={ROOT}")

    try:
        import PySide6
        print(f"PySide6={PySide6.__version__}")
    except Exception as exc:
        print(f"FAIL PySide6: {exc}")
        ok = False

    try:
        encoded = "".join(
            (CORE / f"sentinel_ui_bundle.b64.{index:02d}").read_text(encoding="ascii").strip()
            for index in range(5)
        )
        archive = base64.b64decode(encoded, validate=True)
        digest = hashlib.sha256(archive).hexdigest()
        print(f"ui_bundle_sha256={digest}")
        if digest != EXPECTED_SHA256:
            print(f"FAIL bundle hash expected={EXPECTED_SHA256}")
            ok = False
        else:
            print("ui_bundle_integrity=OK")

        import io
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            source = bundle.read("sentinel_live.py").decode("utf-8")
            compile(source, "sentinel_live.py", "exec")
            print("sentinel_live_syntax=OK")
            required = (
                "class LiveCameraView",
                "class MonitoringPage",
                "class PeoplePage",
                "class EventsPage",
                "class RoomsPage",
                "class EnrollmentPage",
                "class ReportsPage",
                "/faces/enrollment/files",
                "showFullScreen",
                "WA_Hover",
                "SmoothFrameReader",
            )
            missing = [token for token in required if token not in source]
            if missing:
                print(f"FAIL UI features missing={missing}")
                ok = False
            else:
                print("ui_feature_contract=OK")
    except Exception as exc:
        print(f"FAIL UI bundle: {type(exc).__name__}: {exc}")
        ok = False

    try:
        ml_main = (ML / "main.py").read_text(encoding="utf-8")
        sentinel_app = (ML / "sentinel_app.py").read_text(encoding="utf-8")
        tracking_mmap = (ML / "sentinel_tracking_mmap_publisher.py").read_text(encoding="utf-8")
        reader = (CORE / "smooth_frame_reader.py").read_text(encoding="utf-8")
        mmap_reader = (CORE / "mmap_frame_reader.py").read_text(encoding="utf-8")

        for name, source in (
            ("services/ml_service/core_v1/main.py", ml_main),
            ("services/ml_service/core_v1/sentinel_app.py", sentinel_app),
            ("services/ml_service/core_v1/sentinel_tracking_mmap_publisher.py", tracking_mmap),
            ("services/frontend/core_v1/smooth_frame_reader.py", reader),
            ("services/frontend/core_v1/mmap_frame_reader.py", mmap_reader),
        ):
            compile(source, name, "exec")

        if "from .sentinel_app import app, core_cfg" not in ml_main:
            raise RuntimeError("ML main is not using the full Sentinel app")
        if '@app.post("/faces/enrollment/files")' not in sentinel_app:
            raise RuntimeError("file enrollment endpoint is missing")
        if "TrackingMmapPublisher" not in sentinel_app:
            raise RuntimeError("Sentinel backend is not replacing publishers with tracking mmap")
        if "class TrackingMmapPublisher(TrackingJpegPublisher)" not in tracking_mmap:
            raise RuntimeError("ownership-tracking mmap publisher is missing")
        if "SmoothMmapFrameReader" not in reader:
            raise RuntimeError("Sentinel SmoothFrameReader is not routed through mmap")

        print("ml_entrypoint=FULL_STACK")
        print("file_enrollment_endpoint=OK")
        print("camera_transport=TRACKING_MMAP_LATEST_ONLY")
        print("frontend_reader=MMAP_DECODE_FREE")
    except Exception as exc:
        print(f"FAIL backend/frontend wiring: {type(exc).__name__}: {exc}")
        ok = False

    print("SENTINEL_REALTIME_UI_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
