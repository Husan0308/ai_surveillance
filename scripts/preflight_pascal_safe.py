#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fail(message: str) -> int:
    print(f"PASCAL_SAFE_PREFLIGHT=FAIL {message}")
    return 1


def main() -> int:
    enabled = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        print("PASCAL_SAFE_PREFLIGHT=SKIP mode=disabled")
        return 0

    module = ROOT / "services/camera_v2/pascal_safe_pipeline.py"
    backend = ROOT / "services/camera_v2/rfdetr_backend.py"
    if not module.exists():
        return fail("missing pascal_safe_pipeline.py")
    if not backend.exists():
        return fail("missing rfdetr_backend.py")

    source = module.read_text(encoding="utf-8")
    backend_source = backend.read_text(encoding="utf-8")

    # Validate the CURRENT safe-mode contract. The old preflight required a
    # literal call to CameraDetectionV2._install_osd_and_meta(self), but that
    # method contains the PyGObject Gst.Element.unlink() boolean bug. Safe mode
    # now owns the OSD relink itself and verifies the real pad peer instead.
    for token in (
        "def _install_osd_without_nvtracker(self)",
        'wall_src = self.wall_queue.get_static_pad("src")',
        "peer = wall_src.get_peer()",
        "wall_src.unlink(peer)",
        "if wall_src.is_linked() or sink_pad.is_linked()",
        "CameraPersonTrackingV2._install_osd_and_meta = _install_osd_without_nvtracker",
        "CameraPersonTrackingFinal._scheduler = CameraDetectionV2._scheduler",
        "CameraPersonTrackingFinal._inject_boxes_probe = _inject_boxes_with_counts",
        "nvtracker=disabled",
        "osd_link=safe-pad-peer",
    ):
        if token not in source:
            return fail(f"missing safe-mode contract: {token}")

    for token in (
        "install_pascal_safe_pipeline",
        "install_pascal_safe_pipeline()",
        "CAMERA_V2_PASCAL_SAFE",
    ):
        if token not in backend_source:
            return fail(f"RF-DETR backend missing Pascal-safe hook: {token}")

    gpu = "unknown"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip().splitlines()[0]
    except Exception:
        pass

    print(
        "PASCAL_SAFE_PREFLIGHT=PASS "
        f"gpu={gpu!r} tracker=motion-predictor nvtracker=disabled "
        "osd_unlink=pad-peer-safe video_path=RTSP-NVDEC-mux-tiler-OSD-EGL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
