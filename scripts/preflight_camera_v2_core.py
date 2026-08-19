from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the active runtime module sets the production geometry defaults but
# does not construct/start a GStreamer pipeline.
import services.camera_v2.person_tracking_heatmap  # noqa: F401,E402
from services.camera_v2.detection import INFER_HEIGHT, INFER_WIDTH  # noqa: E402
from services.camera_v2.heatmap_filter import ensure_heatmap_filter  # noqa: E402
from services.camera_v2.native_bridge import ensure_bridge  # noqa: E402
from services.camera_v2.person_tracking_final import CameraPersonTrackingFinal  # noqa: E402
from services.ml_service.app.config import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    camera_count = len(settings.cameras)
    if not 1 <= camera_count <= 6:
        raise RuntimeError(f"camera wall requires 1..6 enabled cameras, got {camera_count}")

    frame_width = int(os.environ.get("CAMERA_V2_FRAME_WIDTH", "0"))
    frame_height = int(os.environ.get("CAMERA_V2_FRAME_HEIGHT", "0"))
    tracker_width = int(os.environ.get("CAMERA_V2_TRACKER_WIDTH", "0"))
    tracker_height = int(os.environ.get("CAMERA_V2_TRACKER_HEIGHT", "0"))

    if (INFER_WIDTH, INFER_HEIGHT) != (736, 416):
        raise RuntimeError(
            f"detector geometry must be 736x416, got {INFER_WIDTH}x{INFER_HEIGHT}"
        )
    if (tracker_width, tracker_height) != (512, 288):
        raise RuntimeError(
            f"tracker geometry must be 512x288, got {tracker_width}x{tracker_height}"
        )
    if (frame_width, frame_height) != (1920, 1080):
        raise RuntimeError(
            f"presentation geometry must be 1920x1080, got {frame_width}x{frame_height}"
        )

    sample_source = inspect.getsource(CameraPersonTrackingFinal._on_infer_sample)
    for required in ("mapped_size", "row_stride", "tight_stride", "CAMERA_INFER_LAYOUT"):
        if required not in sample_source:
            raise RuntimeError(f"stride-safe detector copy is missing: {required}")

    native_bridge_source = (ROOT / "services/camera_v2/native_bridge.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("SMOOTHER_SOURCE", "camera_v2_smooth_display_boxes"):
        if forbidden in native_bridge_source:
            raise RuntimeError(f"legacy display smoother is still active: {forbidden}")

    ensure_bridge()
    ensure_heatmap_filter()

    print(
        f"CAMERA_PREFLIGHT cameras={camera_count} core=PASS heatmap=PASS "
        f"display={frame_width}x{frame_height} detector={INFER_WIDTH}x{INFER_HEIGHT} "
        f"tracker={tracker_width}x{tracker_height} stride_safe=PASS custom_bbox_hold=OFF"
    )
    print("CAMERA_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
