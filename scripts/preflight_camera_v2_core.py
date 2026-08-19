from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the active runtime pins the canonical production geometry but does
# not construct/start a GStreamer pipeline.
import services.camera_v2.person_tracking_heatmap  # noqa: F401,E402
from services.camera_v2.detection import INFER_HEIGHT, INFER_WIDTH  # noqa: E402
from services.camera_v2.heatmap_filter import ensure_heatmap_filter  # noqa: E402
from services.camera_v2.native_bridge import ensure_bridge  # noqa: E402
from services.camera_v2.person_tracking_final import CameraPersonTrackingFinal  # noqa: E402
from services.camera_v2.sentinel_video import WALL_HEIGHT, WALL_WIDTH  # noqa: E402
from services.ml_service.app.config import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    camera_count = len(settings.cameras)
    if not 1 <= camera_count <= 6:
        raise RuntimeError(f"camera wall requires 1..6 enabled cameras, got {camera_count}")

    mux_width = int(os.environ.get("CAMERA_V2_FRAME_WIDTH", "0"))
    mux_height = int(os.environ.get("CAMERA_V2_FRAME_HEIGHT", "0"))
    tracker_width = int(os.environ.get("CAMERA_V2_TRACKER_WIDTH", "0"))
    tracker_height = int(os.environ.get("CAMERA_V2_TRACKER_HEIGHT", "0"))

    if (mux_width, mux_height) != (2560, 1440):
        raise RuntimeError(
            f"source-preserving mux must be 2560x1440, got {mux_width}x{mux_height}"
        )
    if (WALL_WIDTH, WALL_HEIGHT) != (1600, 1350):
        raise RuntimeError(
            f"monitoring wall must be 1600x1350 (800x450/tile), got {WALL_WIDTH}x{WALL_HEIGHT}"
        )
    if (INFER_WIDTH, INFER_HEIGHT) != (736, 416):
        raise RuntimeError(
            f"detector geometry must be 736x416, got {INFER_WIDTH}x{INFER_HEIGHT}"
        )
    if (tracker_width, tracker_height) != (512, 288):
        raise RuntimeError(
            f"tracker geometry must be 512x288, got {tracker_width}x{tracker_height}"
        )

    sample_source = inspect.getsource(CameraPersonTrackingFinal._on_infer_sample)
    for required in ("mapped_size", "row_stride", "tight_stride", "CAMERA_INFER_LAYOUT"):
        if required not in sample_source:
            raise RuntimeError(f"stride-safe detector copy is missing: {required}")

    tracker_source = (ROOT / "services/camera_v2/person_tracking_final.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "CAMERA_V2_MAX_DETECT_RESULT_AGE_MS",
        '"interpolation-method", 2',
    ):
        if required not in tracker_source:
            raise RuntimeError(f"fresh detector path guard missing: {required}")

    native_bridge_source = (ROOT / "services/camera_v2/native_bridge.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("SMOOTHER_SOURCE", "camera_v2_smooth_display_boxes"):
        if forbidden in native_bridge_source:
            raise RuntimeError(f"legacy display smoother is still active: {forbidden}")

    # nvtracker is the last geometry-writing component before nvmultistreamtiler.
    # DeepStream therefore already places the current tracker box in rect_params.
    # The label layer is allowed to style borders/text only, never rewrite bbox.
    label_source = (ROOT / "services/camera_v2/native_label_style.c").read_text(
        encoding="utf-8"
    )
    if "sync_rect_from_tracker" in label_source:
        raise RuntimeError("label layer still rewrites NvDCF bbox geometry")
    for forbidden in (
        "obj->rect_params.left = rect->left",
        "obj->rect_params.top = rect->top",
        "obj->rect_params.width = rect->width",
        "obj->rect_params.height = rect->height",
    ):
        if forbidden in label_source:
            raise RuntimeError(f"label layer still mutates bbox geometry: {forbidden}")

    display_source = (ROOT / "services/camera_v2/dynamic_wall.py").read_text(
        encoding="utf-8"
    )
    for required in (
        'self._set_if(self.mux, "interpolation-method", 4)',
        'self._set_if(self.tiler, "interpolation-method", 4)',
        'self._set_if(self.mux, "compute-hw", 1)',
        'self._set_if(self.tiler, "compute-hw", 1)',
        'self._set_if(self.sink, "force-aspect-ratio", True)',
    ):
        if required not in display_source and required not in (
            ROOT / "services/camera_v2/main.py"
        ).read_text(encoding="utf-8"):
            raise RuntimeError(f"display quality/aspect guard missing: {required}")

    focus_source = (ROOT / "services/camera_v2/sentinel_video_pro.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "FOCUS_WIDTH = 1920",
        "FOCUS_HEIGHT = 1080",
        "runtime.set_wall_output_geometry(FOCUS_WIDTH, FOCUS_HEIGHT)",
        'runtime.tiler.set_property("show-source", sid)',
    ):
        if required not in focus_source:
            raise RuntimeError(f"fullscreen focus contract missing: {required}")

    ensure_bridge()
    ensure_heatmap_filter()

    print(
        f"CAMERA_PREFLIGHT cameras={camera_count} core=PASS heatmap=PASS "
        f"mux={mux_width}x{mux_height} grid={WALL_WIDTH}x{WALL_HEIGHT} "
        f"tile={WALL_WIDTH // 2}x{WALL_HEIGHT // 3} focus=1920x1080 "
        f"detector={INFER_WIDTH}x{INFER_HEIGHT} tracker={tracker_width}x{tracker_height} "
        "stride_safe=PASS bbox=tracker-rect-params scaling=lanczos custom_bbox_hold=OFF"
    )
    print("CAMERA_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
