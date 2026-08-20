#!/usr/bin/env python3
from __future__ import annotations

import ast
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


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def imports_module(text: str, suffix: str) -> bool:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if str(node.module).endswith(suffix):
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if str(alias.name).endswith(suffix):
                    return True
    return False


def main() -> int:
    enabled = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        print("PASCAL_SAFE_PREFLIGHT=SKIP mode=disabled")
        return 0

    runtime_path = ROOT / "services/camera_v2/pascal_safe_pipeline.py"
    controller_path = ROOT / "services/camera_v2/camera_wall_runtime.py"
    if not runtime_path.exists():
        return fail("missing pascal_safe_pipeline.py")
    if not controller_path.exists():
        return fail("missing camera_wall_runtime.py")

    runtime = runtime_path.read_text(encoding="utf-8")
    controller = controller_path.read_text(encoding="utf-8")
    ui = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    launcher = source("scripts/run_sentinel_vms.sh")

    for token in (
        "class CameraPascalSafeRuntime(CameraDetectionV2)",
        "def _on_infer_sample(self, sink, cid: str)",
        "mapped_size",
        "row_stride",
        "tight_stride",
        "CAMERA_INFER_LAYOUT",
        "def _install_osd_and_meta(self)",
        "self.wall_queue.unlink(self.sink)",
        "if queue_src.is_linked()",
        "pascal_wall_convert",
        "pascal_osd",
        "CAMERA_PASCAL_SAFE",
        "nvtracker=disabled",
        "tracker=motion-predictor",
        "safe_mux_batches",
        "safe_wall_frames",
    ):
        if token not in runtime:
            return fail(f"missing runtime contract: {token}")

    for forbidden in (
        "CameraPersonTrackingV2",
        "CameraPersonTrackingFinal",
        "libnvds_nvmultiobjecttracker",
        "config_tracker_NvDCF",
    ):
        if forbidden in runtime:
            return fail(f"Pascal runtime still depends on NvDCF path: {forbidden}")
    if imports_module(runtime, "person_tracking") or imports_module(runtime, "person_tracking_final"):
        return fail("Pascal runtime imports a tracker module")

    for token in (
        "from .pascal_safe_pipeline import CameraPascalSafeRuntime",
        "runtime = CameraPascalSafeRuntime()",
        "bound_xid = 0",
        "if target == bound_xid",
        "GstVideo.VideoOverlay.set_window_handle",
        "runtime.bus.set_sync_handler",
        "GRID_COLUMNS = 2",
        "GRID_ROWS = 3",
    ):
        if token not in controller:
            return fail(f"missing camera-wall controller contract: {token}")

    if "from .camera_wall_runtime import CameraWallController" not in ui:
        return fail("UI is not routed through camera_wall_runtime")
    if "ProPipelineController" in ui:
        return fail("legacy ProPipelineController is still active in camera-only UI")

    for token in (
        "export CAMERA_V2_PASCAL_SAFE=1",
        "export CAMERA_V2_BOX_MAX_AGE=1.6",
        "tracker=motion-predictor",
        "nvtracker=disabled",
        "python scripts/preflight_pascal_safe.py",
    ):
        if token not in launcher:
            return fail(f"launcher missing safe-mode contract: {token}")

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
        f"gpu={gpu!r} runtime=CameraPascalSafeRuntime "
        "tracker=motion-predictor nvtracker=disabled stride_safe=1 "
        "video_path=RTSP-NVDEC-mux-tiler-OSD-EGL xid=idempotent stage_counters=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
