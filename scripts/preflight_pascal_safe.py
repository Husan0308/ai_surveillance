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


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    if os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        print("PASCAL_SAFE_PREFLIGHT=SKIP mode=disabled")
        return 0
    if os.environ.get("CAMERA_V2_RTSP_TRANSPORT", "").strip().lower() != "tcp":
        return fail("production runtime must pin CAMERA_V2_RTSP_TRANSPORT=tcp")
    if int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "0")) < 200:
        return fail("production RTSP latency must be at least 200 ms")

    runtime = source("services/camera_v2/pascal_safe_pipeline.py")
    controller = source("services/camera_v2/camera_wall_runtime.py")
    ui = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    launcher = source("scripts/run_sentinel_vms.sh")
    secure = source("services/camera_v2/secure.py")
    detector_backend = source("services/camera_v2/rfdetr_backend.py")
    oldgood = source("services/camera_v2/old_good_rfdetr_tracker.py")

    for token in (
        "class CameraPascalSafeRuntime(CameraDetectionV2)",
        "SecureCameraWallV2._add_camera(self, index, camera)",
        "def _install_analysis_inference(self)",
        "pascal_analysis_tiler",
        "def _analysis_gate_probe",
        "def _on_analysis_sample",
        "CAMERA_DETECT_PATH mode=analysis-tiler",
        "source_path=direct-to-nvstreammux",
        "demux=disabled",
        "mux_batch_retention=bounded",
        "analysis_frames",
        "safe_mux_batches",
        "safe_wall_frames",
        "safe_sink_buffers",
        "CAMERA_STARTUP_STALL",
        "display_failover_requested = True",
        "CAMERA_DISPLAY_FAILOVER",
        "self.tracker = None",
        "nvtracker=disabled",
    ):
        if token not in runtime:
            return fail(f"missing runtime contract: {token}")

    for forbidden in (
        'self._make("nvstreamdemux"',
        "pascal_infer_demux",
        "self.infer_demux",
        "CameraPersonTrackingV2",
        "CameraPersonTrackingFinal",
        "libnvds_nvmultiobjecttracker",
        "config_tracker_NvDCF",
    ):
        if forbidden in runtime:
            return fail(f"Pascal runtime leaked forbidden code path: {forbidden}")

    for token in (
        "OldGoodRFDETRBoxManager",
        "detection.SmoothBoxManager = OldGoodRFDETRBoxManager",
        "detection.CameraDetectionV2._inject_boxes_probe = _no_pretiler_detection_meta",
        "_posttiler_overlay_probe",
        "CAMERA_OLDGOOD_OVERLAY_READY",
        "RFDETR_OLDGOOD_READY",
        "RFDETR_OLDGOOD_RESULT",
        "CAM-05",
        "CAM-06",
        "flow=OFF reid=OFF nvtracker=OFF",
    ):
        if token not in detector_backend:
            return fail(f"missing old-good RF-DETR contract: {token}")

    for token in (
        "class OldGoodRFDETRBoxManager",
        "VisualTracker",
        "hold_ms",
        "memory_ms",
        "prediction_ms",
        "max_result_age",
        "CAM-06",
    ):
        if token not in oldgood:
            return fail(f"missing Core-v1 tracker contract: {token}")

    for forbidden in (
        "FlowAssistedPersonTracker",
        "attach_motion_flow(self)",
    ):
        selected_part = detector_backend.split("def install() -> None:", 1)[-1]
        if forbidden in selected_part:
            return fail(f"RF-DETR old-good path leaked flow logic: {forbidden}")

    for token in (
        'CAMERA_V2_RTSP_TRANSPORT',
        'self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)',
        'self._set_if(element, "protocols", 4)',
        "def _source_pad_added",
        "caps.is_any()",
        "source linked transport=",
    ):
        if token not in secure:
            return fail(f"missing RTSP TCP/dynamic-pad contract: {token}")

    for token in (
        "def set_focus(source_id: int)",
        'runtime.tiler.set_property("show-source", sid)',
        "def focus(self, source_id: int)",
        'self.command_q.put_nowait(("focus", sid))',
        "if target == bound_xid",
        "GstVideo.VideoOverlay.set_window_handle",
    ):
        if token not in controller:
            return fail(f"missing camera-wall controller contract: {token}")

    for token in (
        "cameraClicked = Signal(int)",
        "def _grid_source_at",
        "window.showFullScreen()",
        "window.showMaximized()",
        "self.controller.focus(sid)",
        "self.controller.focus(-1)",
    ):
        if token not in ui:
            return fail(f"missing click-fullscreen UI contract: {token}")

    for token in (
        "export CAMERA_V2_RTSP_TRANSPORT=tcp",
        "export CAMERA_V2_RTSP_LATENCY_MS=250",
        "export CAMERA_V2_DETECT_BACKEND=rfdetr-s",
        "export CAMERA_V2_RFDETR_MODEL_WIDTH=672",
        "export CAMERA_V2_RFDETR_MODEL_HEIGHT=384",
        "detector_path=analysis-tiler",
        "logic=old-good-core-v1",
        "tracker=kalman-byte",
        "flow=OFF",
        "reid=OFF",
        "overlay=post-tiler-wall-space",
        "export CAMERA_V2_DISPLAY_BACKEND=egl",
        "ui=camera-only-2x3-click-fullscreen",
    ):
        if token not in launcher:
            return fail(f"launcher missing safe-mode contract: {token}")

    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        for plugin in (
            "nvurisrcbin", "nvstreammux", "nvmultistreamtiler", "nvvideoconvert",
            "appsink", "nveglglessink", "ximagesink",
        ):
            if Gst.ElementFactory.find(plugin) is None:
                return fail(f"required plugin unavailable: {plugin}")
    except Exception as exc:
        return fail(f"cannot validate GStreamer plugins: {type(exc).__name__}: {exc}")

    gpu = "unknown"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        ).strip().splitlines()[0]
    except Exception:
        pass

    print(
        "PASCAL_SAFE_PREFLIGHT=PASS "
        f"gpu={gpu!r} runtime=CameraPascalSafeRuntime rtsp=tcp latency>=200ms "
        "source_path=direct-to-mux detector_path=analysis-tiler demux=disabled "
        "mux_retention=bounded detector=RF-DETR-S logic=old-good-core-v1 "
        "tracker=kalman-byte flow=OFF reid=OFF nvtracker=disabled "
        "overlay=post-tiler-wall-space display=egl-primary+x11-zero-render-fallback "
        "fullscreen=click+escape"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
