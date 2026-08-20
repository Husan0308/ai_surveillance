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
    enabled = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
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
        "tracker=motion-predictor",
        "nvtracker=disabled",
    ):
        if token not in runtime:
            return fail(f"missing runtime contract: {token}")

    for forbidden in (
        "nvstreamdemux",
        "pascal_infer_demux",
        "CameraPersonTrackingV2",
        "CameraPersonTrackingFinal",
        "libnvds_nvmultiobjecttracker",
        "config_tracker_NvDCF",
    ):
        if forbidden in runtime:
            return fail(f"Pascal runtime leaked forbidden dependency: {forbidden}")

    for token in (
        'CAMERA_V2_RTSP_TRANSPORT',
        'self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)',
        'self._set_if(element, "protocols", 4)',
        'transport={transport}',
        "def _source_pad_added",
        "caps.is_any()",
        "source linked transport=",
    ):
        if token not in secure:
            return fail(f"missing RTSP TCP/dynamic-pad contract: {token}")

    for token in (
        "def set_focus(source_id: int)",
        'runtime.tiler.set_property("show-source", sid)',
        "FOCUS_WIDTH = 1920",
        "FOCUS_HEIGHT = 1080",
        "def focus(self, source_id: int)",
        'self.command_q.put_nowait(("focus", sid))',
        "bound_xid = 0",
        "if target == bound_xid",
        "GstVideo.VideoOverlay.set_window_handle",
    ):
        if token not in controller:
            return fail(f"missing camera-wall controller contract: {token}")

    for token in (
        "cameraClicked = Signal(int)",
        "def _grid_source_at",
        "self.surface.cameraClicked.connect(self._camera_clicked)",
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
        "rtsp=tcp latency=250ms",
        "export CAMERA_V2_PASCAL_SAFE=1",
        "detector_path=analysis-tiler",
        "demux=disabled",
        "tracker=motion-predictor",
        "nvtracker=disabled",
        "export CAMERA_V2_DISPLAY_BACKEND=egl",
        "export CAMERA_V2_EGL_FAILOVER_SEC=8.0",
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
            "nvurisrcbin",
            "nvstreammux",
            "nvmultistreamtiler",
            "nvvideoconvert",
            "appsink",
            "nveglglessink",
            "ximagesink",
        ):
            if Gst.ElementFactory.find(plugin) is None:
                return fail(f"required plugin unavailable: {plugin}")
    except Exception as exc:
        return fail(f"cannot validate GStreamer plugins: {type(exc).__name__}: {exc}")

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
        "rtsp=tcp latency>=200ms source_path=direct-to-mux "
        "detector_path=analysis-tiler demux=disabled mux_retention=bounded "
        "tracker=motion-predictor nvtracker=disabled "
        "display=egl-primary+x11-zero-render-fallback fullscreen=click+escape"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
