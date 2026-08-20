from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def called_attributes(text: str) -> set[str]:
    tree = ast.parse(text)
    return {
        str(node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_rfdetr_small_is_the_active_person_detector_contract() -> None:
    backend = source("services/camera_v2/rfdetr_backend.py")
    pascal = source("services/camera_v2/pascal_safe_pipeline.py")
    controller = source("services/camera_v2/camera_wall_runtime.py")
    launcher = source("scripts/run_sentinel_vms.sh")
    preflight = source("scripts/preflight_rfdetr_core.py")
    pascal_preflight = source("scripts/preflight_pascal_safe.py")

    assert "from rfdetr import RFDETRSmall" in backend
    assert 'RFDETRSmall(device="cuda:0")' in backend
    assert "frame[..., ::-1]" in backend
    assert "include_source_image=False" in backend
    assert "shape=infer_shape" in backend
    assert 'normalized == "person"' in backend
    assert "np.isin(class_id, (0, 1))" in backend
    assert "detection._yolo_worker = rfdetr_worker" in backend
    assert "CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample" in backend

    # GTX 1050 Ti production path is a dedicated detection runtime. It must never
    # import or resolve DeepStream's NvDCF tracker stack.
    assert "class CameraPascalSafeRuntime(CameraDetectionV2)" in pascal
    assert "def _install_osd_and_meta(self)" in pascal
    assert "self.wall_queue.unlink(self.sink)" in pascal
    assert "if queue_src.is_linked()" in pascal
    assert "mapped_size" in pascal
    assert "row_stride" in pascal
    assert "tight_stride" in pascal
    assert "CAMERA_INFER_LAYOUT" in pascal
    assert "safe_mux_batches" in pascal
    assert "safe_wall_frames" in pascal
    assert "tracker=motion-predictor" in pascal
    assert "nvtracker=disabled" in pascal
    for forbidden in (
        "CameraPersonTrackingV2",
        "CameraPersonTrackingFinal",
        "libnvds_nvmultiobjecttracker",
        "config_tracker_NvDCF",
    ):
        assert forbidden not in pascal

    assert "from .pascal_safe_pipeline import CameraPascalSafeRuntime" in controller
    assert "runtime = CameraPascalSafeRuntime()" in controller
    assert "bound_xid = 0" in controller
    assert "if target == bound_xid" in controller
    assert "GstVideo.VideoOverlay.set_window_handle" in controller
    assert "runtime.bus.set_sync_handler" in controller
    assert "GRID_COLUMNS = 2" in controller
    assert "GRID_ROWS = 3" in controller

    assert "export CAMERA_V2_PASCAL_SAFE=1" in launcher
    assert "tracker=motion-predictor" in launcher
    assert "nvtracker=disabled" in launcher
    assert "detector=RF-DETR-S@672x384" in launcher
    assert "python scripts/preflight_pascal_safe.py" in launcher
    assert "python scripts/preflight_sparse_tracker_contract.py" not in launcher
    assert "RFDETR_PREFLIGHT=PASS" in preflight
    assert "PASCAL_SAFE_PREFLIGHT=PASS" in pascal_preflight


def test_ui_is_camera_only_and_uses_clean_controller() -> None:
    native = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    shell = source("services/camera_v2/sentinel_ui.py")

    assert "class NativeVideoHost(QWidget)" in native
    assert "WA_NativeWindow, True" in native
    assert "WA_DontCreateNativeAncestors, True" in native
    assert "WA_NoSystemBackground, True" in native
    assert "WA_OpaquePaintEvent, True" in native
    assert "WA_PaintOnScreen, True" in native
    assert "def paintEngine(self)" in native
    assert "xid = int(self.winId())" in native
    assert "from .camera_wall_runtime import CameraWallController" in native
    assert "self.controller = CameraWallController()" in native
    assert "self.surface.nativeReady.connect(self._start_or_bind)" in native

    for forbidden in (
        "QWindow",
        "createWindowContainer",
        "QFrame",
        "QLabel",
        "ProPipelineController",
        "ProLiveVideoWall",
        "People in Building",
        "open_fullscreen_grid",
    ):
        assert forbidden not in native

    assert 'BUILD_TAG = "2026.08.20-r16-pascal-safe"' in shell
    assert "self.monitoring_page = MonitoringPage()" in shell
    assert "self.setCentralWidget(self.monitoring_page)" in shell
    for forbidden in (
        "QStackedWidget",
        "PeoplePage",
        "EventsPage",
        "RoomsPage",
        "EnrollmentPage",
        "SettingsPage",
        "sidebar",
    ):
        assert forbidden not in shell

    shell_calls = called_attributes(shell)
    assert "showFullScreen" not in shell_calls
    assert "showNormal" not in shell_calls
    assert "showMaximized" in shell_calls
