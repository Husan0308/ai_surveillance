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
    heatmap = source("services/camera_v2/person_tracking_heatmap.py")
    sparse = source("services/camera_v2/sparse_tracker_contract.py")
    sparse_native = source("services/camera_v2/native_sparse_tracker_contract.c")
    launcher = source("scripts/run_sentinel_vms.sh")
    preflight = source("scripts/preflight_rfdetr_core.py")
    sparse_preflight = source("scripts/preflight_sparse_tracker_contract.py")

    assert "from rfdetr import RFDETRSmall" in backend
    assert 'RFDETRSmall(device="cuda:0")' in backend
    assert "frame[..., ::-1]" in backend
    assert 'include_source_image=False' in backend
    assert "shape=infer_shape" in backend
    assert 'normalized == "person"' in backend
    assert "np.isin(class_id, (0, 1))" in backend
    assert 'detection._yolo_worker = rfdetr_worker' in backend

    assert "def _capture_gate_until_sample" in backend
    assert "requested = bool(self.capture_requested.get(cid, False))" in backend
    assert "CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample" in backend

    # Sparse external inference must still complete the DeepStream detector
    # contract for every muxed video frame so NvDCF can consume the full 20 FPS
    # stream between fresh RF-DETR observations.
    assert "install_sparse_tracker_contract()" in backend
    assert "camera_v2_mark_batch_infer_done" in sparse
    assert "marker.mark_batch_infer_done(buffer)" in sparse
    assert "original_inject(self, pad, info)" in sparse
    assert "CAMERA_TRACK_INPUT" in sparse
    assert "frame_meta->bInferDone = TRUE" in sparse_native
    assert "batch_meta->frame_meta_list" in sparse_native
    assert "SPARSE_TRACKER_PREFLIGHT=PASS" in sparse_preflight

    assert "_install_rfdetr_backend()" in heatmap
    assert '"CAMERA_V2_DETECT_WIDTH": "672"' in heatmap
    assert '"CAMERA_V2_DETECT_HEIGHT": "384"' in heatmap
    assert '"CAMERA_V2_MICRO_BATCH": "1"' in heatmap
    assert '"CAMERA_V2_DETECT_CONF": "0.18"' in heatmap

    assert "detector=RF-DETR-S@672x384" in launcher
    assert "python scripts/preflight_rfdetr_core.py" in launcher
    assert "python scripts/preflight_sparse_tracker_contract.py" in launcher
    assert "RFDETR_PREFLIGHT=PASS" in preflight


def test_ui_is_camera_only_and_uses_one_direct_native_qwidget_target() -> None:
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
    assert "mode=direct-native-qwidget" in native
    assert "self.surface = NativeVideoHost(self)" in native
    assert "self.surface.nativeReady.connect(self._start_or_bind)" in native
    assert "ProPipelineController()" in native
    assert "self.controller.poll()" in native

    for forbidden in (
        "QWindow",
        "createWindowContainer",
        "QFrame",
        "QLabel",
        "CameraStatusRow",
        "People in Building",
        "open_fullscreen_grid",
        "ProLiveVideoWall(",
    ):
        assert forbidden not in native

    native_calls = called_attributes(native)
    assert "focus" not in native_calls
    assert "set_fullscreen_mode" not in native_calls

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
