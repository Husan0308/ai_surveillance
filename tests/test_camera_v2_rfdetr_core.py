from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_rfdetr_small_is_the_active_person_detector_contract() -> None:
    backend = source("services/camera_v2/rfdetr_backend.py")
    heatmap = source("services/camera_v2/person_tracking_heatmap.py")
    launcher = source("scripts/run_sentinel_vms.sh")
    preflight = source("scripts/preflight_rfdetr_core.py")

    assert "from rfdetr import RFDETRSmall" in backend
    assert 'RFDETRSmall(device="cuda:0")' in backend
    assert "frame[..., ::-1]" in backend
    assert 'include_source_image=False' in backend
    assert "shape=infer_shape" in backend
    assert 'normalized == "person"' in backend
    assert "np.isin(class_id, (0, 1))" in backend
    assert 'detection._yolo_worker = rfdetr_worker' in backend

    assert "_install_rfdetr_backend()" in heatmap
    assert '"CAMERA_V2_DETECT_WIDTH": "736"' in heatmap
    assert '"CAMERA_V2_DETECT_HEIGHT": "416"' in heatmap
    assert '"CAMERA_V2_MICRO_BATCH": "1"' in heatmap
    assert '"CAMERA_V2_DETECT_CONF": "0.18"' in heatmap

    assert "detector=RF-DETR-S@736x416" in launcher
    assert "python scripts/preflight_rfdetr_core.py" in launcher
    assert "RFDETR_PREFLIGHT=PASS" in preflight


def test_native_video_shell_keeps_egl_inside_monitoring_page() -> None:
    native = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    shell = source("services/camera_v2/sentinel_ui.py")

    assert "WA_DontCreateNativeAncestors, False" in native
    assert "WA_NativeWindow, True" in native
    assert "widget.winId()" in native
    assert "from .sentinel_ui_monitoring_native import MonitoringPage" in shell
    assert "showFullScreen()" not in shell
    assert "showNormal()" not in shell
    assert "window.showMaximized()" in shell
