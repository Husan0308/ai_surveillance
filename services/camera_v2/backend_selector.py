from __future__ import annotations

import os


def install() -> None:
    selected = os.environ.get("CAMERA_V2_DETECT_BACKEND", "old-ui-yolo26m").strip().lower()
    if selected in {"old-ui-yolo26m", "old-ui", "ui-era-yolo", "ui-yolo26m"}:
        from .old_ui_detection_backend import install as backend_install
    elif selected in {"stable-yolo26m", "yolo26m", "yolo", "stable-yolo"}:
        from .stable_yolo_backend import install as backend_install
    elif selected in {"rfdetr-s", "rfdetr", "rf-detr-s", ""}:
        from .rfdetr_backend import install as backend_install
    else:
        raise RuntimeError(f"unsupported CAMERA_V2_DETECT_BACKEND={selected!r}")
    backend_install()
