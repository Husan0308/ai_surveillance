from __future__ import annotations

import os


def install() -> None:
    selected = os.environ.get("CAMERA_V2_DETECT_BACKEND", "old-ui-yolo26m").strip().lower()
    old_ui = selected in {"old-ui-yolo26m", "old-ui", "ui-era-yolo", "ui-yolo26m"}
    if old_ui:
        from .old_ui_detection_backend import install as backend_install
    elif selected in {"stable-yolo26m", "yolo26m", "yolo", "stable-yolo"}:
        from .stable_yolo_backend import install as backend_install
    elif selected in {"rfdetr-s", "rfdetr", "rf-detr-s", ""}:
        from .rfdetr_backend import install as backend_install
    else:
        raise RuntimeError(f"unsupported CAMERA_V2_DETECT_BACKEND={selected!r}")

    backend_install()

    if old_ui:
        # Restore the presentation that was used in rebuild/gpu-v2-clean:
        # confirmed local tracks are yellow and receive a stable
        # Unknown_C{camera}_{track} label.  This is presentation-only; detector
        # and Core-v1 Kalman/Byte association remain unchanged.
        from .legacy_unknown_overlay import install as install_unknown_overlay

        install_unknown_overlay()
