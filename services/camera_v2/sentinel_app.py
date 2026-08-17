from __future__ import annotations

import multiprocessing as mp
import os
import sys

from . import sentinel_exact as ui


def main() -> int:
    # Native DeepStream video is bound into the Qt WId through GstVideoOverlay.
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    mp.freeze_support()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(ui.APP_QSS)

    window = ui.MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
