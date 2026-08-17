from __future__ import annotations

import multiprocessing as mp
import os
import sys

from . import sentinel_exact as ui
from .safe_live_wall import SafeLiveWall


def main() -> int:
    # GstVideoOverlay receives a native X11 WId. Keep the realtime camera surface
    # native, while SafeLiveWall paints UI chrome in a separate translucent window.
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    mp.freeze_support()

    # MonitoringPage resolves LiveWall when MainWindow is instantiated, so patch
    # only the realtime surface implementation; the uploaded Sentinel shell remains
    # unchanged.
    ui.LiveWall = SafeLiveWall

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
