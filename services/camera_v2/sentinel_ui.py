from __future__ import annotations

import multiprocessing as mp
import os

from PySide6.QtWidgets import QApplication, QMainWindow

from .sentinel_ui_monitoring_native import MonitoringPage


BUILD_TAG = "2026.08.20-r17-postmux-fullscreen"


class MainWindow(QMainWindow):
    """Production camera-only shell for the six-camera native wall."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sentinel VMS - Cameras")
        self.resize(1600, 900)
        self.setMinimumSize(960, 540)

        self.monitoring_page = MonitoringPage()
        self.setCentralWidget(self.monitoring_page)

    def closeEvent(self, event) -> None:
        try:
            self.monitoring_page.shutdown()
        finally:
            super().closeEvent(event)


def run() -> int:
    if os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QMainWindow{background:#000000;}"
        "QWidget#cameraOnlyPage{background:#000000;}"
        "QWidget#nativeVideoHost{background:#000000;}"
    )
    print(
        f"SENTINEL_QT build={BUILD_TAG} platform={app.platformName()} "
        f"display={os.environ.get('DISPLAY', 'unset')} mode=camera-only",
        flush=True,
    )

    window = MainWindow()
    window.showMaximized()
    return app.exec()


def main() -> int:
    return run()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
