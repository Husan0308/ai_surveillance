from __future__ import annotations

import multiprocessing as mp
import os
import sys

# The supplied UI was authored as a native PySide6 application. Keep that exact
# shell and patch only the parts required to put the real DeepStream wall behind
# its Monitoring overlays.
from . import sentinel_exact as ui


def _install_reference_contract() -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtWidgets import QWidget

    # Exact names/capacities from the supplied data.py. Camera V2 has two physical
    # views per room in source order: 01/02, 03/04, 05/06.
    ui.ROOMS = (
        {"id": 1, "name": "Lobbi", "capacity": 40, "sources": (0, 1)},
        {"id": 2, "name": "Ofis", "capacity": 25, "sources": (2, 3)},
        {"id": 3, "name": "Ombor", "capacity": 15, "sources": (4, 5)},
    )

    if "QPlainTextEdit" not in ui.APP_QSS:
        ui.APP_QSS = ui.APP_QSS.replace(
            "QLineEdit, QComboBox, QTextEdit",
            "QLineEdit, QComboBox, QPlainTextEdit, QTextEdit",
        ).replace(
            "QLineEdit:focus, QComboBox:focus, QTextEdit:focus",
            "QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus",
        )
    if "QTableWidget" not in ui.APP_QSS:
        ui.APP_QSS = ui.APP_QSS.replace(
            "QDialog { background:",
            "QTableWidget { background: #0e151d; alternate-background-color: #101923; border: 1px solid #22303e; gridline-color: #22303e; selection-background-color: #17313a; }\n"
            "QHeaderView::section { background: #111b25; color: #7e8c99; border: 0; border-bottom: 1px solid #22303e; padding: 9px; font-family: 'DejaVu Sans Mono'; font-size: 10px; }\n"
            "QDialog { background:",
        )

    BaseCameraTile = ui.CameraTile
    BaseLiveWall = ui.LiveWall

    class ExactCameraTile(BaseCameraTile):
        """Input-only transparent tile above the native DeepStream wall.

        DeepStream OSD owns all video/person graphics. The Qt children exist only
        for hit-testing and double-click focus, so they never repaint live pixels.
        """

        def __init__(self, source_id: int, parent=None):
            super().__init__(source_id, parent)
            self.controls.hide()
            self.controls.setEnabled(False)
            self.setAutoFillBackground(False)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        def enterEvent(self, event):
            QWidget.enterEvent(self, event)

        def leaveEvent(self, event):
            self.controls.hide()
            QWidget.leaveEvent(self, event)

        def resizeEvent(self, event):
            QWidget.resizeEvent(self, event)

        def paintEvent(self, event):
            event.accept()

    class ExactLiveWall(BaseLiveWall):
        """Native X11 surface owned exclusively by nveglglessink."""

        def __init__(self, controller, parent=None):
            super().__init__(controller, parent)
            self.setStyleSheet("")
            self.setAutoFillBackground(False)
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
            self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
            _ = int(self.winId())

        def paintEngine(self):
            return None

        def paintEvent(self, event):
            event.accept()

        def resizeEvent(self, event):
            # Relayout transparent mouse-hit tiles only. The continuously PLAYING
            # EGL sink redraws video itself; forcing GstVideoOverlay.expose() on
            # every resize can race with native surface reconfiguration.
            BaseLiveWall.resizeEvent(self, event)

        def event(self, event):
            result = super().event(event)
            if event.type() == QEvent.Type.WinIdChange:
                try:
                    xid = int(self.winId())
                    if xid > 0 and getattr(self, "controller", None) is not None:
                        self.controller.bind_window(xid)
                except Exception:
                    pass
            return result

    ui.CameraTile = ExactCameraTile
    ui.LiveWall = ExactLiveWall


def main() -> int:
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    mp.freeze_support()
    _install_reference_contract()

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
