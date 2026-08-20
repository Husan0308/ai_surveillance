from __future__ import annotations

import multiprocessing as mp
import os
import sys

# The supplied UI was authored as a native PySide6 application. Keep that exact
# shell and patch only the parts required to put the real DeepStream wall behind
# its Monitoring overlays.
from . import sentinel_exact as ui


def _install_reference_contract() -> None:
    from PySide6.QtCore import QEvent, QTimer, Qt
    from PySide6.QtWidgets import QWidget

    # Exact names/capacities from the supplied data.py. Camera V2 has two physical
    # views per room in source order: 01/02, 03/04, 05/06.
    ui.ROOMS = (
        {"id": 1, "name": "Lobbi", "capacity": 40, "sources": (0, 1)},
        {"id": 2, "name": "Ofis", "capacity": 25, "sources": (2, 3)},
        {"id": 3, "name": "Ombor", "capacity": 15, "sources": (4, 5)},
    )

    # Match the uploaded ui.py stylesheet contract exactly where the historical
    # realtime fork had omitted two selectors.
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

        The previous version painted borders/status/badges on six translucent Qt
        child widgets while nveglglessink painted the same X11 area underneath.
        That gives two independent painters ownership of overlapping native pixels
        and is visible as intermittent background flashing over AnyDesk/X11.

        DeepStream OSD already owns camera/person graphics, so these child widgets
        now exist only for hit-testing/double-click focus. They never paint.
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
            # Native GstVideoOverlay owns every video pixel. Do not let Qt erase,
            # border, dim, antialias or otherwise touch the live surface.
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
            # GstVideoOverlay / nveglglessink owns this native X11 surface.
            return None

        def paintEvent(self, event):
            event.accept()

        def resizeEvent(self, event):
            # Keep BaseLiveWall's hit-test tile layout, but do not ask Qt to paint
            # the video. After X11 changes the native drawable size, explicitly
            # ask GstVideoOverlay to redraw the latest completed frame.
            BaseLiveWall.resizeEvent(self, event)
            if getattr(self, "controller", None) is not None:
                QTimer.singleShot(0, self.controller.expose)

        def event(self, event):
            result = super().event(event)
            if event.type() == QEvent.Type.WinIdChange:
                try:
                    xid = int(self.winId())
                    if xid > 0 and getattr(self, "controller", None) is not None:
                        self.controller.bind_window(xid)
                        QTimer.singleShot(0, self.controller.expose)
                except Exception:
                    pass
            return result

    # MainWindow -> MonitoringPage resolves these names at instantiation time, so
    # this keeps the historical exact shell intact and swaps only video integration.
    ui.CameraTile = ExactCameraTile
    ui.LiveWall = ExactLiveWall


def main() -> int:
    # Current deployment is Kubuntu/AnyDesk X11. GstVideoOverlay expects a native
    # XID in this integration; leave an explicitly chosen QPA untouched.
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
    # Match the uploaded main.py: normal window first, rather than forcing a
    # different maximized layout before the operator chooses it.
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
