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
    from PySide6.QtGui import QColor, QPainter, QPen
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
    BaseMonitoringPage = ui.MonitoringPage

    class ExactCameraTile(BaseCameraTile):
        """Realtime overlay with lightweight Sentinel hover feedback.

        The camera pixels stay owned by the native DeepStream/EGL surface. Qt only
        paints a tiny transparent chrome layer, so hover feedback never copies or
        rescales a video frame and cannot reduce the camera pipeline FPS.
        """

        def __init__(self, source_id: int, parent=None):
            super().__init__(source_id, parent)
            self.controls.hide()
            self.controls.setEnabled(False)
            self._hovered = False
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        def enterEvent(self, event):
            self._hovered = True
            self.update()
            QWidget.enterEvent(self, event)

        def leaveEvent(self, event):
            self._hovered = False
            self.controls.hide()
            self.update()
            QWidget.leaveEvent(self, event)

        def resizeEvent(self, event):
            # Do not call BaseCameraTile.resizeEvent(), because that method raises
            # the removed controls. There is no geometry to maintain for them.
            QWidget.resizeEvent(self, event)

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            # Keep a compact visual gutter between native tiler cells. 4 px on each
            # tile side gives the camera wall a little more usable image area than
            # the previous 5 px mask while preserving clear card separation.
            painter.setPen(QPen(QColor(ui.C["bg"]), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(2, 2, -3, -3))

            if self._hovered and not self.focused:
                # Inner shadow/glow: visible enough to show which camera is active,
                # but intentionally subtle so it does not cover detections or faces.
                glow_rect = self.rect().adjusted(7, 7, -7, -7)
                tint = QColor(ui.C["primary"])
                tint.setAlpha(10)
                painter.fillRect(glow_rect, tint)
                for width, alpha, inset in ((9, 24, 8), (5, 55, 7), (2, 210, 6)):
                    color = QColor(ui.C["primary"])
                    color.setAlpha(alpha)
                    painter.setPen(QPen(color, width))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(self.rect().adjusted(inset, inset, -inset, -inset), 7, 7)

            painter.end()

    class ExactLiveWall(BaseLiveWall):
        """Native X11 surface owned by nveglglessink, not Qt backing-store paint."""

        def __init__(self, controller, parent=None):
            super().__init__(controller, parent)
            self.setStyleSheet("")
            self.setAutoFillBackground(False)
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
            self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
            _ = int(self.winId())

        def paintEngine(self):
            # GstVideoOverlay / nveglglessink owns this native X11 surface.
            return None

        def paintEvent(self, event):
            event.accept()

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

    class ExactMonitoringPage(BaseMonitoringPage):
        """Give the six-camera wall slightly more room without changing structure."""

        def __init__(self, controller):
            super().__init__(controller)
            # Recover pixels from empty page chrome first; do not resize/copy video.
            self.layout.setContentsMargins(14, 8, 14, 10)
            self.layout.setSpacing(12)
            self.layout.setStretch(0, 4)
            self.layout.setStretch(1, 1)

            # Keep the identity rail useful but compact so the camera wall remains
            # the visual priority on the 1280-ish AnyDesk/operator window.
            self.known_card.setMinimumWidth(112)
            self.unknown_card.setMinimumWidth(112)
            self.known_card.setMaximumWidth(145)
            self.unknown_card.setMaximumWidth(145)
            self.recent_panel.setMinimumWidth(258)
            self.recent_panel.setMaximumWidth(300)

    # MainWindow -> MonitoringPage resolves these names at instantiation time, so
    # this keeps the historical exact shell intact and swaps only video integration.
    ui.CameraTile = ExactCameraTile
    ui.LiveWall = ExactLiveWall
    ui.MonitoringPage = ExactMonitoringPage


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
