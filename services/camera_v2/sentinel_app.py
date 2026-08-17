from __future__ import annotations

import multiprocessing as mp
import os
import sys

# Keep the supplied Sentinel shell and patch only the realtime/native-video
# integration. The DeepStream wall remains the owner of the camera pixels.
from . import sentinel_exact as ui

SOURCE_WIDTH = 1280.0
SOURCE_HEIGHT = 720.0


def _install_reference_contract() -> None:
    from PySide6.QtCore import QEvent, Qt, QRectF
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
    from PySide6.QtWidgets import QWidget

    ui.ROOMS = (
        {"id": 1, "name": "Lobbi", "capacity": 40, "sources": (0, 1)},
        {"id": 2, "name": "Ofis", "capacity": 25, "sources": (2, 3)},
        {"id": 3, "name": "Ombor", "capacity": 15, "sources": (4, 5)},
    )

    # Preserve the stylesheet compatibility fixes used by the reference UI.
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
        """Transparent chrome for one cell of the native DeepStream wall.

        Important: never paint a translucent full-tile surface here. nveglglessink
        owns the native parent window, and a full QWidget backing-store composite can
        expose stale UI pixels over the video. Only thin lines, text/chips and the
        stable gutter are painted.
        """

        def __init__(self, source_id: int, parent=None):
            super().__init__(source_id, parent)
            self.controls.hide()
            self.controls.setEnabled(False)
            self._hovered = False
            self._tracks: list[dict] = []
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        def set_tracks(self, tracks: list[dict]) -> None:
            self._tracks = [dict(track) for track in tracks]
            self.update()

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
            # Do not call the base resizeEvent because it raises the removed controls.
            QWidget.resizeEvent(self, event)

        @staticmethod
        def _person_label(track: dict) -> str:
            if bool(track.get("known", False)):
                text = str(track.get("label") or "Known").strip()
                return text or "Known"
            try:
                object_id = int(track.get("object_id", 0))
            except (TypeError, ValueError):
                object_id = 0
            return f"Unknown_{object_id:02d}"

        def _draw_tracks(self, painter: QPainter) -> None:
            if not self.online or not self._tracks:
                return

            sx = self.width() / SOURCE_WIDTH
            sy = self.height() / SOURCE_HEIGHT
            font = QFont("DejaVu Sans Mono", 7, QFont.Weight.DemiBold)
            metrics = QFontMetrics(font)

            for track in self._tracks:
                try:
                    left = float(track.get("left", 0.0)) * sx
                    top = float(track.get("top", 0.0)) * sy
                    width = float(track.get("width", 0.0)) * sx
                    height = float(track.get("height", 0.0)) * sy
                except (TypeError, ValueError):
                    continue
                if width < 3.0 or height < 6.0:
                    continue

                # Keep the reference-style box inside the visible camera cell.
                x1 = max(5.0, min(self.width() - 6.0, left))
                y1 = max(26.0, min(self.height() - 7.0, top))
                x2 = max(x1 + 2.0, min(self.width() - 6.0, left + width))
                y2 = max(y1 + 2.0, min(self.height() - 7.0, top + height))
                if x2 <= x1 or y2 <= y1:
                    continue

                known = bool(track.get("known", False))
                color = QColor(ui.C["known"] if known else ui.C["unknown"])
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(color, 2))
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

                text = self._person_label(track)
                text_width = min(max(42, metrics.horizontalAdvance(text) + 12), max(42, int(self.width() * 0.45)))
                chip_h = 18
                chip_y = max(27.0, y1 - chip_h)
                chip_x = min(x1, max(5.0, self.width() - text_width - 6.0))
                chip = QRectF(chip_x, chip_y, text_width, chip_h)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawRoundedRect(chip, 2.5, 2.5)
                painter.setPen(QColor("#07110f"))
                painter.setFont(font)
                painter.drawText(
                    chip.adjusted(6, 0, -5, 0),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    text,
                )

        def paintEvent(self, event):
            # Preserve the reference camera name/FPS/offline/occupancy chrome.
            super().paintEvent(event)

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            # Slightly narrower than the old 5px mask: cameras gain a few pixels while
            # the 2x3 wall still reads as six separate cards.
            painter.setPen(QPen(QColor(ui.C["bg"]), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(2, 2, -3, -3))

            self._draw_tracks(painter)

            if self._hovered and not self.focused:
                # Line-only hover shadow/highlight. Never flood-fill the video.
                shadow = QColor(3, 15, 20, 175)
                painter.setPen(QPen(shadow, 6))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(self.rect().adjusted(6, 6, -6, -6), 7, 7)

                accent = QColor(ui.C["primary"])
                accent.setAlpha(210)
                painter.setPen(QPen(accent, 2))
                painter.drawRoundedRect(self.rect().adjusted(6, 6, -6, -6), 7, 7)

            painter.end()

    class ExactLiveWall(BaseLiveWall):
        """Native X11 video surface with Qt metadata chrome above it."""

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

        def refresh(self, snapshot: dict):
            super().refresh(snapshot)
            grouped: dict[int, list[dict]] = {source_id: [] for source_id in range(len(self.tiles))}
            for track in snapshot.get("tracks", []):
                try:
                    source_id = int(track.get("source_id", -1))
                except (TypeError, ValueError):
                    continue
                if source_id in grouped:
                    grouped[source_id].append(track)
            for tile in self.tiles:
                tile.set_tracks(grouped.get(tile.source_id, []))
                tile.raise_()

    class ExactMonitoringPage(BaseMonitoringPage):
        """Reference Monitoring layout with Total/Known/Unknown and larger live wall."""

        def __init__(self, controller):
            super().__init__(controller)

            # Keep the camera wall the visual priority while retaining a useful rail.
            self.layout.setContentsMargins(16, 8, 16, 10)
            self.layout.setSpacing(12)
            self.layout.setStretch(0, 4)
            self.layout.setStretch(1, 1)

            metrics = self.identity_rail.itemAt(0).layout()
            self.total_card = ui.StatCard("Total", "0", "blue", "Hozir binoda")
            self.total_card.setMinimumWidth(88)
            self.total_card.setMaximumWidth(112)
            self.known_card.setMinimumWidth(88)
            self.known_card.setMaximumWidth(112)
            self.unknown_card.setMinimumWidth(88)
            self.unknown_card.setMaximumWidth(112)
            if metrics is not None:
                metrics.setSpacing(8)
                metrics.insertWidget(0, self.total_card)

            self.recent_panel.setMinimumWidth(292)
            self.recent_panel.setMaximumWidth(338)

        def refresh(self, snapshot: dict):
            super().refresh(snapshot)
            self.total_card.value_label.setText(str(len(snapshot.get("tracks", []))))

        def set_camera_fullscreen(self, source_id: int | None):
            super().set_camera_fullscreen(source_id)
            self.total_card.setVisible(source_id is None)

    # MainWindow resolves these names at instantiation time.
    ui.CameraTile = ExactCameraTile
    ui.LiveWall = ExactLiveWall
    ui.MonitoringPage = ExactMonitoringPage


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
