from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QWidget

from .mmap_frame_reader import SmoothMmapFrameReader


CAMERA_IDS = [f"CAM-{index:02d}" for index in range(1, 7)]


class CameraTile(QWidget):
    def __init__(self, camera_id: str, reader: SmoothMmapFrameReader):
        super().__init__()
        self.camera_id = camera_id
        self.reader = reader
        self._version = -1
        self._image: QImage | None = None
        self._last_frame_mono = 0.0
        self.setMinimumSize(320, 180)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def refresh_if_new(self) -> None:
        image, version = self.reader.latest()
        if image is None or image.isNull() or version <= self._version:
            return
        self._image = image
        self._version = int(version)
        self._last_frame_mono = time.monotonic()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))

        image = self._image
        if image is not None and not image.isNull():
            sw = max(1, image.width())
            sh = max(1, image.height())
            scale = min(self.width() / sw, self.height() / sh)
            dw = max(1, round(sw * scale))
            dh = max(1, round(sh * scale))
            x = (self.width() - dw) / 2.0
            y = (self.height() - dh) / 2.0
            painter.drawImage(QRectF(x, y, dw, dh), image)
        else:
            painter.setPen(QColor("#777777"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "WAITING FOR CAMERA",
            )

        painter.fillRect(8, 8, 86, 24, QColor(0, 0, 0, 165))
        painter.setPen(QColor("#ffffff"))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(15, 25, self.camera_id)

        if self._last_frame_mono and time.monotonic() - self._last_frame_mono > 1.5:
            painter.setPen(QColor("#ff5a5a"))
            painter.drawText(self.width() - 75, 25, "OFFLINE")

        painter.end()


class CameraWallWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Surveillance — Camera Baseline")
        self.setStyleSheet("background:#000;")

        body = QWidget()
        body.setStyleSheet("background:#000;")
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        self.readers: dict[str, SmoothMmapFrameReader] = {}
        self.tiles: list[CameraTile] = []
        for index, camera_id in enumerate(CAMERA_IDS):
            reader = SmoothMmapFrameReader(camera_id)
            reader.start()
            tile = CameraTile(camera_id, reader)
            self.readers[camera_id] = reader
            self.tiles.append(tile)
            grid.addWidget(tile, index // 3, index % 3)

        for column in range(3):
            grid.setColumnStretch(column, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)

        self.setCentralWidget(body)

        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.render_timer.timeout.connect(self._render)
        self.render_timer.start(33)

        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self._print_metrics)
        self.metrics_timer.start(5000)

    def _render(self) -> None:
        for tile in self.tiles:
            tile.refresh_if_new()

    def _print_metrics(self) -> None:
        parts = []
        for camera_id in CAMERA_IDS:
            reader = self.readers[camera_id]
            parts.append(
                f"{camera_id}:frames={reader.frames} "
                f"age={reader.last_frame_age_ms:.0f}ms "
                f"reconnects={reader.reconnects}"
            )
        print("CAMERA_WALL " + " | ".join(parts), flush=True)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showMaximized()
            return
        if event.key() == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.render_timer.stop()
        self.metrics_timer.stop()
        for reader in self.readers.values():
            reader.stop()
        event.accept()


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AI Surveillance Camera Baseline")
    window = CameraWallWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
