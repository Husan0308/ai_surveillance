from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QWidget

from .smooth_frame_reader import SmoothFrameReader


CAMERA_IDS = [f"CAM-{index:02d}" for index in range(1, 7)]


class CameraTile(QWidget):
    def __init__(self, camera_id: str, reader: SmoothFrameReader):
        super().__init__()
        self.camera_id = camera_id
        self.reader = reader
        self.setMinimumSize(320, 180)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))

        image, _version = self.reader.latest()
        if image is not None and not image.isNull():
            source_w = max(1, image.width())
            source_h = max(1, image.height())
            target_w = self.width()
            target_h = self.height()
            scale = min(target_w / source_w, target_h / source_h)
            draw_w = max(1, round(source_w * scale))
            draw_h = max(1, round(source_h * scale))
            x = (target_w - draw_w) // 2
            y = (target_h - draw_h) // 2
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(QRect(x, y, draw_w, draw_h), image)

        # Only a tiny camera label; detection boxes/text are already baked into
        # the MJPEG frame by the ML publisher. No dashboard chrome is drawn.
        painter.fillRect(8, 8, 74, 24, QColor(0, 0, 0, 150))
        painter.setPen(QColor("#ffffff"))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(15, 25, self.camera_id)
        painter.end()


class SimpleDetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Surveillance — Cameras + Detection")
        self.setStyleSheet("background:#000;")
        self.readers = {camera_id: SmoothFrameReader(camera_id) for camera_id in CAMERA_IDS}
        for reader in self.readers.values():
            reader.start()

        body = QWidget()
        body.setStyleSheet("background:#000;")
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        self.tiles = []
        for index, camera_id in enumerate(CAMERA_IDS):
            tile = CameraTile(camera_id, self.readers[camera_id])
            grid.addWidget(tile, index // 3, index % 3)
            self.tiles.append(tile)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)

        self.setCentralWidget(body)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._render)
        self.timer.start(33)

    def _render(self):
        for tile in self.tiles:
            tile.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showMaximized()
            return
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        for reader in self.readers.values():
            reader.stop()
        event.accept()


def run():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AI Surveillance Simple Detection")
    window = SimpleDetectionWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
