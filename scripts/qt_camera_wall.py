#!/usr/bin/env python3
from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QMainWindow, QSizePolicy, QWidget

BASE_URL = "http://127.0.0.1:8001"
CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))


class CameraTile(QFrame):
    double_clicked = Signal(object)

    def __init__(self, camera_id: str, parent=None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self._buffer = bytearray()
        self._reply = None
        self._pixmap: QPixmap | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QFrame{background:#03060b;border:1px solid #202936;border-radius:8px;}")

        self.image = QLabel("CONNECTING…", self)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet("QLabel{background:#03060b;color:#7c8798;border:none;}")

        self.title = QLabel(camera_id, self)
        self.title.setStyleSheet(
            "QLabel{background:rgba(0,0,0,175);color:white;padding:5px 9px;"
            "border-radius:5px;font-weight:600;}"
        )
        self.title.adjustSize()

        self.network = QNetworkAccessManager(self)
        QTimer.singleShot(50, self._connect)

    def _connect(self) -> None:
        self._buffer.clear()
        request = QNetworkRequest(QUrl(f"{BASE_URL}/video/{self.camera_id}"))
        request.setRawHeader(b"Cache-Control", b"no-cache")
        self._reply = self.network.get(request)
        self._reply.readyRead.connect(self._read)
        self._reply.finished.connect(self._finished)

    def _read(self) -> None:
        if self._reply is None:
            return
        self._buffer.extend(bytes(self._reply.readAll()))
        while True:
            start = self._buffer.find(b"\xff\xd8")
            if start < 0:
                if len(self._buffer) > 2_000_000:
                    self._buffer.clear()
                return
            end = self._buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start:
                    del self._buffer[:start]
                return
            jpeg = bytes(self._buffer[start : end + 2])
            del self._buffer[: end + 2]
            image = QImage.fromData(jpeg, "JPG")
            if not image.isNull():
                self._pixmap = QPixmap.fromImage(image)
                self._render()

    def _render(self) -> None:
        if self._pixmap is None or self.image.width() < 2 or self.image.height() < 2:
            return
        self.image.setPixmap(
            self._pixmap.scaled(
                self.image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _finished(self) -> None:
        self._reply = None
        self.image.setText("RECONNECTING…")
        QTimer.singleShot(800, self._connect)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.image.setGeometry(self.rect())
        self.title.move(12, 12)
        self.title.raise_()
        self._render()

    def mouseDoubleClickEvent(self, event) -> None:
        self.double_clicked.emit(self)
        event.accept()


class CameraWall(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Surveillance — Camera Wall")
        self.resize(1500, 950)
        self._fullscreen_tile: CameraTile | None = None

        self.root = QWidget(self)
        self.root.setStyleSheet("background:#020409;")
        self.grid = QGridLayout(self.root)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setSpacing(8)
        self.setCentralWidget(self.root)

        self.tiles: list[CameraTile] = []
        for index, camera_id in enumerate(CAMERAS):
            tile = CameraTile(camera_id, self.root)
            tile.double_clicked.connect(self.toggle_tile)
            self.tiles.append(tile)
            self.grid.addWidget(tile, index // 2, index % 2)

        for row in range(3):
            self.grid.setRowStretch(row, 1)
        for column in range(2):
            self.grid.setColumnStretch(column, 1)

    def toggle_tile(self, tile: CameraTile) -> None:
        if self._fullscreen_tile is None:
            self._fullscreen_tile = tile
            for other in self.tiles:
                if other is not tile:
                    other.hide()
            self.grid.addWidget(tile, 0, 0, 3, 2)
        else:
            self._fullscreen_tile = None
            for index, other in enumerate(self.tiles):
                self.grid.addWidget(other, index // 2, index % 2)
                other.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWall()
    window.show()
    raise SystemExit(app.exec())
