from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from shared.frame_bus import LatestFrameReader


class CameraTile(QFrame):
    def __init__(self, camera_id: str, frame_bus_dir: str, stale_after_ms: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.reader = LatestFrameReader(frame_bus_dir, camera_id)
        self.last_sequence: int | None = None
        self.last_received_ns = 0
        self.stale_after_ns = int(stale_after_ms * 1_000_000)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(camera_id)
        self.status = QLabel("WAITING")
        self.video = QLabel("Waiting for frame...")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(320, 180)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video.setStyleSheet("background: black; color: white;")
        header = QGridLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title, 0, 0)
        header.addWidget(self.status, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.video, 1)

    def refresh(self) -> None:
        packet = self.reader.read_latest(self.last_sequence)
        now = time.monotonic_ns()
        if packet is not None:
            minimum_bytes = packet.stride * packet.height
            if packet.width > 0 and packet.height > 0 and packet.stride >= packet.width * 4 and minimum_bytes <= len(packet.data):
                image = QImage(packet.data, packet.width, packet.height, packet.stride, QImage.Format.Format_RGBA8888)
                pixmap = QPixmap.fromImage(image)
                target = self.video.size()
                if target.width() > 0 and target.height() > 0:
                    pixmap = pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                self.video.setPixmap(pixmap)
                self.last_sequence = packet.sequence
                self.last_received_ns = now
                self.status.setText("LIVE")
                return
        if self.last_received_ns == 0 or now - self.last_received_ns > self.stale_after_ns:
            self.status.setText("WAITING")
            if self.video.pixmap() is None:
                self.video.setText("Waiting for frame...")

    def close_reader(self) -> None:
        self.reader.close()


class CameraWall(QWidget):
    def __init__(self, frame_bus_dir: str, stale_after_ms: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame_bus_dir = frame_bus_dir
        self.stale_after_ms = stale_after_ms
        self.tiles: dict[str, CameraTile] = {}
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)

    def set_cameras(self, camera_ids: list[str]) -> None:
        if list(self.tiles) == camera_ids:
            return
        for tile in self.tiles.values():
            tile.close_reader()
            self.grid.removeWidget(tile)
            tile.deleteLater()
        self.tiles.clear()
        for index, camera_id in enumerate(camera_ids):
            tile = CameraTile(camera_id, self.frame_bus_dir, self.stale_after_ms, self)
            self.tiles[camera_id] = tile
            row, column = divmod(index, 3)
            self.grid.addWidget(tile, row, column)

    def refresh_frames(self) -> None:
        for tile in self.tiles.values():
            tile.refresh()

    def close_readers(self) -> None:
        for tile in self.tiles.values():
            tile.close_reader()
