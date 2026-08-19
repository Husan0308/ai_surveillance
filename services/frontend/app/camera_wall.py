from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from services.frontend.app.mjpeg_reader import SmoothMjpegReader


class CameraTile(QFrame):
    def __init__(self, camera_id: str, ml_video_base_url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.reader = SmoothMjpegReader(camera_id, ml_video_base_url)
        self.last_version = 0

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(camera_id)
        self.status = QLabel("CONNECTING")
        self.video = QLabel("Connecting...")
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

        self.reader.start()

    def refresh(self) -> None:
        image, version = self.reader.latest()
        if image is not None and version > self.last_version:
            self.last_version = version
            pixmap = QPixmap.fromImage(image)
            target = self.video.size()
            if target.width() > 0 and target.height() > 0:
                pixmap = pixmap.scaled(
                    target,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            self.video.setPixmap(pixmap)
            self.status.setText(f"LIVE {self.reader.frames}")
            return

        if self.reader.last_error:
            self.status.setText("RECONNECTING")
            if self.video.pixmap() is None:
                self.video.setText(self.reader.last_error)
        elif self.last_version == 0:
            self.status.setText("CONNECTING")

    def close_reader(self) -> None:
        self.reader.stop()
        self.reader.join()


class CameraWall(QWidget):
    """Canonical six-camera wall: two cameras per row, direct ML MJPEG video."""

    COLUMNS = 2

    def __init__(self, ml_video_base_url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ml_video_base_url = ml_video_base_url
        self.tiles: dict[str, CameraTile] = {}
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        for row in range(3):
            self.grid.setRowStretch(row, 1)
        for column in range(self.COLUMNS):
            self.grid.setColumnStretch(column, 1)

    def set_cameras(self, camera_ids: list[str]) -> None:
        if list(self.tiles) == camera_ids:
            return
        for tile in self.tiles.values():
            tile.close_reader()
            self.grid.removeWidget(tile)
            tile.deleteLater()
        self.tiles.clear()
        for index, camera_id in enumerate(camera_ids):
            tile = CameraTile(camera_id, self.ml_video_base_url, self)
            self.tiles[camera_id] = tile
            row, column = divmod(index, self.COLUMNS)
            self.grid.addWidget(tile, row, column)

    def refresh_frames(self) -> None:
        for tile in self.tiles.values():
            tile.refresh()

    def close_readers(self) -> None:
        for tile in self.tiles.values():
            tile.close_reader()
