from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from services.frontend.app.mjpeg_reader import SmoothMjpegReader
from services.frontend.app.mmap_frame_reader import SmoothMmapFrameReader


class MmapVideoCanvas(QWidget):
    """Paint only newly committed mmap frames; never queue presentation work."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._version = -1
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    @property
    def version(self) -> int:
        return self._version

    def set_frame(self, image: QImage, version: int) -> bool:
        if image is None or image.isNull() or int(version) == self._version:
            return False
        self._image = image
        self._version = int(version)
        self.update()
        return True

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        image = self._image
        if image is not None and not image.isNull():
            source_w = max(1, image.width())
            source_h = max(1, image.height())
            target_w = max(1, self.width())
            target_h = max(1, self.height())
            scale = min(target_w / source_w, target_h / source_h)
            draw_w = max(1, round(source_w * scale))
            draw_h = max(1, round(source_h * scale))
            x = (target_w - draw_w) // 2
            y = (target_h - draw_h) // 2
            # This matches the old proven smooth wall: SmoothPixmapTransform is
            # deliberately OFF for six continuously changing feeds.
            painter.drawImage(QRect(x, y, draw_w, draw_h), image)
        painter.end()


class CameraTile(QFrame):
    def __init__(
        self,
        camera_id: str,
        settings,
        camera_meta: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.settings = settings
        self.camera_meta = dict(camera_meta or {})
        self.mmap_reader: SmoothMmapFrameReader | None = None
        self.mjpeg_reader: SmoothMjpegReader | None = None
        self._fallback_active = False
        self._wait_ticks = 0
        self._had_mmap_frame = False
        self._last_mjpeg_version = 0

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(camera_id)
        self.status = QLabel("CONNECTING")

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.status)

        self.video_host = QWidget(self)
        self.video_host.setMinimumSize(320, 180)
        self.video_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stack = QStackedLayout(self.video_host)
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.mmap_canvas = MmapVideoCanvas(self.video_host)
        self.fallback_video = QLabel("Connecting...")
        self.fallback_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback_video.setStyleSheet("background: black; color: white;")
        self.stack.addWidget(self.mmap_canvas)
        self.stack.addWidget(self.fallback_video)
        self.stack.setCurrentWidget(self.mmap_canvas)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.video_host, 1)

        if settings.video_transport == "mjpeg":
            self._start_fallback("configured mjpeg")
        else:
            self.mmap_reader = SmoothMmapFrameReader(camera_id)
            self.mmap_reader.start()

    def update_stream_profile(self, camera_meta: dict) -> None:
        self.camera_meta = dict(camera_meta or {})

    def _start_fallback(self, reason: str) -> None:
        if self._fallback_active:
            return
        self._fallback_active = True
        if self.mmap_reader is not None:
            self.mmap_reader.stop()
            self.mmap_reader = None
        self.stack.setCurrentWidget(self.fallback_video)
        self.mjpeg_reader = SmoothMjpegReader(self.camera_id, self.settings.ml_video_base_url)
        self.mjpeg_reader.start()
        self.status.setText("MJPEG FALLBACK")
        self.fallback_video.setText(reason)

    def update_tracks(self, _result: dict | None) -> None:
        # Current ByteTrack T-IDs are baked into the mmap presentation frame by
        # ml_service. API track metadata remains available for the rest of UI,
        # but drawing it again here would duplicate boxes.
        return

    def refresh(self) -> None:
        if not self._fallback_active:
            reader = self.mmap_reader
            if reader is None:
                return
            image, version = reader.latest()
            if image is not None and version >= 0:
                if self.mmap_canvas.set_frame(image, version):
                    self._had_mmap_frame = True
                    self._wait_ticks = 0
                self.status.setText(
                    f"LIVE MMAP {image.width()}x{image.height()} · {reader.last_frame_age_ms:.0f}ms"
                )
                return

            self._wait_ticks += 1
            self.status.setText("MMAP WAIT")
            # Only use HTTP fallback if mmap never became available during a
            # generous startup window. Once mmap has worked, its reader handles
            # backend inode replacement/restart itself.
            if not self._had_mmap_frame and self._wait_ticks >= 800:
                self._start_fallback("mmap frame not available")
            return

        reader = self.mjpeg_reader
        if reader is None:
            return
        image, version = reader.latest()
        if image is not None and version > self._last_mjpeg_version:
            self._last_mjpeg_version = version
            pixmap = QPixmap.fromImage(image)
            target = self.fallback_video.size()
            if target.width() > 0 and target.height() > 0:
                pixmap = pixmap.scaled(
                    target,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            self.fallback_video.setPixmap(pixmap)
            self.status.setText(f"MJPEG {reader.frames}")
        elif reader.last_error:
            self.status.setText("RECONNECTING")

    def close_reader(self) -> None:
        if self.mmap_reader is not None:
            self.mmap_reader.stop()
            self.mmap_reader = None
        if self.mjpeg_reader is not None:
            self.mjpeg_reader.stop()
            self.mjpeg_reader.join()
            self.mjpeg_reader = None


class CameraWall(QWidget):
    """Canonical six-camera wall: two per row, proven mmap presentation first."""

    COLUMNS = 2

    def __init__(self, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.tiles: dict[str, CameraTile] = {}

        for index in range(6):
            camera_id = f"CAM-{index + 1:02d}"
            tile = CameraTile(camera_id, settings, parent=self)
            row, column = divmod(index, self.COLUMNS)
            self.grid.addWidget(tile, row, column)
            self.tiles[camera_id] = tile

        for row in range(3):
            self.grid.setRowStretch(row, 1)
        for column in range(self.COLUMNS):
            self.grid.setColumnStretch(column, 1)

    def set_cameras(self, cameras: list[dict]) -> None:
        for row in cameras:
            camera_id = str(row.get("id") or "")
            tile = self.tiles.get(camera_id)
            if tile is not None:
                tile.update_stream_profile(row)

    def update_tracks(self, payload: dict) -> None:
        rows = payload.get("tracks", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            camera_id = str(row.get("camera_id") or "")
            tile = self.tiles.get(camera_id)
            if tile is not None:
                tile.update_tracks(row)

    def refresh_frames(self) -> None:
        for tile in self.tiles.values():
            tile.refresh()

    def close_readers(self) -> None:
        for tile in self.tiles.values():
            tile.close_reader()
