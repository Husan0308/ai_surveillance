from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from services.frontend.app.mjpeg_reader import SmoothMjpegReader
from services.frontend.app.native_video import NativeShmRenderer


class TrackOverlay(QWidget):
    """Sharp vector overlay; coordinates are in the ML display-frame space."""

    def __init__(self, source_width: int, source_height: int, parent: QWidget) -> None:
        super().__init__(parent)
        self.source_width = max(1, int(source_width))
        self.source_height = max(1, int(source_height))
        self.tracks: list[dict] = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

    def set_tracks(self, payload: dict | None) -> None:
        result = dict(payload or {})
        age_ms = result.get("age_ms")
        if age_ms is not None and float(age_ms) > 1200.0:
            self.tracks = []
        else:
            self.tracks = [row for row in result.get("tracks", []) if isinstance(row, dict)]
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self.tracks or self.width() <= 1 or self.height() <= 1:
            return

        scale = min(self.width() / self.source_width, self.height() / self.source_height)
        draw_w = self.source_width * scale
        draw_h = self.source_height * scale
        offset_x = (self.width() - draw_w) * 0.5
        offset_y = (self.height() - draw_h) * 0.5

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 214, 64))
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.setFont(QFont("DejaVu Sans", 10, QFont.Weight.DemiBold))

        for row in self.tracks:
            xyxy = list(row.get("xyxy") or ())
            if len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            left = offset_x + x1 * scale
            top = offset_y + y1 * scale
            width = max(1.0, (x2 - x1) * scale)
            height = max(1.0, (y2 - y1) * scale)
            painter.drawRect(int(left), int(top), int(width), int(height))

            track_id = int(row.get("track_id") or 0)
            confidence = float(row.get("confidence") or 0.0)
            text = f"T{track_id} {confidence:.2f}"
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(text) + 10
            text_h = metrics.height() + 4
            label_top = max(0, int(top) - text_h)
            painter.fillRect(int(left), label_top, text_w, text_h, QColor(8, 14, 20, 210))
            painter.drawText(int(left) + 5, label_top + text_h - 5, text)

        painter.end()


class CameraTile(QFrame):
    def __init__(self, camera_id: str, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.settings = settings
        self.renderer: NativeShmRenderer | None = None
        self.reader: SmoothMjpegReader | None = None
        self.last_version = 0
        self._native_attempts = 0
        self._fallback_active = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(camera_id)
        self.status = QLabel("CONNECTING")

        self.video_host = QWidget(self)
        self.video_host.setMinimumSize(320, 180)
        self.video_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stack = QStackedLayout(self.video_host)
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.native_surface = QWidget(self.video_host)
        self.native_surface.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.native_surface.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.native_surface.setStyleSheet("background: black;")
        _ = int(self.native_surface.winId())

        self.overlay = TrackOverlay(
            settings.source_width,
            settings.source_height,
            self.native_surface,
        )
        self.overlay.setGeometry(self.native_surface.rect())
        self.overlay.raise_()

        self.fallback_video = QLabel("Connecting...")
        self.fallback_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback_video.setStyleSheet("background: black; color: white;")
        self.stack.addWidget(self.native_surface)
        self.stack.addWidget(self.fallback_video)
        self.stack.setCurrentWidget(self.native_surface)

        header = QGridLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title, 0, 0)
        header.addWidget(self.status, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.video_host, 1)

        if settings.video_transport == "mjpeg":
            self._start_fallback("configured mjpeg")
        else:
            QTimer.singleShot(0, self._try_start_native)

    def _try_start_native(self) -> None:
        if self.renderer is not None or self._fallback_active:
            return
        self._native_attempts += 1
        socket_path = Path(self.settings.shm_video_dir) / f"{self.camera_id}.sock"
        try:
            renderer = NativeShmRenderer(
                self.camera_id,
                int(self.native_surface.winId()),
                socket_path,
                self.settings.source_width,
                self.settings.source_height,
                self.settings.source_fps,
                gpu_id=self.settings.gpu_id,
            )
            renderer.start()
            self.renderer = renderer
            self.stack.setCurrentWidget(self.native_surface)
            self.status.setText("LIVE NATIVE")
            self.overlay.raise_()
        except Exception as exc:
            if self._native_attempts < 12:
                self.status.setText("SHM WAIT")
                QTimer.singleShot(500, self._try_start_native)
            else:
                self._start_fallback(str(exc))

    def _start_fallback(self, reason: str) -> None:
        if self._fallback_active:
            return
        if self.renderer is not None:
            self.renderer.stop()
            self.renderer = None
        self._fallback_active = True
        self.stack.setCurrentWidget(self.fallback_video)
        self.reader = SmoothMjpegReader(self.camera_id, self.settings.ml_video_base_url)
        self.reader.start()
        self.status.setText("MJPEG FALLBACK")
        self.fallback_video.setText(reason)

    def update_tracks(self, result: dict | None) -> None:
        self.overlay.set_tracks(result)
        self.overlay.raise_()

    def refresh(self) -> None:
        renderer = self.renderer
        if renderer is not None:
            state = renderer.poll()
            if state == "error":
                self._start_fallback(renderer.last_error or "native renderer error")
                return
            self.status.setText("LIVE NATIVE" if state == "playing" else "NATIVE STARTING")
            self.overlay.raise_()
            return

        reader = self.reader
        if reader is None:
            return
        image, version = reader.latest()
        if image is not None and version > self.last_version:
            self.last_version = version
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
            return

        if reader.last_error:
            self.status.setText("RECONNECTING")
            if self.fallback_video.pixmap() is None:
                self.fallback_video.setText(reader.last_error)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_overlay)

    def _sync_overlay(self) -> None:
        self.overlay.setGeometry(self.native_surface.rect())
        self.overlay.raise_()
        if self.renderer is not None:
            self.renderer.expose()

    def close_reader(self) -> None:
        if self.renderer is not None:
            self.renderer.stop()
            self.renderer = None
        if self.reader is not None:
            self.reader.stop()
            self.reader.join()
            self.reader = None


class CameraWall(QWidget):
    """Canonical six-camera wall: two cameras per row, native shm video first."""

    COLUMNS = 2

    def __init__(self, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
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
            tile = CameraTile(camera_id, self.settings, self)
            self.tiles[camera_id] = tile
            row, column = divmod(index, self.COLUMNS)
            self.grid.addWidget(tile, row, column)

    def update_tracks(self, payload: dict) -> None:
        for result in payload.get("tracks", []) or []:
            if not isinstance(result, dict):
                continue
            camera_id = str(result.get("camera_id") or "")
            tile = self.tiles.get(camera_id)
            if tile is not None:
                tile.update_tracks(result)

    def refresh_frames(self) -> None:
        for tile in self.tiles.values():
            tile.refresh()

    def close_readers(self) -> None:
        for tile in self.tiles.values():
            tile.close_reader()
