from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from services.frontend.app.mjpeg_reader import SmoothMjpegReader
from services.frontend.app.mmap_frame_reader import SmoothMmapFrameReader


BG = "#071018"
CARD = "#0b151e"
CARD_HEAD = "#0d1822"
BORDER = "#1b2b38"
BORDER_LIVE = "#21445a"
TEXT = "#e7eef5"
MUTED = "#8293a3"
GREEN = "#38d996"
AMBER = "#ffca54"


class MmapVideoCanvas(QWidget):
    doubleClicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._version = -1
        # Restore the old acd673ba smooth-wall policy: six continuously changing
        # tiles use the cheap painter path. Only a single focused camera enables
        # Qt's SmoothPixmapTransform filter.
        self._smooth_scaling = False
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    @property
    def version(self) -> int:
        return self._version

    def set_smooth_scaling(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._smooth_scaling:
            return
        self._smooth_scaling = enabled
        self.update()

    def set_frame(self, image: QImage, version: int) -> bool:
        if image is None or image.isNull() or int(version) == self._version:
            return False
        self._image = image
        self._version = int(version)
        self.update()
        return True

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.doubleClicked.emit()
        event.accept()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#020609"))
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

            if self._smooth_scaling:
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(QRect(x, y, draw_w, draw_h), image)
        painter.end()


class CameraTile(QFrame):
    fullscreenRequested = Signal(str)

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
        self._track_count = 0
        self._focused = False

        self.setObjectName("cameraTile")
        self.setStyleSheet(
            f"QFrame#cameraTile{{background:{CARD};border:1px solid {BORDER};border-radius:7px;}}"
        )

        header_widget = QWidget(self)
        header_widget.setFixedHeight(31)
        header_widget.setStyleSheet(f"background:{CARD_HEAD};border:0;border-radius:6px;")
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(10, 0, 7, 0)
        header.setSpacing(7)

        self.title = QLabel(camera_id)
        self.title.setStyleSheet(f"color:{TEXT};font-size:10px;font-weight:750;")
        self.status = QLabel("CONNECTING")
        self.status.setStyleSheet(
            f"color:{AMBER};font:700 8px 'DejaVu Sans Mono';letter-spacing:.4px;"
        )
        self.fullscreen = QToolButton()
        self.fullscreen.setText("⛶")
        self.fullscreen.setToolTip("Fullscreen camera")
        self.fullscreen.setFixedSize(25, 23)
        self.fullscreen.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fullscreen.setStyleSheet(
            "QToolButton{color:#b9c7d2;background:#101d27;border:1px solid #223543;"
            "border-radius:5px;font-size:13px;}"
            "QToolButton:hover{color:white;background:#193044;border-color:#31526a;}"
        )
        self.fullscreen.clicked.connect(lambda: self.fullscreenRequested.emit(self.camera_id))

        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.status)
        header.addWidget(self.fullscreen)

        self.video_host = QWidget(self)
        self.video_host.setMinimumSize(320, 180)
        self.video_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stack = QStackedLayout(self.video_host)
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.mmap_canvas = MmapVideoCanvas(self.video_host)
        self.mmap_canvas.doubleClicked.connect(
            lambda: self.fullscreenRequested.emit(self.camera_id)
        )
        self.fallback_video = QLabel("Connecting…")
        self.fallback_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback_video.setStyleSheet("background:#020609;color:#8293a3;")
        self.stack.addWidget(self.mmap_canvas)
        self.stack.addWidget(self.fallback_video)
        self.stack.setCurrentWidget(self.mmap_canvas)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(header_widget)
        layout.addWidget(self.video_host, 1)

        if settings.video_transport == "mjpeg":
            self._start_fallback("configured MJPEG")
        else:
            self.mmap_reader = SmoothMmapFrameReader(camera_id)
            self.mmap_reader.start()

    def set_presentation_mode(self, *, active: bool, focused: bool) -> None:
        self._focused = bool(focused)
        # Old smooth six-wall: normal tiles use the cheap transform. A single
        # focused camera may use HQ scaling because the other five readers pause.
        self.mmap_canvas.set_smooth_scaling(bool(focused))
        if self.mmap_reader is not None:
            self.mmap_reader.set_active(bool(active))

    def update_stream_profile(self, camera_meta: dict) -> None:
        self.camera_meta = dict(camera_meta or {})
        online = bool(self.camera_meta.get("online"))
        if not online and self._had_mmap_frame:
            self.status.setText("OFFLINE")
            self.status.setStyleSheet("color:#ff6b6b;font:700 8px 'DejaVu Sans Mono';")

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
        self.status.setText("FALLBACK")
        self.status.setStyleSheet(f"color:{AMBER};font:700 8px 'DejaVu Sans Mono';")
        self.fallback_video.setText(reason)

    def update_tracks(self, result: dict | None) -> None:
        if not isinstance(result, dict):
            return
        rows = result.get("tracks") or []
        self._track_count = len(rows) if isinstance(rows, list) else int(result.get("people") or 0)

    def refresh(self) -> None:
        if not self._fallback_active:
            reader = self.mmap_reader
            if reader is None or not reader.active:
                return
            image, version = reader.latest()
            if image is not None and version >= 0:
                if self.mmap_canvas.set_frame(image, version):
                    self._had_mmap_frame = True
                    self._wait_ticks = 0
                self.status.setText("● LIVE")
                self.status.setStyleSheet(
                    f"color:{GREEN};font:700 8px 'DejaVu Sans Mono';letter-spacing:.4px;"
                )
                self.setToolTip(
                    f"{self.camera_id} · {image.width()}x{image.height()} · "
                    f"frame age {reader.last_frame_age_ms:.0f} ms · people {self._track_count}"
                )
                return

            self._wait_ticks += 1
            self.status.setText("WAIT")
            self.status.setStyleSheet(f"color:{AMBER};font:700 8px 'DejaVu Sans Mono';")
            if not self._had_mmap_frame and self._wait_ticks >= 800:
                self._start_fallback("mmap frame unavailable")
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
                transform = (
                    Qt.TransformationMode.SmoothTransformation
                    if self._focused
                    else Qt.TransformationMode.FastTransformation
                )
                pixmap = pixmap.scaled(
                    target,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    transform,
                )
            self.fallback_video.setPixmap(pixmap)
            self.status.setText("FALLBACK")
        elif reader.last_error:
            self.status.setText("RECONNECT")

    def close_reader(self) -> None:
        if self.mmap_reader is not None:
            self.mmap_reader.stop()
            self.mmap_reader = None
        if self.mjpeg_reader is not None:
            self.mjpeg_reader.stop()
            self.mjpeg_reader.join()
            self.mjpeg_reader = None


class CameraWall(QWidget):
    """Two-camera-per-row wall with old smooth latest-only mmap presentation."""

    COLUMNS = 2
    focusChanged = Signal(bool, str)

    def __init__(self, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._focused_camera: str | None = None
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(7)
        self.tiles: dict[str, CameraTile] = {}

        for index in range(6):
            camera_id = f"CAM-{index + 1:02d}"
            tile = CameraTile(camera_id, settings, parent=self)
            tile.fullscreenRequested.connect(self.toggle_focus)
            row, column = divmod(index, self.COLUMNS)
            self.grid.addWidget(tile, row, column)
            self.tiles[camera_id] = tile

        for row in range(3):
            self.grid.setRowStretch(row, 1)
        for column in range(self.COLUMNS):
            self.grid.setColumnStretch(column, 1)

    @property
    def focused_camera(self) -> str | None:
        return self._focused_camera

    def _apply_presentation_policy(self) -> None:
        focused = self._focused_camera
        for camera_id, tile in self.tiles.items():
            active = focused is None or camera_id == focused
            tile.set_presentation_mode(
                active=active,
                focused=focused is not None and camera_id == focused,
            )

    def _rebuild_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()

        if self._focused_camera:
            tile = self.tiles[self._focused_camera]
            tile.show()
            self.grid.setContentsMargins(0, 0, 0, 0)
            self.grid.setSpacing(0)
            self.grid.addWidget(tile, 0, 0, 3, self.COLUMNS)
            self._apply_presentation_policy()
            return

        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(7)
        for index, camera_id in enumerate(sorted(self.tiles)):
            tile = self.tiles[camera_id]
            tile.show()
            row, column = divmod(index, self.COLUMNS)
            self.grid.addWidget(tile, row, column)
        self._apply_presentation_policy()

    def toggle_focus(self, camera_id: str) -> None:
        camera_id = str(camera_id)
        if self._focused_camera == camera_id:
            self._focused_camera = None
        elif camera_id in self.tiles:
            self._focused_camera = camera_id
        self._rebuild_grid()
        self.focusChanged.emit(bool(self._focused_camera), self._focused_camera or "")

    def clear_focus(self) -> None:
        if self._focused_camera is None:
            return
        self._focused_camera = None
        self._rebuild_grid()
        self.focusChanged.emit(False, "")

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
                result = row.get("result") if isinstance(row.get("result"), dict) else row
                tile.update_tracks(result)

    def refresh_frames(self) -> None:
        if self._focused_camera:
            self.tiles[self._focused_camera].refresh()
            return
        for tile in self.tiles.values():
            tile.refresh()

    def close_readers(self) -> None:
        for tile in self.tiles.values():
            tile.close_reader()
