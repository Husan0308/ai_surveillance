from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtNetwork import QNetworkAccessManager
from PySide6.QtWidgets import QFrame, QGridLayout, QSizePolicy, QWidget

from services.frontend.app.mjpeg_reader import MjpegStream
from services.frontend.app.realtime_models import (
    CameraMetadata,
    CameraRow,
    letterbox_rect,
    map_bbox_to_widget,
)


_BG = QColor("#090d12")
_PANEL = QColor("#0e151d")
_BORDER = QColor("#22303e")
_TEXT = QColor("#e7edf3")
_MUTED = QColor("#7e8c99")
_LIVE = QColor("#3ddc97")
_PREDICTED = QColor("#f6b94b")
_OFFLINE = QColor("#f06464")


def _short_track_id(track_id: str) -> str:
    marker = "-T"
    if marker in track_id:
        return "T" + track_id.rsplit(marker, 1)[1]
    return track_id


class CameraTile(QFrame):
    def __init__(
        self,
        camera: CameraRow,
        api_base_url: str,
        manager: QNetworkAccessManager,
        decode_interval_ms: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera = camera
        self.metadata = CameraMetadata(
            source_width=camera.width,
            source_height=camera.height,
            online=camera.online,
            fps=camera.fps,
            last_error=camera.last_error,
        )
        self.pixmap = QPixmap()
        self.stream_state = "CONNECTING"
        self.setMinimumSize(330, 205)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: transparent; border: 0;")

        self.stream = MjpegStream(
            camera.camera_id,
            api_base_url,
            camera.stream_url,
            manager,
            decode_interval_ms=decode_interval_ms,
            parent=self,
        )
        self.stream.frame_received.connect(self._on_frame)
        self.stream.state_changed.connect(self._on_stream_state)
        self.stream.start()

    def update_camera(self, camera: CameraRow) -> None:
        self.camera = camera
        self.metadata.online = camera.online
        self.metadata.fps = camera.fps
        self.metadata.last_error = camera.last_error
        if self.metadata.source_width <= 1 and camera.width > 1:
            self.metadata.source_width = camera.width
        if self.metadata.source_height <= 1 and camera.height > 1:
            self.metadata.source_height = camera.height
        self.update()

    def update_metadata(self, metadata: CameraMetadata) -> None:
        self.metadata = metadata
        self.update()

    def _on_frame(self, image: QImage) -> None:
        self.pixmap = QPixmap.fromImage(image)
        self.update()

    def _on_stream_state(self, state: str) -> None:
        self.stream_state = state
        self.update()

    def close_stream(self) -> None:
        self.stream.stop()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(_BORDER, 1))
        painter.setBrush(_PANEL)
        painter.drawRoundedRect(outer, 7, 7)

        header_h = 34.0
        painter.setPen(_TEXT)
        painter.setFont(QFont("DejaVu Sans", 9, QFont.Weight.Bold))
        painter.drawText(QRectF(12, 4, self.width() - 150, 25), Qt.AlignmentFlag.AlignVCenter, self.camera.camera_id)

        online = bool(self.metadata.online)
        status_color = _LIVE if online else _OFFLINE
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(status_color)
        painter.drawEllipse(QRectF(self.width() - 103, 13, 7, 7))
        painter.setPen(_MUTED)
        painter.setFont(QFont("DejaVu Sans Mono", 8))
        status_text = f"{self.metadata.fps:.1f} fps" if online else "OFFLINE"
        painter.drawText(QRectF(self.width() - 91, 5, 78, 24), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, status_text)

        area = QRectF(5, header_h, max(1.0, self.width() - 10.0), max(1.0, self.height() - header_h - 5.0))
        painter.fillRect(area, QColor("#020406"))

        if not self.pixmap.isNull():
            dx, dy, dw, dh, _ = letterbox_rect(
                area.x(), area.y(), area.width(), area.height(), self.pixmap.width(), self.pixmap.height()
            )
            draw_rect = QRectF(dx, dy, dw, dh)
            painter.drawPixmap(draw_rect, self.pixmap, QRectF(self.pixmap.rect()))

            bbox_map = letterbox_rect(
                area.x(),
                area.y(),
                area.width(),
                area.height(),
                self.metadata.source_width,
                self.metadata.source_height,
            )
            painter.save()
            painter.setClipRect(draw_rect)
            for track in self.metadata.tracks:
                x1, y1, x2, y2 = map_bbox_to_widget(track.bbox_xyxy, bbox_map)
                rect = QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
                predicted = track.state.lower() in {"lost", "predicted", "coasting"}
                tone = _PREDICTED if predicted else _LIVE
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(tone, 2.2))
                painter.drawRect(rect)

                label_text = f"PERSON · {_short_track_id(track.track_id)}"
                painter.setFont(QFont("DejaVu Sans Mono", 8, QFont.Weight.Bold))
                text_w = painter.fontMetrics().horizontalAdvance(label_text) + 10
                tag_h = 19.0
                tag_y = max(draw_rect.top() + 2.0, rect.top() - tag_h)
                tag = QRectF(rect.left(), tag_y, min(float(text_w), max(20.0, draw_rect.right() - rect.left())), tag_h)
                painter.fillRect(tag, tone)
                painter.setPen(_BG)
                painter.drawText(tag.adjusted(5, 0, -3, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label_text)
            painter.restore()
        else:
            painter.setPen(_MUTED)
            painter.setFont(QFont("DejaVu Sans Mono", 9))
            message = "Connecting to API stream..."
            if not online and self.metadata.last_error:
                message = self.metadata.last_error
            painter.drawText(area.adjusted(20, 20, -20, -20), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, message)


class CameraWall(QWidget):
    def __init__(
        self,
        api_base_url: str,
        manager: QNetworkAccessManager,
        decode_interval_ms: int = 33,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.api_base_url = api_base_url
        self.manager = manager
        self.decode_interval_ms = decode_interval_ms
        self.tiles: dict[str, CameraTile] = {}
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(10)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)

    def set_cameras(self, cameras: list[CameraRow]) -> None:
        incoming = {camera.camera_id: camera for camera in cameras}
        for camera_id in list(self.tiles):
            if camera_id not in incoming:
                tile = self.tiles.pop(camera_id)
                tile.close_stream()
                self.grid.removeWidget(tile)
                tile.deleteLater()

        for camera in cameras:
            tile = self.tiles.get(camera.camera_id)
            if tile is None:
                tile = CameraTile(
                    camera,
                    self.api_base_url,
                    self.manager,
                    self.decode_interval_ms,
                    self,
                )
                self.tiles[camera.camera_id] = tile
            else:
                tile.update_camera(camera)

        for tile in self.tiles.values():
            self.grid.removeWidget(tile)
        for index, camera in enumerate(cameras):
            self.grid.addWidget(self.tiles[camera.camera_id], index // 2, index % 2)
        for row in range(max(1, (len(cameras) + 1) // 2)):
            self.grid.setRowStretch(row, 1)

    def update_metadata(self, camera_id: str, metadata: CameraMetadata) -> None:
        tile = self.tiles.get(camera_id)
        if tile is not None:
            tile.update_metadata(metadata)

    def close_streams(self) -> None:
        for tile in self.tiles.values():
            tile.close_stream()
