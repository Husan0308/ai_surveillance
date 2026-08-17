from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import sentinel_exact as ui


SOURCE_WIDTH = 1280.0
SOURCE_HEIGHT = 720.0
GRID_ASPECT = 1280.0 / 1080.0  # 2 columns x 3 rows of 16:9 cameras
FOCUS_ASPECT = 16.0 / 9.0
GUTTER = 5.0


class CameraWallOverlay(QWidget):
    """Reference-style chrome above the native zero-copy DeepStream wall.

    Video stays in the GstVideoOverlay X11 surface. This top-level translucent
    window draws only the exact camera chrome, identity boxes and hover affordance,
    so Qt backing-store pixels never replace live camera pixels.
    """

    fullscreenRequested = Signal(int)

    def __init__(self, wall: "SafeLiveWall", owner: QWidget):
        super().__init__(
            owner,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.wall = wall
        self.cameras: dict[int, dict] = {}
        self.tracks: dict[int, list[dict]] = {i: [] for i in range(6)}
        self.rooms: dict[int, int] = {}
        self.focus_source: int | None = None
        self._hover_source: int | None = None
        self._fullscreen_rects: dict[int, QRectF] = {}

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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

    def set_snapshot(self, snapshot: dict) -> None:
        cameras: dict[int, dict] = {}
        for row in snapshot.get("cameras", []):
            try:
                source_id = int(row.get("source_id", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= source_id < 6:
                cameras[source_id] = dict(row)
        self.cameras = cameras

        grouped: dict[int, list[dict]] = {i: [] for i in range(6)}
        for track in snapshot.get("tracks", []):
            try:
                source_id = int(track.get("source_id", -1))
            except (TypeError, ValueError):
                continue
            if source_id in grouped:
                grouped[source_id].append(dict(track))
        self.tracks = grouped

        rooms: dict[int, int] = {}
        for row in snapshot.get("rooms", []):
            try:
                rooms[int(row.get("room_id", -1))] = int(row.get("count", 0))
            except (TypeError, ValueError):
                continue
        self.rooms = rooms
        self.update()

    def set_focus(self, source_id: int | None) -> None:
        self.focus_source = source_id
        self._hover_source = None
        self.update()

    def _content_rect(self) -> QRectF:
        """Exact rectangle where nveglglessink renders with aspect preservation."""
        bounds = QRectF(self.rect())
        if bounds.width() <= 1 or bounds.height() <= 1:
            return QRectF()
        aspect = FOCUS_ASPECT if self.focus_source is not None else GRID_ASPECT
        available_aspect = bounds.width() / bounds.height()
        if available_aspect > aspect:
            h = bounds.height()
            w = h * aspect
            return QRectF(bounds.left() + (bounds.width() - w) * 0.5, bounds.top(), w, h)
        w = bounds.width()
        h = w / aspect
        return QRectF(bounds.left(), bounds.top() + (bounds.height() - h) * 0.5, w, h)

    def _cell_rect(self, source_id: int) -> QRectF:
        content = self._content_rect()
        if content.isEmpty():
            return QRectF()
        if self.focus_source is not None:
            return content if source_id == self.focus_source else QRectF()
        tile_w = content.width() / 2.0
        tile_h = content.height() / 3.0
        row, col = divmod(source_id, 2)
        return QRectF(content.left() + col * tile_w, content.top() + row * tile_h, tile_w, tile_h)

    def _source_at(self, pos) -> int | None:
        content = self._content_rect()
        if content.isEmpty() or not content.contains(pos):
            return None
        if self.focus_source is not None:
            return self.focus_source
        local_x = pos.x() - content.left()
        local_y = pos.y() - content.top()
        col = min(1, max(0, int(local_x / (content.width() / 2.0))))
        row = min(2, max(0, int(local_y / (content.height() / 3.0))))
        return row * 2 + col

    def _draw_tracks(self, painter: QPainter, source_id: int, video_rect: QRectF) -> None:
        camera = self.cameras.get(source_id, {})
        if not camera.get("online", False):
            return

        src_w = max(1.0, float(camera.get("source_width", SOURCE_WIDTH) or SOURCE_WIDTH))
        src_h = max(1.0, float(camera.get("source_height", SOURCE_HEIGHT) or SOURCE_HEIGHT))
        sx = video_rect.width() / src_w
        sy = video_rect.height() / src_h

        font = QFont("DejaVu Sans Mono", 7)
        metrics = QFontMetrics(font)
        for track in self.tracks.get(source_id, []):
            try:
                left = float(track.get("left", 0.0))
                top = float(track.get("top", 0.0))
                width = float(track.get("width", 0.0))
                height = float(track.get("height", 0.0))
            except (TypeError, ValueError):
                continue
            if width <= 1.0 or height <= 2.0:
                continue

            x1 = video_rect.left() + left * sx
            y1 = video_rect.top() + top * sy
            x2 = video_rect.left() + (left + width) * sx
            y2 = video_rect.top() + (top + height) * sy

            x1 = max(video_rect.left() + 1.0, min(video_rect.right() - 3.0, x1))
            y1 = max(video_rect.top() + 23.0, min(video_rect.bottom() - 3.0, y1))
            x2 = max(x1 + 2.0, min(video_rect.right() - 2.0, x2))
            y2 = max(y1 + 2.0, min(video_rect.bottom() - 2.0, y2))

            known = bool(track.get("known", False))
            tone = QColor(ui.C["known"] if known else ui.C["unknown"])

            # Reference bbox: clean 2px line, no translucent fill.
            painter.setPen(QPen(tone, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            text = self._person_label(track)
            text_width = min(
                max(42, metrics.horizontalAdvance(text) + 10),
                max(42, int(video_rect.width() * 0.48)),
            )
            chip_h = 17.0
            chip_x = min(x1, max(video_rect.left() + 2.0, video_rect.right() - text_width - 2.0))
            chip_y = max(video_rect.top() + 23.0, y1 - chip_h)
            chip = QRectF(chip_x, chip_y, text_width, chip_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tone)
            painter.drawRoundedRect(chip, 2.5, 2.5)
            painter.setPen(QColor(ui.C["bg"]))
            painter.setFont(font)
            painter.drawText(
                chip.adjusted(4, 0, -4, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                text,
            )

    def _draw_camera(self, painter: QPainter, source_id: int) -> None:
        cell = self._cell_rect(source_id)
        if cell.isEmpty():
            return

        # Demo/reference grid spacing: neighboring 5px masks form a 10px gutter.
        painter.setPen(QPen(QColor(ui.C["bg"]), GUTTER * 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(cell)
        rect = cell.adjusted(GUTTER, GUTTER, -GUTTER, -GUTTER)
        if rect.width() <= 20 or rect.height() <= 20:
            return

        camera = self.cameras.get(source_id, {})
        online = bool(camera.get("online", False))
        focused = self.focus_source == source_id

        painter.setPen(QPen(QColor(ui.C["primary"] if focused else ui.C["border"]), 2 if focused else 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 6, 6)

        if not online:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(9, 13, 18, 220))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor(ui.C["offline"]))
            painter.setFont(QFont("DejaVu Sans Mono", 10, QFont.Weight.Bold))
            painter.drawText(
                QRectF(rect.left(), rect.center().y() - 25, rect.width(), 20),
                Qt.AlignmentFlag.AlignCenter,
                "▱  OFFLINE",
            )
            painter.setPen(QColor(ui.C["muted"]))
            painter.setFont(QFont("DejaVu Sans", 8))
            painter.drawText(
                QRectF(rect.left() + 20, rect.center().y() + 2, rect.width() - 40, 30),
                Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                "RTSP timeout — qayta ulanmoqda",
            )
        else:
            self._draw_tracks(painter, source_id, rect)

        painter.setFont(QFont("DejaVu Sans", 9, QFont.Weight.Bold))
        painter.setPen(QColor(ui.C["text"]))
        painter.drawText(rect.left() + 8, rect.top() + 18, ui.camera_name(source_id))

        dot = QColor(ui.C["known"] if online else ui.C["offline"])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot)
        painter.drawEllipse(QRectF(rect.right() - 69, rect.top() + 8, 7, 7))
        painter.setPen(QColor(ui.C["muted"]))
        painter.setFont(QFont("DejaVu Sans Mono", 7))
        fps = float(camera.get("fps", 0.0) or 0.0)
        painter.drawText(
            QRectF(rect.right() - 57, rect.top() + 2, 53, 20),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            f"{fps:.1f} fps" if online else "0.0 fps",
        )

        occupancy = self.rooms.get(source_id // 2 + 1, int(camera.get("count", 0) or 0))
        badge = QRectF(rect.left() + 8, rect.bottom() - 27, 54, 21)
        painter.setPen(QPen(QColor(ui.C["border"]), 1))
        painter.setBrush(QColor(8, 14, 20, 225))
        painter.drawRoundedRect(badge, 5, 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ui.C["primary"]))
        painter.drawEllipse(QRectF(badge.left() + 7, badge.top() + 6, 6, 6))
        painter.drawRoundedRect(QRectF(badge.left() + 5, badge.top() + 13, 10, 5), 2, 2)
        painter.setPen(QColor(ui.C["text"]))
        painter.setFont(QFont("DejaVu Sans Mono", 8, QFont.Weight.Bold))
        painter.drawText(
            QRectF(badge.left() + 19, badge.top(), 29, badge.height()),
            Qt.AlignmentFlag.AlignVCenter,
            str(occupancy),
        )

        if self._hover_source == source_id and not focused:
            # Only a border/shadow is painted; never wash the live video with color.
            painter.setPen(QPen(QColor(2, 12, 17, 220), 6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 7, 7)
            accent = QColor(ui.C["primary"])
            accent.setAlpha(220)
            painter.setPen(QPen(accent, 2))
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 7, 7)

            button = QRectF(rect.right() - 31, rect.top() + 26, 24, 22)
            self._fullscreen_rects[source_id] = button
            painter.setPen(QPen(QColor(ui.C["border"]), 1))
            painter.setBrush(QColor(8, 14, 20, 230))
            painter.drawRoundedRect(button, 4, 4)
            painter.setPen(QColor(ui.C["text"]))
            painter.setFont(QFont("DejaVu Sans", 10))
            painter.drawText(button, Qt.AlignmentFlag.AlignCenter, "⛶")

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._fullscreen_rects.clear()
        if self.focus_source is not None:
            self._draw_camera(painter, self.focus_source)
        else:
            for source_id in range(6):
                self._draw_camera(painter, source_id)
        painter.end()

    def mouseMoveEvent(self, event):
        source_id = self._source_at(event.position())
        if source_id != self._hover_source:
            self._hover_source = source_id
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover_source is not None:
            self._hover_source = None
            self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        source_id = self._source_at(event.position())
        if source_id is not None:
            self.fullscreenRequested.emit(source_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            source_id = self._source_at(event.position())
            button = self._fullscreen_rects.get(source_id) if source_id is not None else None
            if button is not None and button.contains(event.position()):
                self.fullscreenRequested.emit(source_id)
                event.accept()
                return
        super().mouseReleaseEvent(event)


class SafeLiveWall(QWidget):
    """Native DeepStream surface + exact reference overlay, with no frame copies."""

    fullscreenRequested = Signal(int)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.focus_source: int | None = None
        self._boot_requested = False
        self._overlay: CameraWallOverlay | None = None

        self.setMinimumSize(620, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        _ = int(self.winId())

        # One 20 Hz timer handles both overlay geometry and fresh tracker metadata.
        # MainWindow can keep its slower cards/recent-views refresh without making
        # person boxes visibly lag behind the camera.
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(50)
        self._live_timer.timeout.connect(self._live_tick)

    def _ensure_overlay(self) -> CameraWallOverlay:
        if self._overlay is None:
            owner = self.window()
            self._overlay = CameraWallOverlay(self, owner)
            self._overlay.fullscreenRequested.connect(self.fullscreenRequested)
            self._overlay.set_focus(self.focus_source)
        return self._overlay

    def _sync_overlay(self) -> None:
        overlay = self._overlay
        if overlay is None:
            return
        if not self.isVisible() or self.width() <= 1 or self.height() <= 1:
            overlay.hide()
            return
        global_pos = self.mapToGlobal(QPoint(0, 0))
        target = QRectF(global_pos.x(), global_pos.y(), self.width(), self.height()).toRect()
        if overlay.geometry() != target:
            overlay.setGeometry(target)
        if not overlay.isVisible():
            overlay.show()
        overlay.raise_()

    def _live_tick(self) -> None:
        if not self.isVisible():
            return
        overlay = self._ensure_overlay()
        try:
            overlay.set_snapshot(self.controller.snapshot())
        except Exception:
            pass
        self._sync_overlay()

    def paintEngine(self):
        return None

    def paintEvent(self, event):
        event.accept()

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.Type.WinIdChange:
            try:
                xid = int(self.winId())
                if xid > 0:
                    self.controller.bind_window(xid)
            except Exception:
                pass
        if event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show):
            QTimer.singleShot(0, self._sync_overlay)
        return result

    def showEvent(self, event):
        super().showEvent(event)
        overlay = self._ensure_overlay()
        overlay.show()
        self._live_timer.start()
        QTimer.singleShot(0, self._live_tick)
        if not self._boot_requested:
            self._boot_requested = True
            QTimer.singleShot(100, self._start_pipeline)

    def hideEvent(self, event):
        self._live_timer.stop()
        if self._overlay is not None:
            self._overlay.hide()
        super().hideEvent(event)

    def _start_pipeline(self):
        self.controller.start(int(self.winId()))
        self._live_tick()

    def resizeEvent(self, event):
        QTimer.singleShot(0, self._sync_overlay)
        super().resizeEvent(event)

    def set_focus(self, source_id: int | None):
        self.focus_source = source_id
        self.controller.set_focus_source(source_id)
        if self._overlay is not None:
            self._overlay.set_focus(source_id)
        QTimer.singleShot(0, self._sync_overlay)

    def refresh(self, snapshot: dict):
        overlay = self._ensure_overlay()
        overlay.set_snapshot(snapshot)
        self._sync_overlay()

    def closeEvent(self, event):
        self._live_timer.stop()
        if self._overlay is not None:
            self._overlay.close()
            self._overlay.deleteLater()
            self._overlay = None
        super().closeEvent(event)