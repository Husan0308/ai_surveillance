from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QFrame, QLabel, QWidget

from .sentinel_video_pro import ProLiveVideoWall as _BaseProLiveVideoWall
from .sentinel_video_pro import ProPipelineController


class ProLiveVideoWall(_BaseProLiveVideoWall):
    """Professional wall with EGL video isolated from Qt chrome.

    GStreamer renders only into LiveVideoWall.video_surface. Camera headers,
    fullscreen HUD and hover actions are separate native sibling windows, so Qt
    repainting them cannot clear the EGL frame underneath.
    """

    _LIVE_STATES = {"LIVE", "VIDEO_BOUND", "FOCUS", "HEATMAP"}

    def __init__(self, cameras, people, parent: QWidget | None = None) -> None:
        super().__init__(cameras, people, parent)
        self._pipeline_state = "STARTING"
        self._pipeline_detail = ""
        self._ever_live = False
        self._fullscreen_status_key: tuple[str, str] | None = None
        self.tile_headers: list[QFrame] = []
        self.room_labels: list[QLabel] = []

        self.video_surface.lower()

        for sid, camera in enumerate(self.cameras[:6]):
            header = QFrame(self)
            header.setObjectName("cameraNativeHeader")
            header.setAttribute(Qt.WA_NativeWindow, True)
            header.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            header.setStyleSheet(
                "QFrame#cameraNativeHeader{background:#04090e;border:0;"
                "border-bottom:1px solid #314a5b;border-radius:0;}"
            )
            self.tile_headers.append(header)

            cam = self.camera_labels[sid]
            cam.setParent(header)
            cam.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            cam.setStyleSheet(
                "background:#061018;color:#f1f5f8;"
                "border:1px solid #4b687c;border-radius:3px;"
                "padding:2px 6px;font:700 9px 'DejaVu Sans Mono';"
            )
            cam.adjustSize()

            room_text = str(
                getattr(camera, "name", "")
                or getattr(camera, "room", "")
                or f"Camera {sid + 1}"
            )
            room = QLabel(room_text, header)
            room.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            room.setStyleSheet(
                "background:transparent;border:0;color:#cbd5dd;"
                "font-size:9px;font-weight:650;padding:0;"
            )
            room.adjustSize()
            self.room_labels.append(room)

            stat = self.status_labels[sid]
            stat.setParent(header)
            stat.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            stat.setStyleSheet(
                "background:transparent;border:0;color:#39d995;"
                "padding:0;font:700 9px 'DejaVu Sans Mono';"
            )
            stat.adjustSize()

        # Heatmap/fullscreen controls live at the upper-right of the tile, directly
        # below its header. They are native siblings so the EGL child cannot cover them.
        for frame in self.action_frames:
            frame.setAttribute(Qt.WA_NativeWindow, True)
            frame.setStyleSheet(
                "QFrame#cameraHoverActions{background:#050a0f;"
                "border:1px solid #486275;border-radius:5px;}"
                "QToolButton{background:transparent;color:#dce6ed;border:0;"
                "border-radius:3px;padding:4px 7px;font:700 9px 'DejaVu Sans';}"
                "QToolButton:hover{background:#17242e;color:#ffffff;}"
                "QToolButton:checked{background:#0d322c;color:#39d9c5;}"
            )
            frame.adjustSize()
            frame.hide()

        self.exit_button.setAttribute(Qt.WA_NativeWindow, True)

        self.fullscreen_camera_label = QLabel("", self)
        self.fullscreen_camera_label.setAttribute(Qt.WA_NativeWindow, True)
        self.fullscreen_camera_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.fullscreen_camera_label.setStyleSheet(
            "background:#061018;color:#f0f5f8;"
            "border:1px solid #50697d;border-radius:4px;"
            "padding:4px 9px;font:700 10px 'DejaVu Sans Mono';"
        )
        self.fullscreen_camera_label.hide()

        self.fullscreen_fps_label = QLabel("", self)
        self.fullscreen_fps_label.setAttribute(Qt.WA_NativeWindow, True)
        self.fullscreen_fps_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.fullscreen_fps_label.setStyleSheet(
            "background:#061018;color:#39d995;"
            "border-radius:4px;padding:4px 8px;font:700 9px 'DejaVu Sans Mono';"
        )
        self.fullscreen_fps_label.hide()

        self._layout_overlays()
        self._refresh_tile_frames()

    def _layout_overlays(self) -> None:
        super()._layout_overlays()
        if not hasattr(self, "tile_headers"):
            return

        self.video_surface.lower()

        if self._fullscreen_active:
            for header in self.tile_headers:
                header.hide()
            for action in self.action_frames:
                action.hide()
            self._layout_fullscreen_hud()
            return

        for sid in range(min(len(self.tile_headers), len(self.cameras), 6)):
            left, top, width, _height = self._tile_rect(sid)

            header = self.tile_headers[sid]
            header.setGeometry(left + 1, top + 1, max(1, width - 2), 27)
            header.show()
            header.raise_()

            cam = self.camera_labels[sid]
            cam.adjustSize()
            cam.move(7, 4)
            cam.show()

            room = self.room_labels[sid]
            room.adjustSize()
            room.move(14 + cam.width(), 7)
            room.show()

            stat = self.status_labels[sid]
            stat.adjustSize()
            stat.move(max(8, header.width() - stat.width() - 8), 7)
            stat.show()

            if sid < len(self.action_frames):
                actions = self.action_frames[sid]
                actions.adjustSize()
                actions.move(
                    max(left + 8, left + width - actions.width() - 8),
                    top + 34,
                )
                if actions.isVisible():
                    actions.raise_()

    def _refresh_tile_frames(self) -> None:
        # Native video is the background; avoid 24 extra border child windows.
        for borders in self.tile_borders:
            for border in borders:
                border.hide()

        for sid, action in enumerate(self.action_frames):
            show = (
                not self._fullscreen_active
                and self._hover_source is not None
                and sid == self._hover_source
            )
            if action.isVisible() != show:
                action.setVisible(show)
            if show:
                action.raise_()

        if self.exit_button.isVisible():
            self.exit_button.raise_()

    def mouseMoveEvent(self, event) -> None:
        source = None if self._fullscreen_active else self.source_at(event.position().toPoint())
        if source != self._hover_source:
            self._hover_source = source
            self._refresh_tile_frames()
            self._layout_overlays()
        event.accept()

    def leaveEvent(self, event) -> None:
        local = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(local) and self._hover_source is not None:
            self._hover_source = None
            self._refresh_tile_frames()
        event.accept()

    def _layout_fullscreen_hud(self) -> None:
        if not hasattr(self, "fullscreen_camera_label"):
            return
        if not self._fullscreen_active or self._focused_source is None:
            self.fullscreen_camera_label.hide()
            self.fullscreen_fps_label.hide()
            return

        self.fullscreen_camera_label.adjustSize()
        self.fullscreen_camera_label.move(12, 12)
        self.fullscreen_camera_label.show()
        self.fullscreen_camera_label.raise_()

        self.fullscreen_fps_label.adjustSize()
        self.fullscreen_fps_label.move(
            max(12, self.width() - self.fullscreen_fps_label.width() - 14),
            12,
        )
        self.fullscreen_fps_label.show()
        self.fullscreen_fps_label.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.video_surface.setGeometry(self.rect())
        self.video_surface.lower()
        self._layout_overlays()
        self._refresh_tile_frames()
        self._layout_fullscreen_hud()

    def set_pipeline_status(self, status) -> None:
        state = str(getattr(status, "state", "STARTING") or "STARTING").upper()
        detail = str(getattr(status, "detail", "") or "")
        if state == self._pipeline_state and detail == self._pipeline_detail:
            return

        self._pipeline_state = state
        self._pipeline_detail = detail
        if state in self._LIVE_STATES:
            self._ever_live = True

    def set_fullscreen_mode(self, active: bool, source_id: int | None = None) -> None:
        self._fullscreen_active = bool(active)
        self._focused_source = int(source_id) if active and source_id is not None else None
        self._hover_source = None
        self._fullscreen_status_key = None

        for header in self.tile_headers:
            header.setVisible(not active)
        for action in self.action_frames:
            action.hide()
        for widget in self.occupancy_labels:
            widget.hide()
        for borders in self.tile_borders:
            for border in borders:
                border.hide()

        if active and self._focused_source is not None and self._focused_source < len(self.cameras):
            camera = self.cameras[self._focused_source]
            camera_id = str(
                getattr(
                    camera,
                    "camera_id",
                    getattr(camera, "id", f"CAM-{self._focused_source + 1:02d}"),
                )
            )
            self.fullscreen_camera_label.setText(camera_id)
            self.fullscreen_fps_label.setText("LIVE")
        else:
            self.fullscreen_camera_label.hide()
            self.fullscreen_fps_label.hide()

        self.exit_button.setVisible(active)
        self.video_surface.lower()
        self._refresh_tile_frames()
        self._layout_overlays()
        self._layout_fullscreen_hud()

    def update_metrics(self, metrics: dict) -> None:
        super().update_metrics(metrics)

        if self._fullscreen_active and self._focused_source is not None:
            by_source = {
                int(row.get("source_id", -1)): row
                for row in (metrics or {}).get("cameras", [])
                if isinstance(row, dict)
            }
            row = by_source.get(self._focused_source)
            if row and row.get("online"):
                text = f"{int(round(float(row.get('fps', 0.0))))} fps"
                color = "#39d995"
            else:
                text = "OFFLINE"
                color = "#f06464"

            key = (text, color)
            if key == self._fullscreen_status_key:
                return
            self._fullscreen_status_key = key

            self.fullscreen_fps_label.setText(text)
            self.fullscreen_fps_label.setStyleSheet(
                f"background:#061018;color:{color};"
                "border-radius:4px;padding:4px 8px;font:700 9px 'DejaVu Sans Mono';"
            )
            self._layout_fullscreen_hud()
