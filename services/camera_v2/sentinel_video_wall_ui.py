from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QFrame, QLabel, QWidget

from .sentinel_video_pro import ProLiveVideoWall as _BaseProLiveVideoWall
from .sentinel_video_pro import ProPipelineController


class ProLiveVideoWall(_BaseProLiveVideoWall):
    """Native-EGL-safe professional camera wall presentation layer.

    Important rule: never place a full-size opaque Qt child over the EGL video
    window after startup. Qt child backing-store pixels can remain stale over a
    native GstVideoOverlay surface even after the child is hidden. Pipeline state
    is therefore shown by the per-camera FPS/OFFLINE labels and the right rail,
    never by a wall-sized cover.
    """

    _LIVE_STATES = {"LIVE", "VIDEO_BOUND", "FOCUS", "HEATMAP"}

    def __init__(self, cameras, people, parent: QWidget | None = None) -> None:
        super().__init__(cameras, people, parent)
        self._pipeline_state = "STARTING"
        self._ever_live = False
        self.tile_headers: list[QFrame] = []
        self.room_labels: list[QLabel] = []

        for sid, camera in enumerate(self.cameras[:6]):
            header = QFrame(self)
            header.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            header.setStyleSheet(
                "QFrame{background:rgba(4,9,14,224);border:0;"
                "border-bottom:1px solid rgba(49,74,91,190);border-radius:0;}"
            )
            self.tile_headers.append(header)

            room_text = str(
                getattr(camera, "name", "")
                or getattr(camera, "room", "")
                or f"Camera {sid + 1}"
            )
            room = QLabel(room_text, self)
            room.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            room.setStyleSheet(
                "background:transparent;border:0;color:#cbd5dd;"
                "font-size:9px;font-weight:650;padding:0;"
            )
            self.room_labels.append(room)

            if sid < len(self.camera_labels):
                self.camera_labels[sid].setStyleSheet(
                    "background:rgba(6,12,18,238);color:#f1f5f8;"
                    "border:1px solid rgba(75,104,124,165);border-radius:3px;"
                    "padding:2px 6px;font:700 9px 'DejaVu Sans Mono';"
                )
                self.camera_labels[sid].adjustSize()
            if sid < len(self.status_labels):
                self.status_labels[sid].setStyleSheet(
                    "background:transparent;border:0;color:#39d995;"
                    "padding:0;font:700 9px 'DejaVu Sans Mono';"
                )
                self.status_labels[sid].adjustSize()

        for frame in self.action_frames:
            frame.setStyleSheet(
                "QFrame#cameraHoverActions{background:rgba(5,10,15,242);"
                "border:1px solid rgba(72,98,117,215);border-radius:5px;}"
                "QToolButton{background:transparent;color:#dce6ed;border:0;"
                "border-radius:3px;padding:4px 7px;font:700 9px 'DejaVu Sans';}"
                "QToolButton:hover{background:#17242e;color:#ffffff;}"
                "QToolButton:checked{background:#0d322c;color:#39d9c5;}"
            )
            frame.adjustSize()
            frame.hide()

        # Dedicated fullscreen HUD. Grid widgets are never moved onto the native
        # fullscreen surface, avoiding stale CAM/FPS copies.
        self.fullscreen_camera_label = QLabel("", self)
        self.fullscreen_camera_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.fullscreen_camera_label.setStyleSheet(
            "background:rgba(6,12,18,238);color:#f0f5f8;"
            "border:1px solid rgba(80,105,125,160);border-radius:4px;"
            "padding:4px 9px;font:700 10px 'DejaVu Sans Mono';"
        )
        self.fullscreen_camera_label.hide()

        self.fullscreen_fps_label = QLabel("", self)
        self.fullscreen_fps_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.fullscreen_fps_label.setStyleSheet(
            "background:rgba(6,12,18,232);color:#39d995;"
            "border-radius:4px;padding:4px 8px;font:700 9px 'DejaVu Sans Mono';"
        )
        self.fullscreen_fps_label.hide()

        self._layout_overlays()
        self._refresh_tile_frames()

    def _layout_overlays(self) -> None:
        super()._layout_overlays()
        if not hasattr(self, "tile_headers"):
            return

        if self._fullscreen_active:
            for header in self.tile_headers:
                header.hide()
            for room in self.room_labels:
                room.hide()
            self._layout_fullscreen_hud()
            return

        for sid in range(min(len(self.tile_headers), len(self.cameras), 6)):
            left, top, width, height = self._tile_rect(sid)

            header = self.tile_headers[sid]
            header.setGeometry(left + 1, top + 1, max(1, width - 2), 27)
            header.show()
            header.raise_()

            cam = self.camera_labels[sid]
            cam.adjustSize()
            cam.move(left + 8, top + 4)
            cam.show()
            cam.raise_()

            room = self.room_labels[sid]
            room.adjustSize()
            room.move(left + 15 + cam.width(), top + 7)
            room.show()
            room.raise_()

            stat = self.status_labels[sid]
            stat.adjustSize()
            stat.move(left + width - stat.width() - 9, top + 7)
            stat.show()
            stat.raise_()

            if sid < len(self.action_frames):
                actions = self.action_frames[sid]
                actions.adjustSize()
                actions.move(
                    left + 8,
                    max(top + 34, top + height - actions.height() - 8),
                )
                if actions.isVisible():
                    actions.raise_()

    def _refresh_tile_frames(self) -> None:
        # Let the base class draw tile borders, then enforce a strict one-hover
        # policy. This prevents multiple Heatmap/fullscreen palettes remaining
        # visible after child-widget mouse transitions.
        super()._refresh_tile_frames()
        for sid, action in enumerate(self.action_frames):
            show = (
                not self._fullscreen_active
                and self._hover_source is not None
                and sid == self._hover_source
            )
            action.setVisible(show)
            if show:
                action.raise_()

    def mouseMoveEvent(self, event) -> None:
        if self._fullscreen_active:
            self._hover_source = None
        else:
            self._hover_source = self.source_at(event.position().toPoint())
        self._refresh_tile_frames()
        self._layout_overlays()
        event.accept()

    def leaveEvent(self, event) -> None:
        # Moving from the native parent onto its action child can produce a parent
        # leave event. Keep the palette only while the global cursor is still
        # physically inside the wall; otherwise clear it.
        local = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(local):
            self._hover_source = None
            for action in self.action_frames:
                action.hide()
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
        self._layout_overlays()
        self._refresh_tile_frames()
        self._layout_fullscreen_hud()

    def set_pipeline_status(self, status) -> None:
        state = str(getattr(status, "state", "STARTING") or "STARTING").upper()
        self._pipeline_state = state
        if state in self._LIVE_STATES:
            self._ever_live = True

        # Deliberately no full-size Qt cover here. PIPELINE_WARNING is non-fatal
        # and must never hide healthy camera frames. Fatal/startup state is shown
        # by the right rail + per-camera CONNECTING/OFFLINE status instead.
        self._layout_overlays()
        self._layout_fullscreen_hud()

    def set_fullscreen_mode(self, active: bool, source_id: int | None = None) -> None:
        self._fullscreen_active = bool(active)
        self._focused_source = int(source_id) if active and source_id is not None else None
        self._hover_source = None

        for widget in self.camera_labels:
            widget.setVisible(not active)
        for widget in self.status_labels:
            widget.setVisible(not active)
        for widget in self.room_labels:
            widget.setVisible(not active)
        for header in self.tile_headers:
            header.setVisible(not active)
        for widget in self.occupancy_labels:
            widget.hide()
        for action in self.action_frames:
            action.hide()
        for borders in self.tile_borders:
            for border in borders:
                border.setVisible(not active)

        if active and self._focused_source is not None and self._focused_source < len(self.cameras):
            camera = self.cameras[self._focused_source]
            camera_id = str(
                getattr(camera, "camera_id", getattr(camera, "id", f"CAM-{self._focused_source + 1:02d}"))
            )
            self.fullscreen_camera_label.setText(camera_id)
            self.fullscreen_fps_label.setText("LIVE")
        else:
            self.fullscreen_camera_label.hide()
            self.fullscreen_fps_label.hide()

        self.exit_button.setVisible(active)
        super()._refresh_tile_frames()
        self._refresh_tile_frames()
        self._layout_overlays()
        self._layout_fullscreen_hud()

    def update_metrics(self, metrics: dict) -> None:
        super().update_metrics(metrics)
        self._refresh_tile_frames()
        self._layout_overlays()

        if self._fullscreen_active and self._focused_source is not None:
            by_source = {
                int(row.get("source_id", -1)): row
                for row in (metrics or {}).get("cameras", [])
                if isinstance(row, dict)
            }
            row = by_source.get(self._focused_source)
            if row and row.get("online"):
                self.fullscreen_fps_label.setText(f"{float(row.get('fps', 0.0)):.1f} fps")
                self.fullscreen_fps_label.setStyleSheet(
                    "background:rgba(6,12,18,232);color:#39d995;"
                    "border-radius:4px;padding:4px 8px;font:700 9px 'DejaVu Sans Mono';"
                )
            else:
                self.fullscreen_fps_label.setText("OFFLINE")
                self.fullscreen_fps_label.setStyleSheet(
                    "background:rgba(6,12,18,232);color:#f06464;"
                    "border-radius:4px;padding:4px 8px;font:700 9px 'DejaVu Sans Mono';"
                )
            self._layout_fullscreen_hud()
