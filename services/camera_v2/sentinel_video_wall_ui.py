from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .sentinel_ui_base import C
from .sentinel_video_pro import ProLiveVideoWall as _BaseProLiveVideoWall
from .sentinel_video_pro import ProPipelineController


class ProLiveVideoWall(_BaseProLiveVideoWall):
    """Professional presentation wrapper around the native camera wall."""

    _LIVE_STATES = {"LIVE", "VIDEO_BOUND", "FOCUS", "HEATMAP"}
    _ERROR_STATES = {"ERROR", "STOPPED", "PIPELINE_WARNING"}

    def __init__(self, cameras, people, parent: QWidget | None = None) -> None:
        super().__init__(cameras, people, parent)

        self.tile_headers: list[QFrame] = []
        self.room_labels: list[QLabel] = []

        for sid, camera in enumerate(self.cameras[:6]):
            header = QFrame(self)
            header.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            header.setStyleSheet(
                "QFrame{background:rgba(5,10,15,224);border:0;"
                "border-bottom:1px solid rgba(53,76,91,180);border-radius:0;}"
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
                "background:transparent;border:0;color:#c5d0d8;"
                "font-size:9px;font-weight:600;padding:0;"
            )
            self.room_labels.append(room)

            if sid < len(self.camera_labels):
                self.camera_labels[sid].setStyleSheet(
                    "background:rgba(7,13,18,235);color:#eef4f8;"
                    "border:1px solid rgba(76,103,122,150);border-radius:3px;"
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
                "QFrame#cameraHoverActions{background:rgba(5,10,15,238);"
                "border:1px solid rgba(63,87,105,205);border-radius:5px;}"
                "QToolButton{background:transparent;color:#dbe5ec;border:0;"
                "border-radius:3px;padding:4px 7px;font:700 9px 'DejaVu Sans';}"
                "QToolButton:hover{background:#17242e;color:#ffffff;}"
                "QToolButton:checked{background:#0d322c;color:#39d9c5;}"
            )
            frame.adjustSize()

        # Dedicated fullscreen HUD. Grid widgets are never moved onto the focus
        # surface, preventing stale CAM/FPS copies on the native EGL window.
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

        self.state_cover = QFrame(self)
        self.state_cover.setObjectName("cameraWallStateCover")
        self.state_cover.setAttribute(Qt.WA_StyledBackground, True)
        self.state_cover.setStyleSheet(
            "QFrame#cameraWallStateCover{background:#03070b;"
            "border:1px solid #1b2934;border-radius:6px;}"
        )
        cover_layout = QVBoxLayout(self.state_cover)
        cover_layout.setContentsMargins(32, 32, 32, 32)
        cover_layout.setSpacing(8)
        cover_layout.addStretch()

        self.state_eyebrow = QLabel("CAMERA WALL")
        self.state_eyebrow.setAlignment(Qt.AlignCenter)
        self.state_eyebrow.setStyleSheet(
            f"color:{C['muted']};font:700 9px 'DejaVu Sans Mono';letter-spacing:1px;"
        )
        cover_layout.addWidget(self.state_eyebrow)

        self.state_title = QLabel("Cameras starting…")
        self.state_title.setAlignment(Qt.AlignCenter)
        self.state_title.setStyleSheet(
            f"color:{C['text']};font-size:18px;font-weight:800;"
        )
        cover_layout.addWidget(self.state_title)

        self.state_detail = QLabel("Camera sources are connecting")
        self.state_detail.setAlignment(Qt.AlignCenter)
        self.state_detail.setWordWrap(True)
        self.state_detail.setMaximumWidth(680)
        self.state_detail.setStyleSheet(
            f"color:{C['muted']};font:10px 'DejaVu Sans Mono';"
        )
        cover_layout.addWidget(self.state_detail, 0, Qt.AlignHCenter)
        cover_layout.addStretch()

        self._pipeline_state = "STARTING"
        self._layout_overlays()
        self._layout_state_cover()
        self.state_cover.show()
        self.state_cover.raise_()

    def _layout_overlays(self) -> None:
        super()._layout_overlays()
        if not hasattr(self, "tile_headers"):
            return

        if self._fullscreen_active:
            for header in self.tile_headers:
                header.hide()
            for room in self.room_labels:
                room.hide()
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
            room_x = left + 15 + cam.width()
            room.move(room_x, top + 7)
            room.show()
            room.raise_()

            stat = self.status_labels[sid]
            stat.adjustSize()
            stat.move(max(left + width // 2, left + width - stat.width() - 9), top + 7)
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

    def _layout_state_cover(self) -> None:
        if hasattr(self, "state_cover"):
            self.state_cover.setGeometry(self.rect())
            if self.state_cover.isVisible():
                self.state_cover.raise_()
        self._layout_fullscreen_hud()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_overlays()
        self._layout_state_cover()

    def set_pipeline_status(self, status) -> None:
        state = str(getattr(status, "state", "STARTING") or "STARTING").upper()
        raw_detail = str(getattr(status, "detail", "") or "").strip()
        self._pipeline_state = state

        if state in self._LIVE_STATES:
            self.state_cover.hide()
            self._layout_overlays()
            self._layout_fullscreen_hud()
            return

        self.state_cover.show()
        if state in self._ERROR_STATES:
            self.state_eyebrow.setText(
                "CAMERA WARNING" if state == "PIPELINE_WARNING" else "CAMERA ERROR"
            )
            self.state_eyebrow.setStyleSheet(
                f"color:{C['offline']};font:700 9px 'DejaVu Sans Mono';letter-spacing:1px;"
            )
            self.state_title.setText("Cameras are unavailable")
            self.state_detail.setText("Camera sources or connection need attention")
            self.state_detail.setToolTip(raw_detail)
        else:
            self.state_eyebrow.setText("CAMERA WALL")
            self.state_eyebrow.setStyleSheet(
                f"color:{C['muted']};font:700 9px 'DejaVu Sans Mono';letter-spacing:1px;"
            )
            self.state_title.setText("Cameras starting…")
            self.state_detail.setText("Camera sources are connecting")
            self.state_detail.setToolTip("")
        self._layout_state_cover()

    def set_fullscreen_mode(self, active: bool, source_id: int | None = None) -> None:
        # Do not reuse/move the grid labels onto the native fullscreen surface.
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
        self._refresh_tile_frames()
        self._layout_overlays()
        self._layout_state_cover()

    def update_metrics(self, metrics: dict) -> None:
        super().update_metrics(metrics)
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
