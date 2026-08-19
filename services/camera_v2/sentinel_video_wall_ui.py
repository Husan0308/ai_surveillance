from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .sentinel_ui_base import C
from .sentinel_video_pro import ProLiveVideoWall as _BaseProLiveVideoWall
from .sentinel_video_pro import ProPipelineController


class ProLiveVideoWall(_BaseProLiveVideoWall):
    """Presentation-safe wrapper around the native DeepStream video wall.

    The Gst/EGL sink paints directly into a native Qt window. Until that sink is
    PLAYING the native surface can retain old X11 pixels from the previously
    visible stacked page. Keep an opaque Qt cover above the native surface until
    the pipeline is genuinely usable, so stale People/Settings content can never
    appear inside Monitoring camera tiles.
    """

    _LIVE_STATES = {"LIVE", "VIDEO_BOUND", "FOCUS", "HEATMAP", "PIPELINE_WARNING"}
    _ERROR_STATES = {"ERROR", "STOPPED"}

    def __init__(self, cameras, people, parent: QWidget | None = None) -> None:
        super().__init__(cameras, people, parent)

        # Make per-camera hover actions CCTV-sized instead of large app buttons.
        for frame in self.action_frames:
            frame.setStyleSheet(
                "QFrame#cameraHoverActions{background:rgba(5,10,15,238);"
                "border:1px solid rgba(63,87,105,205);border-radius:5px;}"
                "QToolButton{background:transparent;color:#dbe5ec;border:0;"
                "border-radius:3px;padding:4px 7px;font:700 10px 'DejaVu Sans';}"
                "QToolButton:hover{background:#17242e;color:#ffffff;}"
                "QToolButton:checked{background:#0d322c;color:#39d9c5;}"
            )
            frame.adjustSize()

        # Opaque state layer. It deliberately starts visible before nativeReady.
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

        self.state_eyebrow = QLabel("DEEPSTREAM CAMERA WALL")
        self.state_eyebrow.setAlignment(Qt.AlignCenter)
        self.state_eyebrow.setStyleSheet(
            f"color:{C['muted']};font:700 9px 'DejaVu Sans Mono';letter-spacing:1px;"
        )
        cover_layout.addWidget(self.state_eyebrow)

        self.state_title = QLabel("Starting cameras…")
        self.state_title.setAlignment(Qt.AlignCenter)
        self.state_title.setStyleSheet(
            f"color:{C['text']};font-size:18px;font-weight:800;"
        )
        cover_layout.addWidget(self.state_title)

        self.state_detail = QLabel("DeepStream / NVDEC / NvDCF ishga tushmoqda")
        self.state_detail.setAlignment(Qt.AlignCenter)
        self.state_detail.setWordWrap(True)
        self.state_detail.setMaximumWidth(680)
        self.state_detail.setStyleSheet(
            f"color:{C['muted']};font:10px 'DejaVu Sans Mono';"
        )
        cover_layout.addWidget(self.state_detail, 0, Qt.AlignHCenter)
        cover_layout.addStretch()

        self._pipeline_state = "STARTING"
        self._layout_state_cover()
        self.state_cover.show()
        self.state_cover.raise_()

    def _layout_state_cover(self) -> None:
        if hasattr(self, "state_cover"):
            self.state_cover.setGeometry(self.rect())
            if self.state_cover.isVisible():
                self.state_cover.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_state_cover()

    def set_pipeline_status(self, status) -> None:
        state = str(getattr(status, "state", "STARTING") or "STARTING").upper()
        detail = str(getattr(status, "detail", "") or "").strip()
        self._pipeline_state = state

        if state in self._LIVE_STATES:
            self.state_cover.hide()
            return

        self.state_cover.show()
        if state in self._ERROR_STATES:
            self.state_eyebrow.setText("CAMERA PIPELINE ERROR")
            self.state_eyebrow.setStyleSheet(
                f"color:{C['offline']};font:700 9px 'DejaVu Sans Mono';letter-spacing:1px;"
            )
            self.state_title.setText("Camera wall ishga tushmadi")
            self.state_detail.setText(detail or "Pipeline process stopped")
        else:
            self.state_eyebrow.setText("DEEPSTREAM CAMERA WALL")
            self.state_eyebrow.setStyleSheet(
                f"color:{C['muted']};font:700 9px 'DejaVu Sans Mono';letter-spacing:1px;"
            )
            self.state_title.setText("Starting cameras…")
            self.state_detail.setText(detail or "DeepStream / NVDEC / NvDCF ishga tushmoqda")
        self._layout_state_cover()

    def set_fullscreen_mode(self, active: bool, source_id: int | None = None) -> None:
        super().set_fullscreen_mode(active, source_id)
        self._layout_state_cover()
