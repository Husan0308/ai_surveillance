from __future__ import annotations

from PySide6.QtCore import QObject, QEvent, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy


class _GridPolishFilter(QObject):
    """Keeps all six camera cards true 16:9 and centered in the available area."""

    def __init__(self, live_page):
        super().__init__(live_page)
        self.live_page = live_page

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.apply_sizes()
        return False

    def apply_sizes(self):
        live = self.live_page
        if not getattr(live, "tiles", None):
            return

        # The title row remains outside the grid. Fit all 3 rows on-screen by
        # choosing the limiting dimension and preserving 16:9 exactly.
        available_w = max(320, live.width())
        available_h = max(240, live.height() - (0 if not live.title_row.isVisible() else live.title_row.height()) - 14)
        h_gap = live.grid.horizontalSpacing() * 1
        v_gap = live.grid.verticalSpacing() * 2

        max_tile_w_by_width = max(160.0, (available_w - h_gap) / 2.0)
        max_tile_h_by_height = max(90.0, (available_h - v_gap) / 3.0)
        max_tile_w_by_height = max_tile_h_by_height * (16.0 / 9.0)

        tile_w = int(max(160.0, min(max_tile_w_by_width, max_tile_w_by_height)))
        tile_h = int(round(tile_w * 9.0 / 16.0))

        for tile in live.tiles.values():
            tile.setMinimumSize(tile_w, tile_h)
            tile.setMaximumSize(tile_w, tile_h)



def _paint_camera(self, event):
    painter = QPainter(self)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.fillRect(self.rect(), QColor("#030915"))

    image = getattr(self, "_image", None)
    if image is None or image.isNull():
        painter.setPen(QColor("#8293aa"))
        f = QFont("Inter")
        f.setPixelSize(12)
        painter.setFont(f)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Connecting…")
        return

    iw = float(image.width())
    ih = float(image.height())
    tw = float(max(1, self.width()))
    th = float(max(1, self.height()))

    # Primary surveillance image uses CONTAIN. Because the card itself is kept
    # at 16:9, 1280x720 sources fill the card with zero crop and zero stretch.
    fit = min(tw / iw, th / ih)
    fw = iw * fit
    fh = ih * fit
    frame_rect = QRectF((tw - fw) * 0.5, (th - fh) * 0.5, fw, fh)
    painter.drawImage(frame_rect, image)

    # Subtle top/bottom readability layers. These do not cover meaningful image
    # content and keep the dashboard close to a professional NVR look.
    top_grad = QLinearGradient(0, 0, 0, 42)
    top_grad.setColorAt(0.0, QColor(2, 12, 27, 215))
    top_grad.setColorAt(1.0, QColor(2, 12, 27, 20))
    painter.fillRect(QRectF(0, 0, tw, 44), top_grad)

    bottom_grad = QLinearGradient(0, th - 40, 0, th)
    bottom_grad.setColorAt(0.0, QColor(2, 12, 27, 10))
    bottom_grad.setColorAt(1.0, QColor(2, 12, 27, 220))
    painter.fillRect(QRectF(0, max(0.0, th - 40), tw, 40), bottom_grad)

    tile = self.parentWidget()
    camera_id = getattr(tile, "camera_id", "CAM")
    title_map = getattr(__import__("services.frontend.core_v1.dashboard", fromlist=["CAMERA_TITLES"]), "CAMERA_TITLES", {})
    title = title_map.get(camera_id, camera_id)

    f = QFont("Inter")
    f.setPixelSize(12)
    f.setWeight(QFont.Weight.DemiBold)
    painter.setFont(f)
    painter.setPen(QColor("#f8fafc"))
    painter.drawText(QRectF(12, 8, tw - 90, 24), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{camera_id}  {title}")

    painter.setPen(QPen(QColor("#16e487"), 1.6))
    painter.setBrush(QColor("#16e487"))
    painter.drawEllipse(QRectF(tw - 67, 14, 7, 7))
    painter.setPen(QColor("#d8fbe9"))
    f2 = QFont("Inter")
    f2.setPixelSize(10)
    f2.setWeight(QFont.Weight.DemiBold)
    painter.setFont(f2)
    painter.drawText(QRectF(tw - 55, 7, 46, 22), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "LIVE")

    people_text = getattr(getattr(tile, "people", None), "text", lambda: "0 People")()
    fps_text = getattr(getattr(tile, "fps", None), "text", lambda: "-- FPS")()
    painter.setPen(QColor("#e7edf5"))
    painter.drawText(QRectF(12, th - 30, tw * 0.5, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, people_text)
    painter.drawText(QRectF(tw * 0.5, th - 30, tw * 0.5 - 12, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fps_text)


def install(dashboard_module):
    """Apply UI-only polish without touching ML, ReID, RTSP or detection code."""

    # True 16:9 card rendering with in-frame overlays.
    dashboard_module.CameraViewport.paintEvent = _paint_camera

    original_tile_init = dashboard_module.CameraTile.__init__
    def tile_init(self, camera_id: str, number: int):
        original_tile_init(self, camera_id, number)
        # Existing header/footer information is redrawn as overlays so the full
        # card area can be used by the camera image itself.
        self.header.hide()
        self.footer.hide()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "QFrame#cameraTile { background:#030915; border:1px solid #123557; border-radius:10px; }"
        )
    dashboard_module.CameraTile.__init__ = tile_init

    original_live_init = dashboard_module.LivePage.__init__
    def live_init(self):
        original_live_init(self)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(10)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self._polish_filter = _GridPolishFilter(self)
        self.installEventFilter(self._polish_filter)
        self._polish_filter.apply_sizes()
    dashboard_module.LivePage.__init__ = live_init

    original_sidebar_init = dashboard_module.Sidebar.__init__
    def sidebar_init(self, change_page):
        original_sidebar_init(self, change_page)
        self.setStyleSheet(
            "#sidebar { background:#061426; border-right:1px solid #102c48; }"
            "#sidebar QPushButton { color:#e7eef8; text-align:left; padding-left:16px; border-radius:8px; }"
            "#sidebar QPushButton:hover { background:#0b223c; }"
            "#sidebar QPushButton:checked { background:#0d63ff; border:1px solid #2b7bff; }"
        )
    dashboard_module.Sidebar.__init__ = sidebar_init

    original_window_init = dashboard_module.DashboardWindow.__init__
    def window_init(self):
        original_window_init(self)
        self.topbar.setFixedHeight(58)
        self.content_layout.setContentsMargins(12, 8, 12, 12)
        self.content_layout.setSpacing(12)
        self.sidebar.setFixedWidth(190)
        self.right.setFixedWidth(285)
        self.setStyleSheet(self.styleSheet() + """
            QMainWindow, QWidget { background:#020b18; color:#f5f7fb; }
            #rightPanel, #statusCard, #cameraTile, #placeholder {
                background:#07192d; border:1px solid #143a60; border-radius:10px;
            }
            #statCard { background:#0a2038; border-radius:9px; }
            #squareButton, #topButton { background:#0a2038; border:1px solid #143a60; border-radius:7px; }
        """)
    dashboard_module.DashboardWindow.__init__ = window_init
