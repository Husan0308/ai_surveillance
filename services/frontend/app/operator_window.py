from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from services.frontend.app.api_client import ApiClient
from services.frontend.app.camera_wall import CameraWall
from services.frontend.app.config import load_settings


C = {
    "bg": "#070b11",
    "sidebar": "#0a1118",
    "surface": "#0c141d",
    "surface2": "#101b26",
    "surface3": "#132231",
    "border": "#1b2b38",
    "border2": "#294052",
    "text": "#eef4f8",
    "muted": "#8b9baa",
    "faint": "#566775",
    "accent": "#3797ff",
    "accent_soft": "#102a42",
    "known": "#37d99a",
    "unknown": "#ffbf4b",
    "danger": "#ff6b73",
}

PAGE_META = [
    ("Monitoring", "Live cameras and active people"),
    ("People", "Current camera-local tracks"),
    ("Events", "Detection and identity events"),
    ("Enrollment", "Register people from face images"),
    ("Settings", "Application and service settings"),
]


APP_QSS = f"""
* {{
    color:{C['text']};
    font-family:'DejaVu Sans';
    font-size:12px;
}}
QMainWindow, QWidget#appRoot, QWidget#page {{
    background:{C['bg']};
}}
QFrame#sidebar {{
    background:{C['sidebar']};
    border-right:1px solid {C['border']};
}}
QFrame#brandBlock {{
    background:transparent;
    border-bottom:1px solid {C['border']};
}}
QFrame#topbar {{
    background:{C['bg']};
    border-bottom:1px solid {C['border']};
}}
QFrame#panel {{
    background:{C['surface']};
    border:1px solid {C['border']};
    border-radius:10px;
}}
QFrame#metric {{
    background:#09131c;
    border:1px solid #1a2c3a;
    border-radius:8px;
}}
QPushButton#nav {{
    text-align:left;
    padding:0 13px;
    border:1px solid transparent;
    border-radius:8px;
    color:{C['muted']};
    background:transparent;
    font-size:11px;
    font-weight:650;
}}
QPushButton#nav:hover {{
    background:{C['surface2']};
    color:{C['text']};
}}
QPushButton#nav:checked {{
    background:{C['accent_soft']};
    color:#ffffff;
    border:1px solid #1f4d71;
}}
QPushButton#soft {{
    background:{C['surface2']};
    border:1px solid {C['border2']};
    border-radius:7px;
    padding:7px 10px;
    color:{C['text']};
}}
QScrollArea {{
    border:0;
    background:transparent;
}}
QScrollBar:vertical {{
    width:8px;
    background:transparent;
    margin:2px;
}}
QScrollBar::handle:vertical {{
    background:#2a4051;
    border-radius:4px;
    min-height:32px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height:0;
}}
"""


class NavButton(QPushButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("nav")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(42)


class HealthChip(QLabel):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(27)
        self.setMinimumWidth(62)
        self.set_state("starting", "…")

    def set_state(self, state: str, value: str) -> None:
        if state == "ok":
            fg, bg, border = C["known"], "#0b211a", "#1c4b39"
        elif state == "error":
            fg, bg, border = C["danger"], "#261316", "#55282d"
        else:
            fg, bg, border = C["unknown"], "#241d0f", "#59431d"
        self.setText(f"{self.prefix}  {value}")
        self.setStyleSheet(
            f"color:{fg};background:{bg};border:1px solid {border};"
            "border-radius:7px;padding:0 9px;font:700 9px 'DejaVu Sans Mono';"
        )


class MetricBox(QFrame):
    def __init__(self, title: str, color: str) -> None:
        super().__init__()
        self.setObjectName("metric")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(2)

        caption = QLabel(title)
        caption.setStyleSheet(
            f"color:{C['muted']};font:700 8px 'DejaVu Sans Mono';letter-spacing:1px;"
        )
        self.value = QLabel("0")
        self.value.setStyleSheet(f"color:{color};font-size:24px;font-weight:850;")

        layout.addWidget(caption)
        layout.addWidget(self.value)


class RecentViews(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 13, 14, 13)
        layout.setSpacing(9)

        head = QHBoxLayout()
        title = QLabel("Recent Views")
        title.setStyleSheet("font-size:12px;font-weight:800;")
        self.count = QLabel("0 active")
        self.count.setStyleSheet(f"color:{C['muted']};font-size:9px;")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.count)
        layout.addLayout(head)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background:{C['border']};border:0;")
        layout.addWidget(divider)

        self.rows = QVBoxLayout()
        self.rows.setSpacing(6)
        layout.addLayout(self.rows)
        layout.addStretch(1)

    @staticmethod
    def _clear(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_tracks(self, payload: dict) -> None:
        self._clear(self.rows)
        active: list[tuple[str, int, float]] = []

        for camera in payload.get("tracks", []) if isinstance(payload, dict) else []:
            if not isinstance(camera, dict):
                continue
            camera_id = str(camera.get("camera_id") or "")
            for track in camera.get("tracks", []) or []:
                if not isinstance(track, dict):
                    continue
                active.append(
                    (
                        camera_id,
                        int(track.get("track_id") or 0),
                        float(track.get("confidence") or 0.0),
                    )
                )

        self.count.setText(f"{len(active)} active")

        if not active:
            empty = QLabel("No active tracks")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setMinimumHeight(72)
            empty.setStyleSheet(
                f"color:{C['faint']};background:#09121a;"
                f"border:1px dashed {C['border2']};border-radius:8px;font-size:9px;"
            )
            self.rows.addWidget(empty)
            return

        for camera_id, track_id, confidence in active[:7]:
            row = QFrame()
            row.setStyleSheet(
                "QFrame{background:#09131c;border:1px solid #182c3a;border-radius:8px;}"
                "QFrame:hover{background:#0d1a25;border-color:#29475c;}"
            )
            line = QHBoxLayout(row)
            line.setContentsMargins(9, 8, 9, 8)
            line.setSpacing(9)

            avatar = QLabel(f"T{track_id}")
            avatar.setFixedSize(32, 32)
            avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            avatar.setStyleSheet(
                f"background:#13283a;color:{C['accent']};border:1px solid #27506b;"
                "border-radius:16px;font:800 9px 'DejaVu Sans Mono';"
            )

            info = QVBoxLayout()
            info.setSpacing(2)
            name = QLabel(f"Person T{track_id}")
            name.setStyleSheet("font-size:10px;font-weight:750;")
            meta = QLabel(f"{camera_id}   confidence {confidence:.2f}")
            meta.setStyleSheet(f"color:{C['muted']};font:8px 'DejaVu Sans Mono';")
            info.addWidget(name)
            info.addWidget(meta)

            line.addWidget(avatar)
            line.addLayout(info, 1)

            dot = QLabel("●")
            dot.setStyleSheet(f"color:{C['known']};font-size:10px;")
            line.addWidget(dot)
            self.rows.addWidget(row)


class MonitoringPage(QWidget):
    def __init__(self, settings) -> None:
        super().__init__()
        self.setObjectName("page")
        self.camera_wall = CameraWall(settings, self)
        self.camera_wall.focusChanged.connect(self._on_focus)

        self.right = QWidget(self)
        self.right.setFixedWidth(286)
        rail = QVBoxLayout(self.right)
        rail.setContentsMargins(0, 0, 0, 0)
        rail.setSpacing(10)

        summary = QFrame()
        summary.setObjectName("panel")
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(15, 14, 15, 15)
        summary_layout.setSpacing(10)

        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("People in Building")
        title.setStyleSheet("font-size:12px;font-weight:800;")
        caption = QLabel("Current local tracks")
        caption.setStyleSheet(f"color:{C['muted']};font-size:9px;")
        title_box.addWidget(title)
        title_box.addWidget(caption)
        head.addLayout(title_box)
        head.addStretch(1)
        self.live = HealthChip("LIVE")
        self.live.setMinimumWidth(76)
        head.addWidget(self.live)
        summary_layout.addLayout(head)

        self.total = QLabel("0")
        self.total.setStyleSheet("font-size:38px;font-weight:850;letter-spacing:-1px;")
        summary_layout.addWidget(self.total)

        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self.known = MetricBox("KNOWN", C["known"])
        self.unknown = MetricBox("UNKNOWN", C["unknown"])
        metrics.addWidget(self.known, 1)
        metrics.addWidget(self.unknown, 1)
        summary_layout.addLayout(metrics)
        rail.addWidget(summary)

        self.recent = RecentViews()
        rail.addWidget(self.recent, 1)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(10)
        root.addWidget(self.camera_wall, 1)
        root.addWidget(self.right)

    def _on_focus(self, focused: bool, _camera_id: str) -> None:
        self.right.setVisible(not focused)
        layout = self.layout()
        if layout is not None:
            if focused:
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)
            else:
                layout.setContentsMargins(12, 10, 12, 12)
                layout.setSpacing(10)

    def update_ml_health(self, data: dict) -> None:
        online = int(data.get("online_camera_count") or 0)
        total = int(data.get("camera_count") or 6)
        detector = data.get("detector") or {}
        tracker = data.get("tracker") or {}
        ready = detector.get("state") == "ready" and tracker.get("state") == "ready"

        if ready and online == total and total:
            self.live.set_state("ok", "ON")
        elif online:
            self.live.set_state("starting", f"{online}/{total}")
        else:
            self.live.set_state("error", "OFF")

    def update_tracks(self, payload: dict) -> None:
        self.camera_wall.update_tracks(payload)
        self.recent.update_tracks(payload)

        total = 0
        for row in payload.get("tracks", []) if isinstance(payload, dict) else []:
            if isinstance(row, dict):
                total += int(row.get("people") or len(row.get("tracks") or []))

        self.total.setText(str(total))
        self.known.value.setText("0")
        self.unknown.value.setText(str(total))


class PeoplePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(14)

        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("People")
        title.setStyleSheet("font-size:19px;font-weight:850;")
        subtitle = QLabel("Active camera-local person tracks")
        subtitle.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top.addLayout(title_box)
        top.addStretch(1)
        layout.addLayout(top)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.body = QWidget()
        self.body.setObjectName("page")
        self.rows = QVBoxLayout(self.body)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(8)
        self.rows.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.body)
        layout.addWidget(self.scroll, 1)

    def update_tracks(self, payload: dict) -> None:
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        count = 0
        for camera in payload.get("tracks", []) if isinstance(payload, dict) else []:
            if not isinstance(camera, dict):
                continue
            camera_id = str(camera.get("camera_id") or "")
            for track in camera.get("tracks", []) or []:
                if not isinstance(track, dict):
                    continue
                count += 1

                row = QFrame()
                row.setObjectName("panel")
                row.setMinimumHeight(66)
                line = QHBoxLayout(row)
                line.setContentsMargins(12, 9, 12, 9)
                line.setSpacing(11)

                avatar = QLabel(f"T{int(track.get('track_id') or 0)}")
                avatar.setFixedSize(38, 38)
                avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
                avatar.setStyleSheet(
                    f"background:#13283a;color:{C['accent']};border:1px solid #27506b;"
                    "border-radius:19px;font:800 10px 'DejaVu Sans Mono';"
                )

                info = QVBoxLayout()
                info.setSpacing(2)
                name = QLabel(f"Person T{int(track.get('track_id') or 0)}")
                name.setStyleSheet("font-size:11px;font-weight:800;")
                meta = QLabel(
                    f"{camera_id}  ·  confidence {float(track.get('confidence') or 0.0):.2f}"
                )
                meta.setStyleSheet(f"color:{C['muted']};font:9px 'DejaVu Sans Mono';")
                info.addWidget(name)
                info.addWidget(meta)

                state = QLabel("UNKNOWN")
                state.setStyleSheet(
                    f"color:{C['unknown']};background:#241d0f;border:1px solid #59431d;"
                    "border-radius:6px;padding:4px 7px;font:700 8px 'DejaVu Sans Mono';"
                )

                line.addWidget(avatar)
                line.addLayout(info, 1)
                line.addWidget(state)
                self.rows.addWidget(row)

        if not count:
            empty = QFrame()
            empty.setObjectName("panel")
            empty.setMinimumHeight(130)
            e = QVBoxLayout(empty)
            message = QLabel("No active tracks")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setStyleSheet(f"color:{C['muted']};font-size:11px;")
            e.addWidget(message)
            self.rows.addWidget(empty)


class SimplePage(QWidget):
    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        self.setObjectName("page")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(14)

        heading = QLabel(title)
        heading.setStyleSheet("font-size:19px;font-weight:850;")
        text = QLabel(subtitle)
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{C['muted']};font-size:10px;")

        panel = QFrame()
        panel.setObjectName("panel")
        inside = QVBoxLayout(panel)
        inside.setContentsMargins(24, 24, 24, 24)

        icon = QLabel("◇")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"color:{C['accent']};font-size:30px;")
        info = QLabel(subtitle)
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet(f"color:{C['muted']};font-size:11px;")

        inside.addStretch(1)
        inside.addWidget(icon)
        inside.addWidget(info)
        inside.addStretch(1)

        layout.addWidget(heading)
        layout.addWidget(text)
        layout.addWidget(panel, 1)


class OperatorWindow(QMainWindow):
    PAGE_NAMES = [item[0] for item in PAGE_META]

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle("Apsidal — AI Surveillance")
        self.resize(1500, 920)
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(APP_QSS)

        root_widget = QWidget()
        root_widget.setObjectName("appRoot")
        root = QHBoxLayout(root_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(root_widget)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(188)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(10, 0, 10, 10)
        side.setSpacing(4)

        brand = QFrame()
        brand.setObjectName("brandBlock")
        brand.setFixedHeight(76)
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(12, 13, 8, 12)
        brand_layout.setSpacing(1)

        brand_name = QLabel("Apsidal")
        brand_name.setStyleSheet("color:#ffffff;font-size:17px;font-weight:900;")
        brand_sub = QLabel("AI SURVEILLANCE")
        brand_sub.setStyleSheet(
            f"color:{C['accent']};font:700 8px 'DejaVu Sans Mono';letter-spacing:1.4px;"
        )
        brand_layout.addWidget(brand_name)
        brand_layout.addWidget(brand_sub)
        side.addWidget(brand)

        nav_label = QLabel("WORKSPACE")
        nav_label.setStyleSheet(
            f"color:{C['faint']};font:700 8px 'DejaVu Sans Mono';"
            "letter-spacing:1px;padding:12px 10px 5px 10px;"
        )
        side.addWidget(nav_label)

        self.nav: list[NavButton] = []
        icons = ["▦", "◉", "⚡", "＋", "⚙"]
        for index, (icon, name) in enumerate(zip(icons, self.PAGE_NAMES)):
            button = NavButton(f"{icon}    {name}")
            button.clicked.connect(lambda _checked=False, i=index: self.switch_page(i))
            side.addWidget(button)
            self.nav.append(button)

        side.addStretch(1)

        footer = QFrame()
        footer.setStyleSheet(f"border-top:1px solid {C['border']};background:transparent;")
        footer_l = QVBoxLayout(footer)
        footer_l.setContentsMargins(10, 12, 10, 4)
        footer_l.setSpacing(3)
        local = QLabel("●  LOCAL SYSTEM")
        local.setStyleSheet(f"color:{C['known']};font:700 8px 'DejaVu Sans Mono';")
        build = QLabel("Apsidal Edge")
        build.setStyleSheet(f"color:{C['faint']};font-size:8px;")
        footer_l.addWidget(local)
        footer_l.addWidget(build)
        side.addWidget(footer)

        root.addWidget(self.sidebar)

        self.content = QWidget()
        content = QVBoxLayout(self.content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        self.topbar = QFrame()
        self.topbar.setObjectName("topbar")
        self.topbar.setFixedHeight(66)
        top = QHBoxLayout(self.topbar)
        top.setContentsMargins(20, 0, 18, 0)
        top.setSpacing(8)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self.page_title = QLabel("Monitoring")
        self.page_title.setStyleSheet("font-size:17px;font-weight:850;")
        self.page_subtitle = QLabel(PAGE_META[0][1])
        self.page_subtitle.setStyleSheet(f"color:{C['muted']};font-size:9px;")
        title_box.addWidget(self.page_title)
        title_box.addWidget(self.page_subtitle)
        top.addLayout(title_box)
        top.addStretch(1)

        self.api_badge = HealthChip("API")
        self.ai_badge = HealthChip("AI")
        self.cam_badge = HealthChip("CAM")
        self.cam_badge.setMinimumWidth(72)
        top.addWidget(self.api_badge)
        top.addWidget(self.ai_badge)
        top.addWidget(self.cam_badge)

        self.clock = QLabel()
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.clock.setMinimumWidth(78)
        self.clock.setStyleSheet(
            f"color:{C['muted']};font:9px 'DejaVu Sans Mono';margin-left:5px;"
        )
        top.addWidget(self.clock)
        content.addWidget(self.topbar)

        self.stack = QStackedWidget()
        self.monitoring = MonitoringPage(self.settings)
        self.people = PeoplePage()
        self.events = SimplePage(
            "Events",
            "Detection and event history will appear here through api_service.",
        )
        self.enrollment = SimplePage(
            "Enrollment",
            "Face enrollment remains disabled until the Face stage is explicitly enabled.",
        )
        self.settings_page = SimplePage(
            "Settings",
            "Application configuration and service connection settings.",
        )

        for page in (
            self.monitoring,
            self.people,
            self.events,
            self.enrollment,
            self.settings_page,
        ):
            self.stack.addWidget(page)

        content.addWidget(self.stack, 1)
        root.addWidget(self.content, 1)

        self.monitoring.camera_wall.focusChanged.connect(self._on_camera_focus)

        self.api = ApiClient(self.settings.api_base_url, self)
        self.api.api_health_received.connect(self._on_api_health)
        self.api.ml_health_received.connect(self._on_ml_health)
        self.api.cameras_received.connect(self._on_cameras)
        self.api.tracks_received.connect(self._on_tracks)
        self.api.request_failed.connect(self._on_request_failed)

        self.api_timer = QTimer(self)
        self.api_timer.setInterval(self.settings.refresh_interval_ms)
        self.api_timer.timeout.connect(self.api.refresh_all)
        self.api_timer.start()

        self.track_timer = QTimer(self)
        self.track_timer.setInterval(self.settings.track_refresh_interval_ms)
        self.track_timer.timeout.connect(self.api.refresh_tracks)
        self.track_timer.start()

        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(self.settings.frame_refresh_interval_ms)
        self.frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.frame_timer.timeout.connect(self.monitoring.camera_wall.refresh_frames)
        self.frame_timer.start()

        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start()
        self._tick_clock()

        self.escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.escape.activated.connect(self.monitoring.camera_wall.clear_focus)

        self.switch_page(0)
        self.api.refresh_all()
        self.api.refresh_tracks()

    def _tick_clock(self) -> None:
        self.clock.setText(datetime.now().strftime("%H:%M:%S"))

    def switch_page(self, index: int) -> None:
        if not 0 <= int(index) < self.stack.count():
            return
        self.stack.setCurrentIndex(index)
        self.page_title.setText(PAGE_META[index][0])
        self.page_subtitle.setText(PAGE_META[index][1])
        for i, button in enumerate(self.nav):
            button.setChecked(i == index)

    def _on_camera_focus(self, focused: bool, _camera_id: str) -> None:
        self.sidebar.setVisible(not focused)
        self.topbar.setVisible(not focused)

    def _on_api_health(self, data: dict) -> None:
        status = str(data.get("status") or "unknown")
        self.api_badge.set_state("ok" if status == "ok" else "error", "●" if status == "ok" else "×")

    def _on_ml_health(self, data: dict) -> None:
        self.monitoring.update_ml_health(data)
        detector = data.get("detector") or {}
        tracker = data.get("tracker") or {}
        ready = detector.get("state") == "ready" and tracker.get("state") == "ready"
        self.ai_badge.set_state("ok" if ready else "starting", "●" if ready else "…")
        online = int(data.get("online_camera_count") or 0)
        total = int(data.get("camera_count") or 6)
        state = "ok" if online == total and total else ("starting" if online else "error")
        self.cam_badge.set_state(state, f"{online}/{total}")

    def _on_cameras(self, data: dict) -> None:
        cameras = [row for row in data.get("cameras", []) if isinstance(row, dict)]
        self.monitoring.camera_wall.set_cameras(cameras)

    def _on_tracks(self, data: dict) -> None:
        self.monitoring.update_tracks(data)
        self.people.update_tracks(data)

    def _on_request_failed(self, request_name: str, _reason: str) -> None:
        if request_name == "api_health":
            self.api_badge.set_state("error", "×")
        elif request_name == "ml_health":
            self.ai_badge.set_state("error", "×")
        elif request_name == "cameras":
            self.cam_badge.set_state("error", "×")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.monitoring.camera_wall.close_readers()
        super().closeEvent(event)
