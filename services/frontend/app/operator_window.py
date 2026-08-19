from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
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
    "bg": "#071018",
    "sidebar": "#09131c",
    "panel": "#0b151e",
    "panel2": "#0e1a24",
    "border": "#1b2b38",
    "border2": "#233847",
    "text": "#e7eef5",
    "muted": "#8394a4",
    "faint": "#596b79",
    "accent": "#2f86f6",
    "accent2": "#5aa2ff",
    "known": "#38d996",
    "unknown": "#ffbf4b",
    "danger": "#ff6b6b",
}


APP_QSS = f"""
QMainWindow, QWidget#appRoot {{ background:{C['bg']}; color:{C['text']}; }}
QFrame#sidebar {{ background:{C['sidebar']}; border-right:1px solid {C['border']}; }}
QFrame#topbar {{ background:{C['sidebar']}; border-bottom:1px solid {C['border']}; }}
QFrame#panel {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:8px; }}
QPushButton#nav {{
    text-align:left; padding:10px 12px; border:0; border-radius:7px;
    color:{C['muted']}; background:transparent; font-size:11px; font-weight:650;
}}
QPushButton#nav:hover {{ background:#101f2b; color:{C['text']}; }}
QPushButton#nav:checked {{ background:#10263a; color:#ffffff; border-left:3px solid {C['accent']}; }}
QScrollArea {{ border:0; background:transparent; }}
QScrollBar:vertical {{ width:8px; background:transparent; }}
QScrollBar::handle:vertical {{ background:#294051; border-radius:4px; min-height:30px; }}
"""


class NavButton(QPushButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("nav")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(40)


class Badge(QLabel):
    def __init__(self, text: str = "STARTING") -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(74)
        self.setFixedHeight(26)
        self.set_state("starting", text)

    def set_state(self, state: str, text: str) -> None:
        if state == "ok":
            fg, bg, border = C["known"], "#0d211b", "#1d4a39"
        elif state == "error":
            fg, bg, border = C["danger"], "#251316", "#53262d"
        else:
            fg, bg, border = C["unknown"], "#251e10", "#55431f"
        self.setText(text)
        self.setStyleSheet(
            f"color:{fg};background:{bg};border:1px solid {border};"
            "border-radius:6px;padding:2px 8px;font:700 9px 'DejaVu Sans Mono';"
        )


class MetricBox(QFrame):
    def __init__(self, title: str, color: str) -> None:
        super().__init__()
        self.setStyleSheet("QFrame{background:#09121a;border:1px solid #182936;border-radius:7px;}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(1)
        caption = QLabel(title)
        caption.setStyleSheet(
            f"color:{C['muted']};font:700 8px 'DejaVu Sans Mono';letter-spacing:1px;"
        )
        self.value = QLabel("0")
        self.value.setStyleSheet(f"color:{color};font-size:23px;font-weight:850;")
        layout.addWidget(caption)
        layout.addWidget(self.value)


class RecentViews(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 11, 12, 11)
        layout.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("Recent Views")
        title.setStyleSheet("font-size:11px;font-weight:750;")
        self.count = QLabel("0 active")
        self.count.setStyleSheet(f"color:{C['muted']};font-size:9px;")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.count)
        layout.addLayout(head)

        self.rows = QVBoxLayout()
        self.rows.setSpacing(5)
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
            empty = QLabel("No active person tracks")
            empty.setStyleSheet(
                f"color:{C['faint']};background:#09121a;border:1px dashed #203442;"
                "border-radius:6px;padding:14px;font-size:9px;"
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rows.addWidget(empty)
            return

        for camera_id, track_id, confidence in active[:8]:
            row = QFrame()
            row.setStyleSheet("QFrame{background:#0a141d;border:1px solid #172a37;border-radius:6px;}")
            line = QHBoxLayout(row)
            line.setContentsMargins(8, 7, 8, 7)
            avatar = QLabel("T")
            avatar.setFixedSize(28, 28)
            avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            avatar.setStyleSheet(
                f"background:#13283a;color:{C['accent2']};border:1px solid #24455c;"
                "border-radius:14px;font-weight:800;"
            )
            text = QVBoxLayout()
            text.setSpacing(0)
            name = QLabel(f"Person T{track_id}")
            name.setStyleSheet("font-size:10px;font-weight:700;")
            meta = QLabel(f"{camera_id}  ·  {confidence:.2f}")
            meta.setStyleSheet(f"color:{C['muted']};font-size:8px;")
            text.addWidget(name)
            text.addWidget(meta)
            line.addWidget(avatar)
            line.addLayout(text)
            line.addStretch(1)
            self.rows.addWidget(row)


class MonitoringPage(QWidget):
    def __init__(self, settings) -> None:
        super().__init__()
        self.camera_wall = CameraWall(settings, self)
        self.camera_wall.focusChanged.connect(self._on_focus)

        self.right = QWidget(self)
        self.right.setFixedWidth(252)
        rail = QVBoxLayout(self.right)
        rail.setContentsMargins(0, 0, 0, 0)
        rail.setSpacing(8)

        summary = QFrame()
        summary.setObjectName("panel")
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(12, 11, 12, 12)
        summary_layout.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("People in Building")
        title.setStyleSheet("font-size:11px;font-weight:750;")
        self.live = Badge("STARTING")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.live)
        summary_layout.addLayout(head)

        self.total = QLabel("0")
        self.total.setStyleSheet("font-size:34px;font-weight:850;letter-spacing:-1px;")
        summary_layout.addWidget(self.total)

        metrics = QHBoxLayout()
        metrics.setSpacing(6)
        self.known = MetricBox("KNOWN", C["known"])
        self.unknown = MetricBox("UNKNOWN", C["unknown"])
        metrics.addWidget(self.known, 1)
        metrics.addWidget(self.unknown, 1)
        summary_layout.addLayout(metrics)
        rail.addWidget(summary)

        self.recent = RecentViews()
        rail.addWidget(self.recent, 1)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 8)
        root.setSpacing(8)
        root.addWidget(self.camera_wall, 1)
        root.addWidget(self.right)

    def _on_focus(self, focused: bool, _camera_id: str) -> None:
        self.right.setVisible(not focused)
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(0, 0, 0, 0) if focused else layout.setContentsMargins(8, 7, 8, 8)
            layout.setSpacing(0 if focused else 8)

    def update_ml_health(self, data: dict) -> None:
        online = int(data.get("online_camera_count") or 0)
        total = int(data.get("camera_count") or 6)
        detector = data.get("detector") or {}
        tracker = data.get("tracker") or {}
        ready = detector.get("state") == "ready" and tracker.get("state") == "ready"
        if ready and online == total and total:
            self.live.set_state("ok", "● LIVE")
        elif online:
            self.live.set_state("starting", f"● {online}/{total}")
        else:
            self.live.set_state("error", "OFFLINE")

    def update_tracks(self, payload: dict) -> None:
        self.camera_wall.update_tracks(payload)
        self.recent.update_tracks(payload)
        total = 0
        for row in payload.get("tracks", []) if isinstance(payload, dict) else []:
            if isinstance(row, dict):
                total += int(row.get("people") or len(row.get("tracks") or []))
        # ReID/Face are intentionally not enabled yet, so every active local T-ID
        # is Unknown. The counters are already wired for the later identity stage.
        self.total.setText(str(total))
        self.known.value.setText("0")
        self.unknown.value.setText(str(total))


class PeoplePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        title = QLabel("People")
        title.setStyleSheet("font-size:20px;font-weight:800;")
        subtitle = QLabel("Active camera-local tracks")
        subtitle.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.body = QWidget()
        self.rows = QVBoxLayout(self.body)
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
                row = QLabel(
                    f"Person T{int(track.get('track_id') or 0):02d}     {camera_id}     "
                    f"confidence {float(track.get('confidence') or 0.0):.2f}"
                )
                row.setStyleSheet(
                    "background:#0b151e;border:1px solid #1b2b38;border-radius:7px;"
                    "padding:11px;color:#dce6ee;font-size:10px;"
                )
                self.rows.addWidget(row)
        if not count:
            empty = QLabel("No active tracks")
            empty.setStyleSheet(f"color:{C['muted']};padding:20px;")
            self.rows.addWidget(empty)


class SimplePage(QWidget):
    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        heading = QLabel(title)
        heading.setStyleSheet("font-size:20px;font-weight:800;")
        text = QLabel(subtitle)
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        panel = QFrame()
        panel.setObjectName("panel")
        inside = QVBoxLayout(panel)
        inside.setContentsMargins(20, 20, 20, 20)
        info = QLabel(subtitle)
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet(f"color:{C['muted']};font-size:11px;")
        inside.addWidget(info, 1)
        layout.addWidget(heading)
        layout.addWidget(text)
        layout.addWidget(panel, 1)


class OperatorWindow(QMainWindow):
    PAGE_NAMES = ["Monitoring", "People", "Events", "Enrollment", "Settings"]

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
        self.sidebar.setFixedWidth(166)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(10, 13, 10, 10)
        side.setSpacing(5)
        brand = QLabel("APS  Apsidal")
        brand.setStyleSheet(
            f"color:white;font-size:15px;font-weight:850;padding:8px 8px 15px 8px;"
        )
        side.addWidget(brand)
        self.nav: list[NavButton] = []
        icons = ["▦", "◉", "⚡", "＋", "⚙"]
        for index, (icon, name) in enumerate(zip(icons, self.PAGE_NAMES)):
            button = NavButton(f"{icon}   {name}")
            button.clicked.connect(lambda _checked=False, i=index: self.switch_page(i))
            side.addWidget(button)
            self.nav.append(button)
        side.addStretch(1)
        version = QLabel("LOCAL  ·  mmap")
        version.setStyleSheet(
            f"color:{C['faint']};font:8px 'DejaVu Sans Mono';padding:8px;"
        )
        side.addWidget(version)
        root.addWidget(self.sidebar)

        self.content = QWidget()
        content = QVBoxLayout(self.content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        self.topbar = QFrame()
        self.topbar.setObjectName("topbar")
        self.topbar.setFixedHeight(55)
        top = QHBoxLayout(self.topbar)
        top.setContentsMargins(14, 0, 14, 0)
        top.setSpacing(9)
        self.page_title = QLabel("Monitoring")
        self.page_title.setStyleSheet("font-size:16px;font-weight:800;")
        top.addWidget(self.page_title)
        top.addStretch(1)
        self.api_badge = Badge("API")
        self.ai_badge = Badge("AI")
        self.cam_badge = Badge("0/6")
        top.addWidget(self.api_badge)
        top.addWidget(self.ai_badge)
        top.addWidget(self.cam_badge)
        self.clock = QLabel()
        self.clock.setStyleSheet(f"color:{C['muted']};font:9px 'DejaVu Sans Mono';margin-left:6px;")
        top.addWidget(self.clock)
        content.addWidget(self.topbar)

        self.stack = QStackedWidget()
        self.monitoring = MonitoringPage(self.settings)
        self.people = PeoplePage()
        self.events = SimplePage("Events", "Detection/event persistence will be connected through api_service; live tracking already remains independent.")
        self.enrollment = SimplePage("Enrollment", "Face enrollment stays disabled until the Face stage is explicitly enabled.")
        self.settings_page = SimplePage("Settings", f"API: {self.settings.api_base_url}\nML video fallback: {self.settings.ml_video_base_url}\nPrimary video: SIGBUS-safe mmap")
        for page in (self.monitoring, self.people, self.events, self.enrollment, self.settings_page):
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
        self.page_title.setText(self.PAGE_NAMES[index])
        for i, button in enumerate(self.nav):
            button.setChecked(i == index)

    def _on_camera_focus(self, focused: bool, _camera_id: str) -> None:
        self.sidebar.setVisible(not focused)
        self.topbar.setVisible(not focused)

    def _on_api_health(self, data: dict) -> None:
        status = str(data.get("status") or "unknown")
        self.api_badge.set_state("ok" if status == "ok" else "error", "API ●" if status == "ok" else "API ×")

    def _on_ml_health(self, data: dict) -> None:
        self.monitoring.update_ml_health(data)
        detector = data.get("detector") or {}
        tracker = data.get("tracker") or {}
        ready = detector.get("state") == "ready" and tracker.get("state") == "ready"
        self.ai_badge.set_state("ok" if ready else "starting", "AI ●" if ready else "AI …")
        online = int(data.get("online_camera_count") or 0)
        total = int(data.get("camera_count") or 6)
        self.cam_badge.set_state("ok" if online == total and total else "starting", f"{online}/{total}")

    def _on_cameras(self, data: dict) -> None:
        cameras = [row for row in data.get("cameras", []) if isinstance(row, dict)]
        self.monitoring.camera_wall.set_cameras(cameras)

    def _on_tracks(self, data: dict) -> None:
        self.monitoring.update_tracks(data)
        self.people.update_tracks(data)

    def _on_request_failed(self, request_name: str, _reason: str) -> None:
        if request_name == "api_health":
            self.api_badge.set_state("error", "API ×")
        elif request_name == "ml_health":
            self.ai_badge.set_state("error", "AI ×")
        elif request_name == "cameras":
            self.cam_badge.set_state("error", "CAM ×")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.monitoring.camera_wall.close_readers()
        super().closeEvent(event)
