from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time
from collections import deque

if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .detection import _yolo_worker
from .qt_runtime_v2 import CameraQtRuntimeV2

ROOMS = [
    ("dev", "Dev room", ("CAM-01", "CAM-04")),
    ("entrance", "Entrance", ("CAM-02", "CAM-05")),
    ("main", "Main room", ("CAM-03", "CAM-06")),
]
CAMERA_TITLES = {
    "CAM-01": "Dev room 2",
    "CAM-04": "Dev room 1",
    "CAM-02": "Entrance 2",
    "CAM-05": "Entrance 1",
    "CAM-03": "Main room 1",
    "CAM-06": "Main room 2",
}


class RuntimeController:
    def __init__(self, runtime: CameraQtRuntimeV2) -> None:
        self.runtime = runtime
        self.loop_thread: threading.Thread | None = None
        self.started = False
        self.stopping = False

    def start(self) -> None:
        if self.started:
            return
        r = self.runtime
        self.loop_thread = threading.Thread(target=r.loop.run, name="camera-v2-glib-loop", daemon=True)
        self.loop_thread.start()

        result = r.pipeline.set_state(r.Gst.State.PLAYING)
        if result == r.Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("Qt DeepStream pipeline failed to enter PLAYING")

        ctx = mp.get_context("spawn")
        r.job_q = ctx.Queue(maxsize=1)
        r.result_q = ctx.Queue(maxsize=2)
        r.worker = ctx.Process(target=_yolo_worker, args=(r.job_q, r.result_q), daemon=True)
        r.worker.start()
        r.scheduler_thread = threading.Thread(
            target=r._scheduler,
            name="camera-v2-yolo-scheduler",
            daemon=True,
        )
        r.scheduler_thread.start()
        self.started = True
        print(
            "CAMERA_QT_V2 started: Qt main-loop + real DeepStream pipeline; "
            "6xRTSP/NVDEC->mux->YOLO/NvDCF->demux->6xOSD/EGL",
            flush=True,
        )

    def stop(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        r = self.runtime
        try:
            r.det_stop.set()
            r._clear_requests()
        except Exception:
            pass
        try:
            r.mailbox.close()
        except Exception:
            pass
        if r.job_q is not None:
            try:
                r.job_q.put_nowait(None)
            except Exception:
                pass
        try:
            r.pipeline.set_state(r.Gst.State.NULL)
        except Exception:
            pass
        try:
            r.stop()
        except Exception:
            pass
        if self.loop_thread is not None:
            self.loop_thread.join(timeout=2.0)
        if r.scheduler_thread is not None:
            r.scheduler_thread.join(timeout=2.0)
        if r.worker is not None:
            r.worker.join(timeout=3.0)
            if r.worker.is_alive():
                r.worker.terminate()
                r.worker.join(timeout=1.0)
        for tee, pad in getattr(r, "tee_request_pads", []):
            try:
                tee.release_request_pad(pad)
            except Exception:
                pass
        try:
            r.release_qt_pads()
        except Exception:
            pass
        for pad in getattr(r, "_request_pads", []):
            try:
                r.mux.release_request_pad(pad)
            except Exception:
                pass
        self.started = False


class NativeVideoSurface(QWidget):
    resized = Signal(str, int, int)

    def __init__(self, camera_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 203)
        self.setStyleSheet("background:#030507;border:0;")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        size = event.size()
        self.resized.emit(self.camera_id, size.width(), size.height())


class StatCard(QFrame):
    def __init__(self, label: str, value: str, hint: str, tone: str = "default") -> None:
        super().__init__()
        self.setObjectName("statCard")
        self.setMinimumHeight(98)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(15, 12, 15, 12)
        lay.setSpacing(2)
        title = QLabel(label)
        title.setObjectName("statLabel")
        self.value = QLabel(value)
        self.value.setObjectName("statValue" + tone.capitalize())
        self.hint = QLabel(hint)
        self.hint.setObjectName("statHint")
        lay.addWidget(title)
        lay.addWidget(self.value)
        lay.addWidget(self.hint)
        lay.addStretch(1)

    def set_value(self, value: str, hint: str | None = None) -> None:
        self.value.setText(value)
        if hint is not None:
            self.hint.setText(hint)


class CameraCard(QFrame):
    def __init__(self, camera_id: str, room_name: str) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.setObjectName("cameraCard")
        self.setMinimumWidth(410)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top = QFrame(self)
        top.setObjectName("cameraTop")
        top.setFixedHeight(42)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(11, 4, 10, 4)
        tl.setSpacing(8)
        names = QVBoxLayout()
        names.setSpacing(0)
        name = QLabel(CAMERA_TITLES.get(camera_id, camera_id))
        name.setObjectName("cameraName")
        room = QLabel(room_name)
        room.setObjectName("cameraRoom")
        names.addWidget(name)
        names.addWidget(room)
        tl.addLayout(names)
        tl.addStretch(1)
        self.people = QLabel("0 person")
        self.people.setObjectName("cameraMeta")
        self.dot = QLabel("●")
        self.dot.setObjectName("cameraDotOff")
        self.fps = QLabel("OFFLINE")
        self.fps.setObjectName("cameraMeta")
        tl.addWidget(self.people)
        tl.addWidget(self.dot)
        tl.addWidget(self.fps)
        root.addWidget(top)
        self.surface = NativeVideoSurface(camera_id, self)
        root.addWidget(self.surface, 1)
        self.set_card_width(600)

    def set_card_width(self, width: int) -> None:
        video_h = max(230, int(max(410, width) * 9 / 16))
        self.setFixedHeight(42 + video_h)

    def update_live(self, fps: float, people: int) -> None:
        online = fps > 2.0
        obj = "cameraDotOn" if online else "cameraDotOff"
        if self.dot.objectName() != obj:
            self.dot.setObjectName(obj)
            self.dot.style().unpolish(self.dot)
            self.dot.style().polish(self.dot)
        self.fps.setText(f"{fps:.1f} FPS" if online else "OFFLINE")
        self.people.setText(f"{people} person")


class NavButton(QPushButton):
    def __init__(self, text: str, active: bool = False) -> None:
        super().__init__(text)
        self.setObjectName("navActive" if active else "navButton")
        self.setMinimumHeight(36)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class SentinelWindow(QMainWindow):
    runtime_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sentinel VMS · Monitoring")
        self.setMinimumSize(1280, 760)
        self.runtime: CameraQtRuntimeV2 | None = None
        self.controller: RuntimeController | None = None
        self._pipeline_started = False
        self._boot_started = False
        self._ui_only = os.environ.get("CAMERA_V2_UI_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._last_status_t = time.monotonic()
        self._previous_frames: dict[str, int] = {cid: 0 for cid in CAMERA_TITLES}
        self._fps_samples = {cid: deque(maxlen=4) for cid in CAMERA_TITLES}
        self.camera_cards: dict[str, CameraCard] = {}
        self.room_badges: dict[str, QLabel] = {}
        self._build_ui()
        self._apply_style()
        self.runtime_failed.connect(self._show_runtime_error)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._refresh_realtime)
        self.timer.start()

    def start_after_show(self) -> None:
        if self._boot_started:
            return
        self._boot_started = True
        if self._ui_only:
            self.pipeline_badge.setText("UI ONLY · PIPELINE OFF")
            self.status_text.setText("UI layout test · real pipeline intentionally disabled")
            return
        QTimer.singleShot(120, self._bootstrap_runtime)

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("root")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = QFrame(root)
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(224)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)
        brand = QFrame(sidebar)
        brand.setObjectName("brandBox")
        brand.setFixedHeight(64)
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(16, 9, 12, 9)
        shield = QLabel("⬢")
        shield.setObjectName("shield")
        bl.addWidget(shield)
        bt = QVBoxLayout()
        bt.setSpacing(0)
        bn = QLabel("SENTINEL VMS")
        bn.setObjectName("brandName")
        bs = QLabel("person analytics · 6 cam")
        bs.setObjectName("brandSub")
        bt.addWidget(bn)
        bt.addWidget(bs)
        bl.addLayout(bt)
        bl.addStretch(1)
        side.addWidget(brand)

        nav = QWidget(sidebar)
        nl = QVBoxLayout(nav)
        nl.setContentsMargins(8, 8, 8, 8)
        nl.setSpacing(2)
        nl.addWidget(NavButton("▣   Monitoring", True))
        for text in (
            "♙   People", "⌁   Events", "▤   Rooms", "⚙   Kameralar",
            "+   Enrollment", "◇   Diagnostics", "▥   Reports",
        ):
            b = NavButton(text)
            b.setEnabled(False)
            b.setToolTip("Backend moduli ulanganda aktiv bo‘ladi")
            nl.addWidget(b)
        nl.addStretch(1)
        side.addWidget(nav, 1)
        build = QLabel("build 2026.08 · edge worker")
        build.setObjectName("buildText")
        build.setFixedHeight(40)
        side.addWidget(build)
        shell.addWidget(sidebar)

        content = QWidget(root)
        content.setObjectName("content")
        right = QVBoxLayout(content)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        header = QFrame(content)
        header.setObjectName("header")
        header.setFixedHeight(64)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 8, 18, 8)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("Monitoring")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Jonli oqim · 3×2 kameralar · xonalar bo‘yicha guruhlangan")
        subtitle.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hl.addLayout(title_box)
        hl.addStretch(1)
        self.heat_button = QPushButton("♨  Heatmap")
        self.heat_button.setObjectName("heatToggle")
        self.heat_button.setCheckable(True)
        self.heat_button.setMinimumHeight(34)
        self.heat_button.toggled.connect(self._toggle_heatmap)
        hl.addWidget(self.heat_button)
        self.full_button = QPushButton("⛶  Fullscreen")
        self.full_button.setObjectName("headerButton")
        self.full_button.setMinimumHeight(34)
        self.full_button.clicked.connect(self._toggle_fullscreen)
        hl.addWidget(self.full_button)
        right.addWidget(header)

        scroll = QScrollArea(content)
        scroll.setObjectName("mainScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        page.setObjectName("page")
        pl = QVBoxLayout(page)
        pl.setContentsMargins(24, 20, 24, 28)
        pl.setSpacing(0)
        stats = QGridLayout()
        stats.setHorizontalSpacing(12)
        self.stat_people = StatCard("HOZIR KO‘RINMOQDA", "0", "real NvDCF current tracks")
        self.stat_online = StatCard("ONLINE KAMERA", "0 / 6", "real RTSP / NVDEC", "known")
        self.stat_detector = StatCard("PERSON DETECTOR", "0.0 Hz", "YOLO26m / camera")
        self.stat_offline = StatCard("OFFLINE KAMERA", "6", "real source cadence", "unknown")
        for i, card in enumerate((self.stat_people, self.stat_online, self.stat_detector, self.stat_offline)):
            stats.addWidget(card, 0, i)
            stats.setColumnStretch(i, 1)
        pl.addLayout(stats)

        live = QHBoxLayout()
        live.setContentsMargins(0, 22, 0, 10)
        label = QLabel("LIVE GRID")
        label.setObjectName("liveGridLabel")
        live.addWidget(label)
        live.addStretch(1)
        self.status_text = QLabel("Qt UI ready · pipeline waiting")
        self.status_text.setObjectName("statusText")
        live.addWidget(self.status_text)
        self.pipeline_badge = QLabel("UI READY")
        self.pipeline_badge.setObjectName("pipelineBadge")
        live.addWidget(self.pipeline_badge)
        pl.addLayout(live)

        for room_key, room_name, cids in ROOMS:
            heading = QHBoxLayout()
            heading.setContentsMargins(0, 7, 0, 7)
            room_title = QLabel(room_name)
            room_title.setObjectName("roomTitle")
            badge = QLabel("2 kamera · 0 person")
            badge.setObjectName("roomBadge")
            self.room_badges[room_key] = badge
            heading.addWidget(room_title)
            heading.addWidget(badge)
            heading.addStretch(1)
            pl.addLayout(heading)
            row = QHBoxLayout()
            row.setSpacing(12)
            for cid in cids:
                card = CameraCard(cid, room_name)
                self.camera_cards[cid] = card
                card.surface.resized.connect(self._surface_resized)
                row.addWidget(card, 1)
            pl.addLayout(row)
            pl.addSpacing(16)
        pl.addStretch(1)
        scroll.setWidget(page)
        right.addWidget(scroll, 1)
        shell.addWidget(content, 1)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
        QWidget#root, QWidget#content, QWidget#page, QScrollArea#mainScroll { background:#080b10; color:#d9e1e8; }
        QScrollArea#mainScroll > QWidget > QWidget { background:#080b10; }
        QScrollBar:vertical { width:8px; background:#080b10; }
        QScrollBar::handle:vertical { background:#28323d; min-height:32px; border-radius:4px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        QFrame#sidebar { background:#0b1017; border-right:1px solid #202833; }
        QFrame#brandBox { border-bottom:1px solid #202833; }
        QLabel#shield { color:#5ed6a8; font-size:19px; }
        QLabel#brandName { color:#edf2f6; font-size:13px; font-weight:700; letter-spacing:1px; }
        QLabel#brandSub { color:#667486; font-family:monospace; font-size:9px; }
        QLabel#buildText { color:#556171; border-top:1px solid #202833; padding-left:12px; font-family:monospace; font-size:9px; }
        QPushButton#navButton, QPushButton#navActive { border:0; border-radius:5px; text-align:left; padding:0 12px; background:transparent; color:#8d9aa8; font-size:12px; }
        QPushButton#navButton:hover { background:#111922; color:#dce4eb; }
        QPushButton#navActive { background:#13201e; color:#65d6aa; font-weight:600; }
        QPushButton#navButton:disabled { color:#46515e; }
        QFrame#header { background:#090d12; border-bottom:1px solid #202833; }
        QLabel#pageTitle { color:#eef3f7; font-size:17px; font-weight:700; }
        QLabel#pageSubtitle { color:#687687; font-size:10px; }
        QPushButton#headerButton, QPushButton#heatToggle { background:#0d131a; color:#9eabb8; border:1px solid #283442; border-radius:5px; padding:6px 11px; font-size:10px; font-weight:600; }
        QPushButton#headerButton:hover, QPushButton#heatToggle:hover { color:#eef3f7; border-color:#445266; }
        QPushButton#heatToggle:checked { background:#19352c; border-color:#356b57; color:#71ddb3; }
        QFrame#statCard, QFrame#cameraCard { background:#0d1218; border:1px solid #242d38; border-radius:6px; }
        QLabel#statLabel { color:#738091; font-family:monospace; font-size:9px; letter-spacing:1px; }
        QLabel#statValueDefault, QLabel#statValueKnown, QLabel#statValueUnknown { color:#e9eff4; font-size:25px; font-weight:700; }
        QLabel#statValueKnown { color:#66d7aa; }
        QLabel#statValueUnknown { color:#e8b75c; }
        QLabel#statHint { color:#596777; font-size:9px; }
        QLabel#liveGridLabel { color:#788596; font-family:monospace; font-size:10px; font-weight:700; letter-spacing:1px; }
        QLabel#statusText { color:#596777; font-family:monospace; font-size:9px; margin-right:8px; }
        QLabel#pipelineBadge { color:#75d9b1; background:#102018; border:1px solid #244c3c; border-radius:4px; padding:5px 9px; font-family:monospace; font-size:9px; }
        QLabel#roomTitle { color:#e4eaf0; font-size:13px; font-weight:600; }
        QLabel#roomBadge { color:#687585; background:#111821; border-radius:4px; padding:3px 7px; font-family:monospace; font-size:9px; }
        QFrame#cameraTop { background:#0b1016; border-bottom:1px solid #232c37; }
        QLabel#cameraName { color:#e7edf2; font-size:11px; font-weight:600; }
        QLabel#cameraRoom { color:#687585; font-family:monospace; font-size:9px; }
        QLabel#cameraMeta { color:#748191; font-family:monospace; font-size:9px; }
        QLabel#cameraDotOn { color:#59d49f; font-size:11px; }
        QLabel#cameraDotOff { color:#e15a60; font-size:11px; }
        """)

    def _surface_resized(self, cid: str, width: int, height: int) -> None:
        if self.runtime is not None:
            self.runtime.update_render_rectangle(cid, width, height)

    def _native_handles(self) -> dict[str, int]:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        handles: dict[str, int] = {}
        for cid, card in self.camera_cards.items():
            card.surface.show()
            card.surface.createWinId()
            handle = int(card.surface.winId())
            if handle <= 0:
                raise RuntimeError(f"{cid}: Qt native child window was not created")
            handles[cid] = handle
        return handles

    def _bootstrap_runtime(self) -> None:
        try:
            self.pipeline_badge.setText("BUILDING PIPELINE")
            self.status_text.setText("DeepStream graph · tracker · demux · 6 sinks")
            QApplication.processEvents()
            handles = self._native_handles()
            runtime = CameraQtRuntimeV2()
            runtime.set_heatmap_visible(self.heat_button.isChecked())
            runtime.bind_window_handles(handles)
            self.runtime = runtime
            self.controller = RuntimeController(runtime)
            for cid, card in self.camera_cards.items():
                runtime.update_render_rectangle(cid, card.surface.width(), card.surface.height())
            self.controller.start()
            self._pipeline_started = True
            self.pipeline_badge.setText("VIDEO STARTING")
            self.status_text.setText("RTSP/NVDEC starting · detector warms independently")
        except Exception as exc:
            self.runtime_failed.emit(str(exc))

    def _toggle_heatmap(self, checked: bool) -> None:
        self.heat_button.setText("♨  Heatmap ON" if checked else "♨  Heatmap")
        if self.runtime is not None:
            self.runtime.set_heatmap_visible(bool(checked))

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
            self.full_button.setText("⛶  Fullscreen")
        else:
            self.showFullScreen()
            self.full_button.setText("×  Exit full")

    @staticmethod
    def _recent_rate(times: deque[float], now: float, horizon: float = 5.0) -> float:
        recent = [t for t in times if now - t <= horizon]
        if len(recent) < 2:
            return 0.0
        span = max(0.2, recent[-1] - recent[0])
        return (len(recent) - 1) / span

    def _refresh_realtime(self) -> None:
        if self.runtime is None:
            return
        runtime = self.runtime
        now = time.monotonic()
        elapsed = max(0.25, now - self._last_status_t)
        self._last_status_t = now
        online = 0
        room_counts = {key: 0 for key, _, _ in ROOMS}
        for cid, stat in runtime.stats.items():
            previous = self._previous_frames.get(cid, stat.frames)
            delta = max(0, stat.frames - previous)
            self._previous_frames[cid] = stat.frames
            instant = delta / elapsed
            self._fps_samples[cid].append(instant)
            fps = sum(self._fps_samples[cid]) / max(1, len(self._fps_samples[cid]))
            if fps > 2.0:
                online += 1
            people = runtime.camera_person_count(cid)
            card = self.camera_cards.get(cid)
            if card:
                card.update_live(fps, people)
            for key, _name, cids in ROOMS:
                if cid in cids:
                    room_counts[key] += people
                    break
        with runtime.det_lock:
            tracked = int(runtime.tracked_now)
            ready = bool(runtime.det_ready)
            error = str(runtime.det_error or "")
            result_age = float(getattr(runtime, "detector_result_age_ms", 0.0))
        rates = [self._recent_rate(rows, now) for rows in runtime.detector_times.values()]
        avg_hz = sum(rates) / len(rates) if rates else 0.0
        self.stat_people.set_value(str(tracked), "real current-frame NvDCF tracks")
        self.stat_online.set_value(f"{online} / 6", "real RTSP / NVDEC")
        self.stat_detector.set_value(f"{avg_hz:.1f} Hz", f"result age {result_age:.0f} ms")
        self.stat_offline.set_value(str(6 - online), "real source cadence")
        for key, _name, cids in ROOMS:
            self.room_badges[key].setText(f"{len(cids)} kamera · {room_counts[key]} person")
        if error:
            self.pipeline_badge.setText("DETECTOR ERROR")
            self.status_text.setText(error[:90])
        elif online == 6 and ready:
            self.pipeline_badge.setText("LIVE · 6/6")
            self.status_text.setText(f"YOLO {avg_hz:.1f} Hz/cam · NvDCF · heat {'ON' if runtime.heatmap_visible() else 'hidden/accumulating'}")
        elif online > 0:
            self.pipeline_badge.setText(f"VIDEO · {online}/6")
            self.status_text.setText("video live · YOLO warming" if not ready else "partial camera connectivity")

    def _show_runtime_error(self, message: str) -> None:
        self._pipeline_started = False
        self.pipeline_badge.setText("PIPELINE ERROR")
        self.status_text.setText(message[:120])
        print(f"CAMERA_QT_V2 ERROR {message}", file=sys.stderr, flush=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(40, self._sync_card_sizes)

    def _sync_card_sizes(self) -> None:
        for card in self.camera_cards.values():
            card.set_card_width(max(410, card.width()))
            self._surface_resized(card.camera_id, card.surface.width(), card.surface.height())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_F11:
            self._toggle_fullscreen()
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showMaximized()
            self.full_button.setText("⛶  Fullscreen")
            return
        if event.key() == Qt.Key.Key_H:
            self.heat_button.toggle()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.controller is not None:
            self.controller.stop()
        event.accept()


def main() -> int:
    mp.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Camera V2")
    app.setFont(QFont("Sans Serif", 10))
    window = SentinelWindow()
    window.showMaximized()
    QTimer.singleShot(0, window.start_after_show)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
