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
from .qt_runtime import CameraQtRuntime


# Actual room grouping from the six office feeds currently used by camera_v2.
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
CAMERA_ROOM = {cid: room_name for _, room_name, cids in ROOMS for cid in cids}


class RuntimeController:
    def __init__(self, runtime: CameraQtRuntime) -> None:
        self.runtime = runtime
        self.loop_thread: threading.Thread | None = None
        self.started = False
        self.stopping = False

    def start(self) -> None:
        if self.started:
            return
        r = self.runtime
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

        result = r.pipeline.set_state(r.Gst.State.PLAYING)
        if result == r.Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("Qt DeepStream pipeline failed to enter PLAYING")

        self.loop_thread = threading.Thread(
            target=r.loop.run,
            name="camera-v2-glib-loop",
            daemon=True,
        )
        self.loop_thread.start()
        self.started = True
        print(
            "CAMERA_QT started: real-time 6-card UI; "
            "NVDEC->mux->YOLO/NvDCF->demux->6xOSD/EGL; fake_streams=0",
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
            r.stop()
        except Exception:
            pass
        try:
            r.pipeline.set_state(r.Gst.State.NULL)
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
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 205)
        self.setStyleSheet("background:#030507;border:0;")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        s = event.size()
        self.resized.emit(self.camera_id, s.width(), s.height())


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "—", hint: str = "", tone: str = "default") -> None:
        super().__init__()
        self.setObjectName("statCard")
        self.setMinimumHeight(105)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 13)
        layout.setSpacing(3)
        self.label = QLabel(label.upper())
        self.label.setObjectName("statLabel")
        self.value = QLabel(value)
        self.value.setObjectName(f"statValue_{tone}")
        self.hint = QLabel(hint)
        self.hint.setObjectName("statHint")
        layout.addWidget(self.label)
        layout.addWidget(self.value)
        layout.addWidget(self.hint)
        layout.addStretch(1)

    def set_value(self, value: str, hint: str | None = None) -> None:
        self.value.setText(value)
        if hint is not None:
            self.hint.setText(hint)


class CameraCard(QFrame):
    def __init__(self, camera_id: str, room_name: str) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.room_name = room_name
        self.setObjectName("cameraCard")
        self.setMinimumWidth(420)
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top = QFrame(self)
        top.setObjectName("cameraTop")
        top.setFixedHeight(46)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(12, 5, 10, 5)
        tl.setSpacing(8)
        names = QVBoxLayout()
        names.setSpacing(0)
        self.name_label = QLabel(CAMERA_TITLES.get(camera_id, camera_id))
        self.name_label.setObjectName("cameraName")
        self.room_label = QLabel(room_name)
        self.room_label.setObjectName("cameraRoom")
        names.addWidget(self.name_label)
        names.addWidget(self.room_label)
        tl.addLayout(names)
        tl.addStretch(1)
        self.dot = QLabel("●")
        self.dot.setObjectName("cameraDotOff")
        self.fps = QLabel("0.0 fps")
        self.fps.setObjectName("cameraFps")
        self.people = QLabel("0 person")
        self.people.setObjectName("cameraCount")
        tl.addWidget(self.people)
        tl.addWidget(self.dot)
        tl.addWidget(self.fps)
        layout.addWidget(top)

        self.surface = NativeVideoSurface(camera_id, self)
        layout.addWidget(self.surface, 1)

    def update_live(self, fps: float, people: int) -> None:
        online = fps > 2.0
        self.dot.setObjectName("cameraDotOn" if online else "cameraDotOff")
        self.dot.style().unpolish(self.dot)
        self.dot.style().polish(self.dot)
        self.fps.setText(f"{fps:.1f} fps" if online else "OFFLINE")
        self.people.setText(f"{people} person")

    def set_card_width(self, width: int) -> None:
        # Header 46 + real 16:9 video. Keep the cameras visually large like the
        # supplied web UI rather than shrinking six feeds into a tiny fixed wall.
        video_h = max(225, int(max(400, width) * 9 / 16))
        self.setFixedHeight(46 + video_h)


class NavButton(QPushButton):
    def __init__(self, text: str, active: bool = False) -> None:
        super().__init__(text)
        self.setObjectName("navActive" if active else "navButton")
        self.setMinimumHeight(38)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class SentinelWindow(QMainWindow):
    runtime_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sentinel VMS · Monitoring")
        self.setMinimumSize(1280, 760)
        self.runtime = CameraQtRuntime()
        self.runtime.set_heatmap_visible(False)
        self.controller = RuntimeController(self.runtime)
        self.runtime_failed.connect(self._show_runtime_error)
        self._pipeline_started = False
        self._last_status_t = time.monotonic()
        self._previous_frames = {cid: 0 for cid in self.runtime.stats}
        self._fps_samples = {cid: deque(maxlen=4) for cid in self.runtime.stats}
        self.camera_cards: dict[str, CameraCard] = {}
        self.room_badges: dict[str, QLabel] = {}
        self._build_ui()
        self._apply_style()

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._refresh_realtime)
        self.timer.start()
        QTimer.singleShot(180, self._start_runtime)

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
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)

        brand = QFrame(sidebar)
        brand.setObjectName("brandBox")
        brand.setFixedHeight(66)
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(16, 10, 12, 10)
        shield = QLabel("⬢")
        shield.setObjectName("shield")
        bl.addWidget(shield)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand_name = QLabel("SENTINEL VMS")
        brand_name.setObjectName("brandName")
        brand_sub = QLabel("vision analytics · 6 cam")
        brand_sub.setObjectName("brandSub")
        brand_text.addWidget(brand_name)
        brand_text.addWidget(brand_sub)
        bl.addLayout(brand_text)
        bl.addStretch(1)
        sl.addWidget(brand)

        nav_wrap = QWidget(sidebar)
        nl = QVBoxLayout(nav_wrap)
        nl.setContentsMargins(8, 8, 8, 8)
        nl.setSpacing(2)
        nl.addWidget(NavButton("▣   Monitoring", active=True))
        for text in (
            "♙   People", "⌁   Events", "▤   Rooms", "⚙   Kameralar",
            "+   Enrollment", "◇   Diagnostics", "▥   Reports",
        ):
            b = NavButton(text)
            b.setEnabled(False)
            b.setToolTip("Keyingi real-time modulda ulanadi")
            nl.addWidget(b)
        nl.addStretch(1)
        sl.addWidget(nav_wrap, 1)

        build = QLabel("build 2026.08 · edge worker")
        build.setObjectName("buildText")
        build.setFixedHeight(42)
        sl.addWidget(build)
        shell.addWidget(sidebar)

        right = QWidget(root)
        right.setObjectName("content")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        header = QFrame(right)
        header.setObjectName("header")
        header.setFixedHeight(66)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 8, 18, 8)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("Monitoring")
        title.setObjectName("pageTitle")
        sub = QLabel("Jonli oqim · xonalar bo‘yicha guruhlangan · real-time")
        sub.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        hl.addLayout(title_box)
        hl.addStretch(1)

        self.heat_button = QPushButton("♨  Heatmap")
        self.heat_button.setObjectName("heatToggle")
        self.heat_button.setCheckable(True)
        self.heat_button.setChecked(False)
        self.heat_button.setMinimumHeight(34)
        self.heat_button.toggled.connect(self._toggle_heatmap)
        hl.addWidget(self.heat_button)

        self.full_button = QPushButton("⛶  Fullscreen")
        self.full_button.setObjectName("headerButton")
        self.full_button.setMinimumHeight(34)
        self.full_button.clicked.connect(self._toggle_fullscreen)
        hl.addWidget(self.full_button)
        rl.addWidget(header)

        scroll = QScrollArea(right)
        scroll.setObjectName("mainScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        page.setObjectName("page")
        self.page = page
        pl = QVBoxLayout(page)
        pl.setContentsMargins(24, 22, 24, 28)
        pl.setSpacing(0)

        stats = QGridLayout()
        stats.setHorizontalSpacing(12)
        stats.setVerticalSpacing(12)
        self.stat_tracked = StatCard("Hozir kuzatilmoqda", "0", "NvDCF current tracks")
        self.stat_online = StatCard("Online kamera", "0 / 6", "real RTSP/NVDEC", "known")
        self.stat_detector = StatCard("Detector", "0.0 Hz", "YOLO26m / camera")
        self.stat_heat = StatCard("Movement heat", "0", "stationary deposit = 0", "unknown")
        for i, card in enumerate((self.stat_tracked, self.stat_online, self.stat_detector, self.stat_heat)):
            stats.addWidget(card, 0, i)
            stats.setColumnStretch(i, 1)
        pl.addLayout(stats)

        livebar = QHBoxLayout()
        livebar.setContentsMargins(0, 24, 0, 10)
        live_label = QLabel("LIVE GRID")
        live_label.setObjectName("liveGridLabel")
        livebar.addWidget(live_label)
        livebar.addStretch(1)
        self.pipeline_badge = QLabel("STARTING · 0/6")
        self.pipeline_badge.setObjectName("pipelineBadge")
        livebar.addWidget(self.pipeline_badge)
        pl.addLayout(livebar)

        for room_key, room_name, cids in ROOMS:
            heading = QHBoxLayout()
            heading.setContentsMargins(0, 8, 0, 8)
            label = QLabel(room_name)
            label.setObjectName("roomTitle")
            badge = QLabel("2 kamera · 0 tracks")
            badge.setObjectName("roomBadge")
            self.room_badges[room_key] = badge
            heading.addWidget(label)
            heading.addWidget(badge)
            heading.addStretch(1)
            pl.addLayout(heading)

            row = QHBoxLayout()
            row.setSpacing(12)
            for cid in cids:
                card = CameraCard(cid, room_name)
                self.camera_cards[cid] = card
                card.surface.resized.connect(self.runtime.update_render_rectangle)
                row.addWidget(card, 1)
            pl.addLayout(row)
            pl.addSpacing(16)

        pl.addStretch(1)
        scroll.setWidget(page)
        rl.addWidget(scroll, 1)
        shell.addWidget(right, 1)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#root, QWidget#content, QWidget#page, QScrollArea#mainScroll {
                background:#080b10; color:#d9e1e8;
            }
            QScrollArea#mainScroll > QWidget > QWidget { background:#080b10; }
            QScrollBar:vertical { width:9px; background:#080b10; margin:0; }
            QScrollBar::handle:vertical { background:#26313e; min-height:36px; border-radius:4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }

            QFrame#sidebar { background:#0b1017; border-right:1px solid #202833; }
            QFrame#brandBox { border-bottom:1px solid #202833; }
            QLabel#shield { color:#58d3a3; font-size:19px; }
            QLabel#brandName { color:#edf2f6; font-size:13px; font-weight:700; letter-spacing:1px; }
            QLabel#brandSub { color:#687585; font-family:monospace; font-size:9px; }
            QLabel#buildText { color:#556171; border-top:1px solid #202833; padding-left:12px; font-family:monospace; font-size:9px; }

            QPushButton#navButton, QPushButton#navActive {
                border:0; border-radius:5px; text-align:left; padding:0 12px;
                background:transparent; color:#8d9aa8; font-size:12px;
            }
            QPushButton#navButton:hover { background:#111922; color:#dce4eb; }
            QPushButton#navActive { background:#13201e; color:#65d6aa; font-weight:600; }
            QPushButton#navButton:disabled { color:#44505d; }

            QFrame#header { background:#090d12; border-bottom:1px solid #202833; }
            QLabel#pageTitle { color:#eef3f7; font-size:17px; font-weight:700; }
            QLabel#pageSubtitle { color:#687687; font-size:10px; }
            QPushButton#headerButton, QPushButton#heatToggle {
                background:#0d131a; color:#9eabb8; border:1px solid #283442;
                border-radius:5px; padding:6px 11px; font-size:10px; font-weight:600;
            }
            QPushButton#headerButton:hover, QPushButton#heatToggle:hover { color:#eef3f7; border-color:#445266; }
            QPushButton#heatToggle:checked { background:#19352c; border-color:#356b57; color:#71ddb3; }

            QFrame#statCard, QFrame#cameraCard { background:#0d1218; border:1px solid #242d38; border-radius:6px; }
            QLabel#statLabel { color:#738091; font-family:monospace; font-size:9px; letter-spacing:1px; }
            QLabel#statValue_default, QLabel#statValue_known, QLabel#statValue_unknown {
                color:#e9eff4; font-size:27px; font-weight:700;
            }
            QLabel#statValue_known { color:#66d7aa; }
            QLabel#statValue_unknown { color:#e8b75c; }
            QLabel#statHint { color:#596777; font-size:9px; }

            QLabel#liveGridLabel { color:#788596; font-family:monospace; font-size:10px; font-weight:700; letter-spacing:1px; }
            QLabel#pipelineBadge { color:#75d9b1; background:#102018; border:1px solid #244c3c; border-radius:4px; padding:5px 9px; font-family:monospace; font-size:9px; }
            QLabel#roomTitle { color:#e4eaf0; font-size:13px; font-weight:600; }
            QLabel#roomBadge { color:#687585; background:#111821; border-radius:4px; padding:3px 7px; font-family:monospace; font-size:9px; }

            QFrame#cameraTop { background:#0b1016; border-bottom:1px solid #232c37; }
            QLabel#cameraName { color:#e7edf2; font-size:11px; font-weight:600; }
            QLabel#cameraRoom { color:#687585; font-family:monospace; font-size:9px; }
            QLabel#cameraFps, QLabel#cameraCount { color:#748191; font-family:monospace; font-size:9px; }
            QLabel#cameraDotOn { color:#59d49f; font-size:11px; }
            QLabel#cameraDotOff { color:#e15a60; font-size:11px; }
            """
        )

    def _native_handles(self) -> dict[str, int]:
        handles: dict[str, int] = {}
        for cid, card in self.camera_cards.items():
            card.surface.show()
            handle = int(card.surface.winId())
            if handle <= 0:
                raise RuntimeError(f"{cid}: Qt could not create a native video child window")
            handles[cid] = handle
        return handles

    def _start_runtime(self) -> None:
        try:
            # Handles must exist BEFORE PLAYING. The runtime additionally installs
            # the synchronous prepare-window-handle handler required by GStreamer.
            self.runtime.bind_window_handles(self._native_handles())
            self.controller.start()
            self._pipeline_started = True
            QTimer.singleShot(200, self._sync_card_sizes)
        except Exception as exc:
            self.runtime_failed.emit(str(exc))

    def _sync_card_sizes(self) -> None:
        for card in self.camera_cards.values():
            card.set_card_width(max(420, card.width()))
            self.runtime.update_render_rectangle(card.camera_id, card.surface.width(), card.surface.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(40, self._sync_card_sizes)

    def _toggle_heatmap(self, checked: bool) -> None:
        self.runtime.set_heatmap_visible(bool(checked))
        self.heat_button.setText("♨  Heatmap ON" if checked else "♨  Heatmap")

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
        now = time.monotonic()
        elapsed = max(0.25, now - self._last_status_t)
        self._last_status_t = now
        online = 0
        room_counts: dict[str, int] = {key: 0 for key, _, _ in ROOMS}

        for cid, stat in self.runtime.stats.items():
            previous = self._previous_frames.get(cid, stat.frames)
            delta = max(0, stat.frames - previous)
            self._previous_frames[cid] = stat.frames
            instant = delta / elapsed
            self._fps_samples[cid].append(instant)
            fps = sum(self._fps_samples[cid]) / max(1, len(self._fps_samples[cid]))
            if fps > 2.0:
                online += 1
            people = self.runtime.camera_person_count(cid)
            card = self.camera_cards.get(cid)
            if card is not None:
                card.update_live(fps, people)
            for key, _name, cids in ROOMS:
                if cid in cids:
                    room_counts[key] += people
                    break

        with self.runtime.det_lock:
            tracked = int(self.runtime.tracked_now)
            ready = bool(self.runtime.det_ready)
            error = str(self.runtime.det_error or "")
            result_age = float(getattr(self.runtime, "detector_result_age_ms", 0.0))

        rates = [self._recent_rate(rows, now) for rows in self.runtime.detector_times.values()]
        avg_hz = sum(rates) / len(rates) if rates else 0.0
        heat_updates = self.runtime.heatmap_updates_total()

        self.stat_tracked.set_value(str(tracked), "real current-frame NvDCF tracks")
        self.stat_online.set_value(f"{online} / 6", "real RTSP/NVDEC")
        self.stat_detector.set_value(f"{avg_hz:.1f} Hz", f"YOLO result age {result_age:.0f} ms")
        self.stat_heat.set_value(str(heat_updates), "movement-only · hidden still accumulates")

        for key, _name, cids in ROOMS:
            badge = self.room_badges[key]
            badge.setText(f"{len(cids)} kamera · {room_counts[key]} tracks")

        if error:
            self.pipeline_badge.setText("DETECTOR ERROR")
            self.pipeline_badge.setStyleSheet("color:#f0b0b3;background:#281316;border:1px solid #663138;border-radius:4px;padding:5px 9px;font-family:monospace;font-size:9px;")
        elif online == 6 and ready:
            self.pipeline_badge.setText("LIVE · 6/6")
            self.pipeline_badge.setStyleSheet("")
        elif self._pipeline_started:
            self.pipeline_badge.setText(f"LIVE · {online}/6")
        else:
            self.pipeline_badge.setText("STARTING · 0/6")

    def _show_runtime_error(self, message: str) -> None:
        self._pipeline_started = False
        self.pipeline_badge.setText("PIPELINE ERROR")
        print(f"CAMERA_QT ERROR {message}", file=sys.stderr, flush=True)

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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
