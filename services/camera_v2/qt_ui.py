from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time

# Prefer X11/XWayland when a DISPLAY is available. nveglglessink embeds through
# GstVideoOverlay using the QWidget native window handle (XID on X11).
if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .detection import _yolo_worker
from .person_heatmap import CameraPersonHeatmap


class RuntimeController:
    """Start/stop CameraPersonHeatmap without taking over the Qt main thread.

    The normal CLI runtime blocks inside GLib.MainLoop and owns UNIX signal
    handlers. For Qt we start the exact same camera/detection/tracking pipeline,
    but run only GLib.MainLoop in a helper thread. YOLO queues/scheduler and all
    DeepStream cleanup are kept equivalent to the CLI path.
    """

    def __init__(self, runtime: CameraPersonHeatmap) -> None:
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

        self.loop_thread = threading.Thread(
            target=r.loop.run,
            name="camera-v2-glib-loop",
            daemon=True,
        )
        self.loop_thread.start()

        result = r.pipeline.set_state(r.Gst.State.PLAYING)
        if result == r.Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("Camera V2 Qt pipeline failed to enter PLAYING")

        self.started = True
        print(
            "CAMERA_QT started: native PySide6 shell + embedded nveglglessink; "
            "camera_hot_path=unchanged heatmap_accumulation=always_on",
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
        if self.loop_thread is not None:
            self.loop_thread.join(timeout=2.0)

        try:
            r.pipeline.set_state(r.Gst.State.NULL)
        except Exception:
            pass

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
        for pad in getattr(r, "_request_pads", []):
            try:
                r.mux.release_request_pad(pad)
            except Exception:
                pass

        self.started = False


class NativeVideoSurface(QWidget):
    """A real native child window dedicated to GstVideoOverlay rendering."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.setAutoFillBackground(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(960, 360)


class NavButton(QPushButton):
    def __init__(self, text: str, active: bool = False, enabled: bool = True) -> None:
        super().__init__(text)
        self.setObjectName("navActive" if active else "navButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.setEnabled(enabled)
        self.setMinimumHeight(38)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class SentinelWindow(QMainWindow):
    runtime_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sentinel VMS · Monitoring")
        self.setMinimumSize(1280, 760)

        self.runtime = CameraPersonHeatmap()
        # UI starts clean. Movement heat is still accumulated from frame 1.
        self.runtime.set_heatmap_visible(False)
        self.controller = RuntimeController(self.runtime)
        self.runtime_failed.connect(self._show_runtime_error)

        self._previous_frames = {cid: 0 for cid in self.runtime.stats}
        self._last_status_t = time.monotonic()
        self._pipeline_started = False

        self._build_ui()
        self._apply_style()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self._refresh_status)
        self.status_timer.start()

        QTimer.singleShot(100, self._start_runtime)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        # Sidebar adapted from the supplied Sentinel VMS design. The standalone
        # Heatmap route is intentionally removed: heatmap lives on Monitoring.
        self.sidebar = QFrame(root)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(190)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(12, 15, 12, 12)
        side.setSpacing(6)

        brand = QLabel("◆  SENTINEL VMS")
        brand.setObjectName("brand")
        side.addWidget(brand)
        brand_sub = QLabel("6 CAM · EDGE VISION")
        brand_sub.setObjectName("brandSub")
        side.addWidget(brand_sub)
        side.addSpacing(16)

        side.addWidget(NavButton("▣   Monitoring", active=True))
        for label in ("◎   People", "◇   Events", "▤   Reports", "⚙   Cameras"):
            button = NavButton(label, enabled=False)
            button.setToolTip("Keyingi bosqichda ulanadi")
            side.addWidget(button)
        side.addStretch(1)

        build = QLabel("GPU WALL · DEEPSTREAM 7.1\nbuild camera_v2")
        build.setObjectName("buildText")
        side.addWidget(build)
        shell.addWidget(self.sidebar)

        content = QWidget(root)
        content.setObjectName("content")
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.header = QFrame(content)
        self.header.setObjectName("header")
        self.header.setFixedHeight(74)
        h = QHBoxLayout(self.header)
        h.setContentsMargins(22, 10, 18, 10)
        h.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("Live Monitoring")
        title.setObjectName("pageTitle")
        subtitle = QLabel("6 camera · person detection · NvDCF tracking")
        subtitle.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        h.addLayout(title_box)
        h.addStretch(1)

        self.live_pill = QLabel("STARTING · 0/6")
        self.live_pill.setObjectName("livePill")
        self.live_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self.live_pill)

        self.heat_button = QPushButton("HEATMAP")
        self.heat_button.setObjectName("heatToggle")
        self.heat_button.setCheckable(True)
        self.heat_button.setChecked(False)
        self.heat_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.heat_button.setMinimumSize(112, 36)
        self.heat_button.toggled.connect(self._toggle_heatmap)
        h.addWidget(self.heat_button)

        self.full_button = QPushButton("FULLSCREEN")
        self.full_button.setObjectName("toolButton")
        self.full_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.full_button.setMinimumSize(104, 36)
        self.full_button.clicked.connect(self._toggle_fullscreen)
        h.addWidget(self.full_button)
        body.addWidget(self.header)

        stage_wrap = QWidget(content)
        stage_layout = QVBoxLayout(stage_wrap)
        stage_layout.setContentsMargins(14, 14, 14, 10)
        stage_layout.setSpacing(8)

        self.video_panel = QFrame(stage_wrap)
        self.video_panel.setObjectName("videoPanel")
        panel_layout = QVBoxLayout(self.video_panel)
        panel_layout.setContentsMargins(1, 1, 1, 1)
        panel_layout.setSpacing(0)

        panel_bar = QFrame(self.video_panel)
        panel_bar.setObjectName("videoBar")
        panel_bar.setFixedHeight(34)
        bar_layout = QHBoxLayout(panel_bar)
        bar_layout.setContentsMargins(11, 0, 10, 0)
        bar_layout.setSpacing(8)
        grid_label = QLabel("LIVE CAMERAS · 3×2")
        grid_label.setObjectName("gridLabel")
        bar_layout.addWidget(grid_label)
        bar_layout.addStretch(1)
        self.mode_label = QLabel("HEAT HIDDEN · ACCUMULATING")
        self.mode_label.setObjectName("modeLabel")
        bar_layout.addWidget(self.mode_label)
        panel_layout.addWidget(panel_bar)

        self.video_surface = NativeVideoSurface(self.video_panel)
        panel_layout.addWidget(self.video_surface, 1)
        stage_layout.addWidget(self.video_panel, 1)

        self.footer_status = QLabel("NVDEC · YOLO26m 640×384 · NvDCF 480×288 · movement heatmap")
        self.footer_status.setObjectName("footerStatus")
        stage_layout.addWidget(self.footer_status)
        body.addWidget(stage_wrap, 1)
        shell.addWidget(content, 1)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#content { background: #070a0f; color: #d8e0ea; }
            QFrame#sidebar { background: #0b1017; border-right: 1px solid #1d2733; }
            QLabel#brand { color: #f0f5fa; font-size: 14px; font-weight: 700; letter-spacing: 1px; }
            QLabel#brandSub { color: #657386; font-family: monospace; font-size: 10px; }
            QLabel#buildText { color: #4e5c6e; font-family: monospace; font-size: 9px; padding: 8px 4px; }

            QPushButton#navButton, QPushButton#navActive {
                text-align: left; border: 0; border-radius: 5px; padding: 0 12px;
                color: #8694a6; background: transparent; font-size: 12px;
            }
            QPushButton#navActive { background: #111c22; color: #71e0b6; font-weight: 600; }
            QPushButton#navButton:disabled { color: #465261; }

            QFrame#header { background: #090d13; border-bottom: 1px solid #1b2530; }
            QLabel#pageTitle { color: #edf3f8; font-size: 18px; font-weight: 700; }
            QLabel#pageSubtitle { color: #69788b; font-size: 11px; }
            QLabel#livePill {
                color: #7be0b9; background: #0e201b; border: 1px solid #1e4f40;
                border-radius: 4px; padding: 7px 10px; font-family: monospace; font-size: 10px;
            }

            QPushButton#heatToggle, QPushButton#toolButton {
                color: #9eabb9; background: #0d131b; border: 1px solid #263242;
                border-radius: 5px; padding: 7px 12px; font-size: 10px; font-weight: 700;
            }
            QPushButton#heatToggle:hover, QPushButton#toolButton:hover { border-color: #44556a; color: #e4ebf2; }
            QPushButton#heatToggle:checked {
                color: #07130e; background: #64d9aa; border-color: #64d9aa;
            }

            QFrame#videoPanel { background: #020406; border: 1px solid #202a36; }
            QFrame#videoBar { background: #0a0f15; border-bottom: 1px solid #1d2732; }
            QLabel#gridLabel { color: #b8c4d1; font-family: monospace; font-size: 10px; font-weight: 700; }
            QLabel#modeLabel { color: #617084; font-family: monospace; font-size: 9px; }
            QLabel#footerStatus { color: #566476; font-family: monospace; font-size: 9px; padding-left: 2px; }
            """
        )

    def _bind_video_overlay(self) -> None:
        import gi

        gi.require_version("GstVideo", "1.0")
        from gi.repository import GstVideo

        xid = int(self.video_surface.winId())
        if xid <= 0:
            raise RuntimeError("Qt video surface has no native window handle")
        GstVideo.VideoOverlay.set_window_handle(self.runtime.sink, xid)
        try:
            GstVideo.VideoOverlay.handle_events(self.runtime.sink, False)
        except Exception:
            pass
        print(
            f"CAMERA_QT video_overlay_bound handle={xid} qt_platform={QApplication.platformName()}",
            flush=True,
        )

    def _start_runtime(self) -> None:
        try:
            self._bind_video_overlay()
            self.controller.start()
            self._pipeline_started = True
        except Exception as exc:
            self.runtime_failed.emit(str(exc))

    def _toggle_heatmap(self, checked: bool) -> None:
        self.runtime.set_heatmap_visible(bool(checked))
        if checked:
            self.mode_label.setText("HEAT VISIBLE · MOVEMENT ONLY")
        else:
            self.mode_label.setText("HEAT HIDDEN · ACCUMULATING")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
            self.full_button.setText("FULLSCREEN")
        else:
            self.showFullScreen()
            self.full_button.setText("EXIT FULL")

    def _refresh_status(self) -> None:
        now = time.monotonic()
        elapsed = max(0.25, now - self._last_status_t)
        self._last_status_t = now

        active = 0
        total_fps = 0.0
        for cid, stat in self.runtime.stats.items():
            previous = self._previous_frames.get(cid, stat.frames)
            delta = max(0, stat.frames - previous)
            self._previous_frames[cid] = stat.frames
            fps = delta / elapsed
            if fps > 2.0:
                active += 1
                total_fps += fps

        if not self._pipeline_started:
            self.live_pill.setText("STARTING · 0/6")
            return

        avg = total_fps / active if active else 0.0
        self.live_pill.setText(f"LIVE · {active}/6 · {avg:.1f} FPS")
        if active == 6:
            self.live_pill.setStyleSheet(
                "color:#7be0b9;background:#0e201b;border:1px solid #1e4f40;"
                "border-radius:4px;padding:7px 10px;font-family:monospace;font-size:10px;"
            )
        else:
            self.live_pill.setStyleSheet(
                "color:#f3c969;background:#211b0c;border:1px solid #5a4517;"
                "border-radius:4px;padding:7px 10px;font-family:monospace;font-size:10px;"
            )

    def _show_runtime_error(self, message: str) -> None:
        self._pipeline_started = False
        self.live_pill.setText("PIPELINE ERROR")
        self.footer_status.setText(f"ERROR · {message}")
        print(f"CAMERA_QT ERROR {message}", file=sys.stderr, flush=True)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_F11:
            self._toggle_fullscreen()
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showMaximized()
            self.full_button.setText("FULLSCREEN")
            return
        if event.key() == Qt.Key.Key_H:
            self.heat_button.toggle()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.status_timer.stop()
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
