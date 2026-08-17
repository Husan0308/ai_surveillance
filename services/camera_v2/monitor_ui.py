from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass

# This UI is intentionally a thin shell around the existing working pipeline.
# It does not decode video, copy frames, run inference, or compose cameras in Qt.
# DeepStream still owns the complete hot path and renders one tiled wall directly
# into the native Qt X11 window through GstVideoOverlay.

TILE_WIDTH = 640
TILE_HEIGHT = 360
GRID_COLUMNS = 2
GRID_ROWS = 3
WALL_WIDTH = TILE_WIDTH * GRID_COLUMNS   # 1280
WALL_HEIGHT = TILE_HEIGHT * GRID_ROWS    # 1080
CAMERA_COUNT = 6


@dataclass(frozen=True)
class UiStatus:
    state: str
    detail: str = ""


def _put_status(status_q, state: str, detail: str = "") -> None:
    try:
        status_q.put_nowait((state, detail))
    except Exception:
        pass


def _pipeline_process(window_id: int, command_q, status_q) -> None:
    """Run the proven Camera V2 tracking pipeline in its own process.

    Keeping GStreamer/DeepStream outside the Qt process avoids toolkit/GLib event
    loop interference. The only integration point is the X11 window id supplied to
    GstVideoOverlay. Camera ingest, NVDEC, YOLO26m, NvDCF and OSD remain unchanged.
    """
    runtime = None
    try:
        if window_id <= 0:
            raise RuntimeError("invalid Qt native window id")

        # The source cameras and detector/tracker remain unchanged. Only the tiled
        # output geometry changes. 1280x1080 has the SAME pixel count as the old
        # 1920x720 wall, while preserving each camera at exactly 640x360.
        os.environ["CAMERA_V2_WALL_WIDTH"] = str(WALL_WIDTH)
        os.environ["CAMERA_V2_WALL_HEIGHT"] = str(WALL_HEIGHT)

        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        from .person_tracking_final import CameraPersonTrackingFinal

        runtime = CameraPersonTrackingFinal()
        if len(runtime.cameras) != CAMERA_COUNT:
            raise RuntimeError(f"monitor UI requires 6 cameras, found {len(runtime.cameras)}")

        # Reconfigure ONLY nvmultistreamtiler before PLAYING. The old wall is 3x2
        # at 1920x720 -> 640x360 per camera. This makes it 2x3 at 1280x1080, so the
        # apparent size of every camera is exactly preserved.
        runtime.wall_width = WALL_WIDTH
        runtime.wall_height = WALL_HEIGHT
        runtime.tiler.set_property("rows", GRID_ROWS)
        runtime.tiler.set_property("columns", GRID_COLUMNS)
        runtime.tiler.set_property("width", WALL_WIDTH)
        runtime.tiler.set_property("height", WALL_HEIGHT)
        if runtime.tiler.find_property("show-source") is not None:
            runtime.tiler.set_property("show-source", -1)

        xid = int(window_id)

        def bind_overlay(overlay) -> None:
            GstVideo.VideoOverlay.set_window_handle(overlay, xid)
            try:
                GstVideo.VideoOverlay.handle_events(overlay, False)
            except Exception:
                pass

        # Pre-bind the known sink, then also handle prepare-window-handle in the
        # synchronous bus callback. No Qt call is made from the streaming thread.
        bind_overlay(runtime.sink)

        def on_sync_message(_bus, message, _data=None):
            try:
                prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
            except Exception:
                structure = message.get_structure()
                prepare = bool(structure and structure.get_name() == "prepare-window-handle")
            if not prepare:
                return Gst.BusSyncReply.PASS
            try:
                bind_overlay(message.src)
                _put_status(status_q, "VIDEO_BOUND", f"xid={xid}")
                return Gst.BusSyncReply.DROP
            except Exception as exc:
                _put_status(status_q, "ERROR", f"video overlay: {exc}")
                return Gst.BusSyncReply.PASS

        runtime.bus.set_sync_handler(on_sync_message, None)

        def observe_bus(_bus, message):
            if message.type == Gst.MessageType.STATE_CHANGED and message.src == runtime.pipeline:
                try:
                    _old, new, _pending = message.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        _put_status(status_q, "LIVE", "6-camera DeepStream pipeline PLAYING")
                except Exception:
                    pass
            elif message.type == Gst.MessageType.ERROR:
                try:
                    err, _debug = message.parse_error()
                    src = message.src.get_name() if message.src else "unknown"
                    _put_status(status_q, "PIPELINE_WARNING", f"{src}: {err.message}")
                except Exception:
                    pass

        runtime.bus.connect("message", observe_bus)

        def poll_commands() -> bool:
            stop_requested = False
            latest_focus = None
            got_focus = False
            while True:
                try:
                    command, value = command_q.get_nowait()
                except queue.Empty:
                    break
                if command == "stop":
                    stop_requested = True
                elif command == "focus":
                    latest_focus = int(value)
                    got_focus = True

            if got_focus and runtime.tiler.find_property("show-source") is not None:
                source_id = latest_focus if 0 <= latest_focus < CAMERA_COUNT else -1
                runtime.tiler.set_property("show-source", source_id)
                _put_status(status_q, "FOCUS", str(source_id))

            if stop_requested:
                runtime.stop()
                return False
            return True

        runtime.GLib.timeout_add(50, poll_commands)
        _put_status(
            status_q,
            "STARTING",
            f"2 columns x 3 rows; tile={TILE_WIDTH}x{TILE_HEIGHT}; wall={WALL_WIDTH}x{WALL_HEIGHT}",
        )

        # Important: use the existing run() unchanged. It owns YOLO worker startup,
        # scheduler, PLAYING/NULL transitions and cleanup exactly like the CLI path.
        rc = runtime.run()
        _put_status(status_q, "STOPPED", f"exit={rc}")
    except BaseException as exc:
        _put_status(status_q, "ERROR", f"{type(exc).__name__}: {exc}")
        try:
            if runtime is not None:
                runtime.stop()
                runtime.pipeline.set_state(runtime.Gst.State.NULL)
        except Exception:
            pass


class PipelineController:
    def __init__(self) -> None:
        self.ctx = mp.get_context("spawn")
        self.command_q = self.ctx.Queue(maxsize=32)
        self.status_q = self.ctx.Queue(maxsize=64)
        self.process: mp.Process | None = None
        self.last_status = UiStatus("WAITING")

    def start(self, window_id: int) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.process = self.ctx.Process(
            target=_pipeline_process,
            args=(int(window_id), self.command_q, self.status_q),
            name="camera-v2-deepstream-ui",
            daemon=False,
        )
        self.process.start()
        self.last_status = UiStatus("STARTING")

    def focus(self, source_id: int | None) -> None:
        value = -1 if source_id is None else int(source_id)
        try:
            self.command_q.put_nowait(("focus", value))
        except queue.Full:
            pass

    def poll_status(self) -> UiStatus:
        latest = None
        while True:
            try:
                latest = self.status_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.last_status = UiStatus(str(latest[0]), str(latest[1]))
        if self.process is not None and not self.process.is_alive():
            if self.last_status.state not in {"STOPPED", "ERROR"}:
                self.last_status = UiStatus("ERROR", f"pipeline process exited: {self.process.exitcode}")
        return self.last_status

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive():
            try:
                self.command_q.put(("stop", 0), timeout=0.5)
            except Exception:
                pass
            process.join(timeout=6.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        self.process = None


def main() -> int:
    # Prefer X11 on the current AnyDesk/Kubuntu deployment because GstVideoOverlay
    # receives an XID. Do not override callers who explicitly chose another QPA.
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    try:
        from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
        from PySide6.QtGui import QCloseEvent, QKeyEvent, QMouseEvent
        from PySide6.QtWidgets import (
            QApplication,
            QFrame,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print("PySide6 is required for monitor UI: pip install PySide6", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    class VideoWall(QWidget):
        nativeReady = Signal(int)
        focusChanged = Signal(object)

        def __init__(self, controller: PipelineController, parent=None):
            super().__init__(parent)
            self.controller = controller
            self.focused_source: int | None = None
            self.hovered_source: int | None = None
            self._native_emitted = False

            # Fixed canvas preserves 640x360 per camera exactly. QScrollArea handles
            # displays that are shorter than 1080 instead of shrinking the cameras.
            self.setFixedSize(WALL_WIDTH, WALL_HEIGHT)
            self.setMouseTracking(True)
            self.setAutoFillBackground(False)
            self.setAttribute(Qt.WA_NativeWindow, True)
            self.setAttribute(Qt.WA_PaintOnScreen, True)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setFocusPolicy(Qt.StrongFocus)
            _ = int(self.winId())

            self.buttons: list[QPushButton] = []
            for source_id in range(CAMERA_COUNT):
                button = QPushButton("⛶", self)
                button.setFixedSize(36, 30)
                button.setCursor(Qt.PointingHandCursor)
                button.setToolTip(f"CAM-{source_id + 1:02d} fullscreen")
                button.setStyleSheet(
                    "QPushButton{background:rgba(7,12,20,210);color:#f2f6fb;"
                    "border:1px solid rgba(255,255,255,55);border-radius:5px;"
                    "font-size:17px;font-weight:700;}"
                    "QPushButton:hover{background:rgba(21,35,52,235);border-color:#55d9ff;}"
                )
                button.clicked.connect(lambda _checked=False, sid=source_id: self.toggle_focus(sid))
                button.hide()
                self.buttons.append(button)
            self._layout_buttons()

        def paintEngine(self):
            # EGL owns this native X11 surface. Prevent Qt backing-store paint from
            # covering the video. Child buttons are separate Qt child windows.
            return None

        def paintEvent(self, event):
            event.accept()

        def showEvent(self, event):
            super().showEvent(event)
            if not self._native_emitted:
                self._native_emitted = True
                xid = int(self.winId())
                QTimer.singleShot(100, lambda: self.nativeReady.emit(xid))

        @staticmethod
        def source_at(pos: QPoint) -> int | None:
            x, y = pos.x(), pos.y()
            if x < 0 or y < 0 or x >= WALL_WIDTH or y >= WALL_HEIGHT:
                return None
            col = min(GRID_COLUMNS - 1, x // TILE_WIDTH)
            row = min(GRID_ROWS - 1, y // TILE_HEIGHT)
            source_id = int(row * GRID_COLUMNS + col)
            return source_id if 0 <= source_id < CAMERA_COUNT else None

        def _button_rect(self, source_id: int) -> QRect:
            if self.focused_source is not None:
                return QRect(WALL_WIDTH - 48, 12, 36, 30)
            row, col = divmod(source_id, GRID_COLUMNS)
            left = col * TILE_WIDTH
            top = row * TILE_HEIGHT
            return QRect(left + TILE_WIDTH - 48, top + 12, 36, 30)

        def _layout_buttons(self) -> None:
            for source_id, button in enumerate(self.buttons):
                button.setGeometry(self._button_rect(source_id))
                button.raise_()

        def _show_hover_button(self, source_id: int | None) -> None:
            self.hovered_source = source_id
            for sid, button in enumerate(self.buttons):
                visible = source_id is not None and sid == source_id
                if self.focused_source is not None:
                    visible = sid == self.focused_source and source_id == self.focused_source
                button.setVisible(visible)
                if visible:
                    button.raise_()

        def mouseMoveEvent(self, event: QMouseEvent) -> None:
            source_id = self.focused_source if self.focused_source is not None else self.source_at(event.position().toPoint())
            self._show_hover_button(source_id)
            super().mouseMoveEvent(event)

        def enterEvent(self, event) -> None:
            self.setFocus(Qt.MouseFocusReason)
            super().enterEvent(event)

        def leaveEvent(self, event) -> None:
            # Keep the button visible while the cursor is over that child button.
            global_pos = self.mapFromGlobal(self.cursor().pos())
            if not self.rect().contains(global_pos):
                self._show_hover_button(None)
            super().leaveEvent(event)

        def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
            source_id = self.focused_source if self.focused_source is not None else self.source_at(event.position().toPoint())
            if source_id is not None:
                self.toggle_focus(source_id)
            super().mouseDoubleClickEvent(event)

        def toggle_focus(self, source_id: int) -> None:
            if self.focused_source == source_id:
                self.set_focus(None)
            else:
                self.set_focus(source_id)

        def set_focus(self, source_id: int | None) -> None:
            if source_id is not None and not 0 <= source_id < CAMERA_COUNT:
                source_id = None
            self.focused_source = source_id
            self.controller.focus(source_id)
            self._layout_buttons()
            if source_id is None:
                self._show_hover_button(None)
            else:
                self._show_hover_button(source_id)
            self.focusChanged.emit(source_id)

    class MonitorWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.controller = PipelineController()
            self.setWindowTitle("AI Surveillance · Monitoring")
            self.setMinimumSize(980, 680)

            central = QWidget(self)
            central.setStyleSheet("background:#080d14;")
            root = QVBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            header = QFrame()
            header.setFixedHeight(58)
            header.setStyleSheet("background:#0d141e;border-bottom:1px solid #1d2a38;")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(18, 0, 18, 0)

            title = QLabel("Monitoring")
            title.setStyleSheet("color:#f2f6fb;font-size:18px;font-weight:700;")
            header_layout.addWidget(title)
            header_layout.addSpacing(12)

            subtitle = QLabel("6 cameras · 2 per row · 640×360 each")
            subtitle.setStyleSheet("color:#7890a8;font-size:12px;")
            header_layout.addWidget(subtitle)
            header_layout.addStretch(1)

            self.grid_button = QPushButton("Grid")
            self.grid_button.setCursor(Qt.PointingHandCursor)
            self.grid_button.setStyleSheet(
                "QPushButton{background:#121d29;color:#dce8f3;border:1px solid #26394b;"
                "border-radius:6px;padding:7px 13px;font-weight:600;}"
                "QPushButton:hover{border-color:#55d9ff;}"
            )
            self.grid_button.clicked.connect(lambda: self.wall.set_focus(None))
            self.grid_button.hide()
            header_layout.addWidget(self.grid_button)
            header_layout.addSpacing(12)

            self.status = QLabel("WAITING")
            self.status.setStyleSheet("color:#8ea3b7;font-size:11px;font-weight:700;")
            self.status.setToolTip("")
            header_layout.addWidget(self.status)
            root.addWidget(header)

            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(False)
            self.scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.scroll.setFrameShape(QFrame.NoFrame)
            self.scroll.setStyleSheet(
                "QScrollArea{background:#05080d;border:0;}"
                "QScrollBar:vertical{background:#0b1119;width:11px;margin:0;}"
                "QScrollBar::handle:vertical{background:#2b3b4d;min-height:40px;border-radius:5px;}"
                "QScrollBar:horizontal{background:#0b1119;height:11px;margin:0;}"
                "QScrollBar::handle:horizontal{background:#2b3b4d;min-width:40px;border-radius:5px;}"
            )

            self.wall = VideoWall(self.controller)
            self.wall.nativeReady.connect(self._start_pipeline)
            self.wall.focusChanged.connect(self._focus_changed)
            self.scroll.setWidget(self.wall)
            root.addWidget(self.scroll, 1)
            self.setCentralWidget(central)

            self.poll_timer = QTimer(self)
            self.poll_timer.timeout.connect(self._poll_status)
            self.poll_timer.start(200)

        def _start_pipeline(self, xid: int) -> None:
            self.status.setText("STARTING")
            self.controller.start(int(xid))

        def _focus_changed(self, source_id) -> None:
            self.grid_button.setVisible(source_id is not None)
            if source_id is None:
                self.setWindowTitle("AI Surveillance · Monitoring")
            else:
                self.setWindowTitle(f"AI Surveillance · CAM-{int(source_id) + 1:02d}")
                # When a single source is zoomed the 1280x1080 native wall remains
                # centered; no second decoder or stream is created.
                self.scroll.ensureVisible(0, 0)

        def _poll_status(self) -> None:
            status = self.controller.poll_status()
            self.status.setText(status.state)
            self.status.setToolTip(status.detail)
            if status.state == "LIVE":
                self.status.setStyleSheet("color:#42d89c;font-size:11px;font-weight:700;")
            elif status.state in {"ERROR", "PIPELINE_WARNING"}:
                self.status.setStyleSheet("color:#ff6577;font-size:11px;font-weight:700;")
            elif status.state in {"STARTING", "VIDEO_BOUND", "FOCUS"}:
                self.status.setStyleSheet("color:#55d9ff;font-size:11px;font-weight:700;")
            else:
                self.status.setStyleSheet("color:#8ea3b7;font-size:11px;font-weight:700;")

        def keyPressEvent(self, event: QKeyEvent) -> None:
            if event.key() == Qt.Key_Escape and self.wall.focused_source is not None:
                self.wall.set_focus(None)
                event.accept()
                return
            if event.key() == Qt.Key_F11:
                if self.isFullScreen():
                    self.showMaximized()
                else:
                    self.showFullScreen()
                event.accept()
                return
            super().keyPressEvent(event)

        def closeEvent(self, event: QCloseEvent) -> None:
            self.poll_timer.stop()
            self.controller.stop()
            event.accept()

    app = QApplication(sys.argv)
    app.setApplicationName("AI Surveillance Monitoring")
    app.setOrganizationName("Apsidal")
    app.setStyle("Fusion")

    window = MonitorWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
