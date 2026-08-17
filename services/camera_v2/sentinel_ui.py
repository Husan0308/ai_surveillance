from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .sentinel_config import ROOMS, delete_camera, list_cameras, next_camera_id, room_cameras, save_camera
from .sentinel_store import SentinelStore

TILE_WIDTH = 640
TILE_HEIGHT = 360
GRID_COLUMNS = 2


@dataclass(frozen=True)
class UiStatus:
    state: str
    detail: str = ""


def _active_camera_rows() -> list[dict]:
    return [row for row in list_cameras(include_disabled=False) if row.get("enabled", True)]


def _put_status(status_q, state: str, detail) -> None:
    try:
        status_q.put_nowait((state, detail))
    except Exception:
        pass


def _pipeline_process(window_id: int, command_q, status_q) -> None:
    runtime = None
    try:
        rows = _active_camera_rows()
        count = len(rows)
        if count < 1:
            raise RuntimeError("No enabled cameras")
        grid_rows = max(1, math.ceil(count / GRID_COLUMNS))
        wall_width = TILE_WIDTH * GRID_COLUMNS
        wall_height = TILE_HEIGHT * grid_rows

        os.environ["CAMERA_V2_TILER_COLUMNS"] = str(GRID_COLUMNS)
        os.environ["CAMERA_V2_WALL_WIDTH"] = str(wall_width)
        os.environ["CAMERA_V2_WALL_HEIGHT"] = str(wall_height)

        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        from .person_tracking_heatmap import CameraPersonTrackingHeatmap

        runtime = CameraPersonTrackingHeatmap()
        runtime.set_heatmap_render_enabled(False)
        xid = int(window_id)
        if xid <= 0:
            raise RuntimeError("invalid Qt native window id")

        def bind_overlay(overlay) -> None:
            GstVideo.VideoOverlay.set_window_handle(overlay, xid)
            try:
                GstVideo.VideoOverlay.handle_events(overlay, False)
            except Exception:
                pass

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
                        _put_status(
                            status_q,
                            "LIVE",
                            f"{len(runtime.cameras)} camera DeepStream pipeline PLAYING",
                        )
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
            heatmap_value = None
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
                elif command == "heatmap":
                    heatmap_value = bool(value)

            if got_focus and runtime.tiler.find_property("show-source") is not None:
                source_id = latest_focus if 0 <= latest_focus < len(runtime.cameras) else -1
                runtime.tiler.set_property("show-source", source_id)
                _put_status(status_q, "FOCUS", str(source_id))
            if heatmap_value is not None:
                runtime.set_heatmap_render_enabled(heatmap_value)
                _put_status(status_q, "HEATMAP", int(heatmap_value))
            if stop_requested:
                runtime.stop()
                return False
            return True

        last_frames = {cid: 0 for cid in runtime.stats}
        last_seen = {cid: 0.0 for cid in runtime.stats}
        last_metric_t = time.monotonic()

        def publish_metrics() -> bool:
            nonlocal last_metric_t
            now = time.monotonic()
            elapsed = max(0.2, now - last_metric_t)
            last_metric_t = now
            camera_rows = []
            for index, camera in enumerate(runtime.cameras):
                stat = runtime.stats[camera.camera_id]
                previous = last_frames.get(camera.camera_id, 0)
                delta = max(0, int(stat.frames) - int(previous))
                last_frames[camera.camera_id] = int(stat.frames)
                if delta > 0:
                    last_seen[camera.camera_id] = now
                online = now - last_seen.get(camera.camera_id, 0.0) <= 2.5
                camera_rows.append(
                    {
                        "id": camera.camera_id,
                        "source_id": index,
                        "fps": delta / elapsed,
                        "online": online,
                        "room": getattr(camera, "room", ""),
                        "name": getattr(camera, "name", camera.camera_id),
                    }
                )
            _put_status(
                status_q,
                "METRICS",
                {
                    "total_people": int(getattr(runtime, "tracked_now", 0)),
                    "camera_count": len(camera_rows),
                    "online_cameras": sum(1 for row in camera_rows if row["online"]),
                    "cameras": camera_rows,
                    "heatmap_visible": bool(runtime.heatmap_render_enabled),
                },
            )
            return True

        runtime.GLib.timeout_add(50, poll_commands)
        runtime.GLib.timeout_add(500, publish_metrics)
        _put_status(
            status_q,
            "STARTING",
            f"{GRID_COLUMNS} columns x {grid_rows} rows; tile={TILE_WIDTH}x{TILE_HEIGHT}",
        )
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
        self.command_q = self.ctx.Queue(maxsize=64)
        self.status_q = self.ctx.Queue(maxsize=128)
        self.process: mp.Process | None = None
        self.last_status = UiStatus("WAITING")
        self.metrics: dict = {
            "total_people": 0,
            "camera_count": 0,
            "online_cameras": 0,
            "cameras": [],
            "heatmap_visible": False,
        }

    def _reset_queues(self) -> None:
        self.command_q = self.ctx.Queue(maxsize=64)
        self.status_q = self.ctx.Queue(maxsize=128)

    def start(self, window_id: int) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.process = self.ctx.Process(
            target=_pipeline_process,
            args=(int(window_id), self.command_q, self.status_q),
            name="sentinel-camera-v2",
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

    def heatmap(self, visible: bool) -> None:
        try:
            self.command_q.put_nowait(("heatmap", bool(visible)))
        except queue.Full:
            pass

    def poll(self) -> tuple[UiStatus, dict]:
        while True:
            try:
                state, detail = self.status_q.get_nowait()
            except queue.Empty:
                break
            if str(state) == "METRICS" and isinstance(detail, dict):
                self.metrics = detail
            else:
                self.last_status = UiStatus(str(state), str(detail))
        if self.process is not None and not self.process.is_alive():
            if self.last_status.state not in {"STOPPED", "ERROR"}:
                self.last_status = UiStatus(
                    "ERROR", f"pipeline process exited: {self.process.exitcode}"
                )
        return self.last_status, dict(self.metrics)

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive():
            try:
                self.command_q.put(("stop", 0), timeout=0.5)
            except Exception:
                pass
            process.join(timeout=7.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        self.process = None
        self._reset_queues()
        self.last_status = UiStatus("WAITING")


STYLE = """
QWidget { background:#070c12; color:#dce7f2; font-family:'DejaVu Sans'; font-size:13px; }
QMainWindow { background:#070c12; }
QFrame#Sidebar { background:#0a1119; border-right:1px solid #1d2b38; }
QFrame#Header { background:#080e15; border-bottom:1px solid #1d2b38; }
QLabel#Brand { font-size:18px; font-weight:800; letter-spacing:2px; color:#f3f7fb; }
QLabel#PageTitle { font-size:24px; font-weight:800; color:#f3f7fb; }
QLabel#Subtitle { color:#7890a8; }
QLabel#CardTitle { color:#93a9bd; font-size:11px; font-weight:700; letter-spacing:1px; }
QLabel#BigNumber { font-size:32px; font-weight:900; color:#42d7c5; }
QFrame#Card { background:#0d151e; border:1px solid #203142; border-radius:8px; }
QPushButton { background:#0d1721; border:1px solid #24384a; border-radius:6px; padding:8px 12px; color:#dce8f3; }
QPushButton:hover { border-color:#36d7cf; background:#11212c; }
QPushButton:checked { background:#143039; border-color:#36d7cf; color:#43e2d7; }
QPushButton#Nav { text-align:left; border:0; padding:12px 16px; color:#91a9bd; background:transparent; }
QPushButton#Nav:hover { background:#10202a; color:#dce8f3; }
QPushButton#Nav:checked { background:#142b35; color:#42ddd2; }
QPushButton#Primary { background:#3ed6c9; color:#051014; border:0; font-weight:800; }
QPushButton#Danger { color:#ff7887; border-color:#6b3039; }
QLineEdit,QTextEdit,QComboBox { background:#0b131c; border:1px solid #26394a; border-radius:5px; padding:8px; color:#e2ebf3; }
QLineEdit:focus,QTextEdit:focus,QComboBox:focus { border-color:#3dd7cf; }
QScrollArea { border:0; background:#070c12; }
QScrollBar:vertical { background:#091019; width:10px; }
QScrollBar::handle:vertical { background:#2a4052; border-radius:5px; min-height:36px; }
QToolTip { background:#111d28; color:#e7eff5; border:1px solid #30485b; }
"""


def main() -> int:
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    try:
        from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
        from PySide6.QtGui import QCloseEvent, QIcon, QKeyEvent, QMouseEvent, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QStackedWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print("PySide6 is required for Sentinel UI: pip install PySide6", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    def page_header(title_text: str, subtitle_text: str) -> tuple[QWidget, QVBoxLayout]:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(26, 18, 26, 18)
        layout.setSpacing(4)
        title = QLabel(title_text)
        title.setObjectName("PageTitle")
        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return widget, layout

    class StatCard(QFrame):
        def __init__(self, title: str, value: str = "0", note: str = ""):
            super().__init__()
            self.setObjectName("Card")
            self.setFixedHeight(104)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(16, 12, 16, 12)
            t = QLabel(title.upper())
            t.setObjectName("CardTitle")
            self.value = QLabel(value)
            self.value.setObjectName("BigNumber")
            self.note = QLabel(note)
            self.note.setObjectName("Subtitle")
            lay.addWidget(t)
            lay.addWidget(self.value)
            lay.addWidget(self.note)

    class VideoWall(QWidget):
        nativeReady = Signal(int)
        focusChanged = Signal(object)

        def __init__(self, controller: PipelineController, camera_rows: list[dict], parent=None):
            super().__init__(parent)
            self.controller = controller
            self.camera_rows = list(camera_rows)
            self.camera_count = len(camera_rows)
            self.grid_rows = max(1, math.ceil(max(1, self.camera_count) / GRID_COLUMNS))
            self.wall_width = TILE_WIDTH * GRID_COLUMNS
            self.wall_height = TILE_HEIGHT * self.grid_rows
            self.focused_source: int | None = None
            self.hovered_source: int | None = None
            self._native_emitted = False
            self.heatmap_source: int | None = None

            self.setFixedSize(self.wall_width, self.wall_height)
            self.setMouseTracking(True)
            self.setAttribute(Qt.WA_NativeWindow, True)
            self.setAttribute(Qt.WA_PaintOnScreen, True)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setFocusPolicy(Qt.StrongFocus)
            _ = int(self.winId())

            self.title_labels: list[QLabel] = []
            self.heat_buttons: list[QPushButton] = []
            self.full_buttons: list[QPushButton] = []
            for source_id, row in enumerate(self.camera_rows):
                label = QLabel(str(row.get("id", f"CAM-{source_id + 1:02d}")), self)
                label.setStyleSheet(
                    "background:rgba(5,12,18,205);color:#f0f5f8;border:1px solid rgba(90,120,140,80);"
                    "border-radius:4px;padding:4px 8px;font-weight:800;"
                )
                label.adjustSize()
                self.title_labels.append(label)

                heat = QPushButton("Heatmap", self)
                heat.setFixedSize(82, 30)
                heat.setCursor(Qt.PointingHandCursor)
                heat.setStyleSheet(
                    "QPushButton{background:rgba(7,15,23,220);color:#46ddd3;border:1px solid rgba(70,221,211,100);"
                    "border-radius:5px;font-size:11px;font-weight:700;}"
                    "QPushButton:hover{background:rgba(16,42,49,235);border-color:#46ddd3;}"
                )
                heat.clicked.connect(lambda _checked=False, sid=source_id: self.open_heatmap(sid))
                heat.hide()
                self.heat_buttons.append(heat)

                full = QPushButton("⛶", self)
                full.setFixedSize(36, 30)
                full.setCursor(Qt.PointingHandCursor)
                full.setToolTip("Fullscreen camera")
                full.setStyleSheet(
                    "QPushButton{background:rgba(7,15,23,220);color:#f3f6f8;border:1px solid rgba(255,255,255,55);"
                    "border-radius:5px;font-size:17px;font-weight:700;}"
                    "QPushButton:hover{background:rgba(20,38,52,235);border-color:#55d9ff;}"
                )
                full.clicked.connect(lambda _checked=False, sid=source_id: self.toggle_focus(sid))
                full.hide()
                self.full_buttons.append(full)
            self._layout_overlays()

        def paintEngine(self):
            return None

        def paintEvent(self, event):
            event.accept()

        def showEvent(self, event):
            super().showEvent(event)
            if not self._native_emitted:
                self._native_emitted = True
                xid = int(self.winId())
                QTimer.singleShot(120, lambda: self.nativeReady.emit(xid))

        def source_at(self, pos: QPoint) -> int | None:
            x, y = pos.x(), pos.y()
            if x < 0 or y < 0 or x >= self.wall_width or y >= self.wall_height:
                return None
            col = min(GRID_COLUMNS - 1, x // TILE_WIDTH)
            row = min(self.grid_rows - 1, y // TILE_HEIGHT)
            source_id = int(row * GRID_COLUMNS + col)
            return source_id if 0 <= source_id < self.camera_count else None

        def _tile_origin(self, source_id: int) -> tuple[int, int]:
            row, col = divmod(source_id, GRID_COLUMNS)
            return col * TILE_WIDTH, row * TILE_HEIGHT

        def _layout_overlays(self) -> None:
            for sid in range(self.camera_count):
                if self.focused_source is not None:
                    left, top = 0, 0
                    self.title_labels[sid].setVisible(sid == self.focused_source)
                    if sid == self.focused_source:
                        self.title_labels[sid].move(12, 12)
                        self.heat_buttons[sid].setGeometry(self.wall_width - 142, 12, 82, 30)
                        self.full_buttons[sid].setGeometry(self.wall_width - 50, 12, 36, 30)
                else:
                    left, top = self._tile_origin(sid)
                    self.title_labels[sid].setVisible(True)
                    self.title_labels[sid].move(left + 10, top + 10)
                    self.heat_buttons[sid].setGeometry(left + TILE_WIDTH - 142, top + 10, 82, 30)
                    self.full_buttons[sid].setGeometry(left + TILE_WIDTH - 50, top + 10, 36, 30)
                self.title_labels[sid].raise_()
                self.heat_buttons[sid].raise_()
                self.full_buttons[sid].raise_()

        def _show_hover(self, source_id: int | None) -> None:
            self.hovered_source = source_id
            for sid in range(self.camera_count):
                visible = source_id is not None and sid == source_id
                if self.focused_source is not None:
                    visible = sid == self.focused_source and source_id == self.focused_source
                self.heat_buttons[sid].setVisible(visible)
                self.full_buttons[sid].setVisible(visible)
                if visible:
                    self.heat_buttons[sid].raise_()
                    self.full_buttons[sid].raise_()

        def mouseMoveEvent(self, event: QMouseEvent) -> None:
            source_id = self.focused_source if self.focused_source is not None else self.source_at(event.position().toPoint())
            self._show_hover(source_id)
            super().mouseMoveEvent(event)

        def leaveEvent(self, event) -> None:
            local = self.mapFromGlobal(self.cursor().pos())
            if not self.rect().contains(local):
                self._show_hover(None)
            super().leaveEvent(event)

        def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
            sid = self.focused_source if self.focused_source is not None else self.source_at(event.position().toPoint())
            if sid is not None:
                self.toggle_focus(sid)
            super().mouseDoubleClickEvent(event)

        def open_heatmap(self, source_id: int) -> None:
            self.heatmap_source = source_id
            self.set_focus(source_id)
            self.controller.heatmap(True)

        def toggle_focus(self, source_id: int) -> None:
            if self.focused_source == source_id:
                self.set_focus(None)
            else:
                self.heatmap_source = None
                self.controller.heatmap(False)
                self.set_focus(source_id)

        def set_focus(self, source_id: int | None) -> None:
            if source_id is not None and not 0 <= source_id < self.camera_count:
                source_id = None
            self.focused_source = source_id
            self.controller.focus(source_id)
            if source_id is None:
                self.heatmap_source = None
                self.controller.heatmap(False)
            self._layout_overlays()
            self._show_hover(source_id if source_id is not None else None)
            self.focusChanged.emit(source_id)

    class MonitoringPage(QWidget):
        def __init__(self, controller: PipelineController):
            super().__init__()
            self.controller = controller
            self.camera_rows = _active_camera_rows()
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            header, h = page_header("Monitoring", "Live cameras · hover a camera for Heatmap or Fullscreen")
            action_row = QHBoxLayout()
            self.total_card = StatCard("Total people", "0", "Local NvDCF tracks")
            self.online_card = StatCard("Cameras online", "0/0", "Live RTSP health")
            self.total_card.setFixedWidth(190)
            self.online_card.setFixedWidth(190)
            action_row.addWidget(self.total_card)
            action_row.addWidget(self.online_card)
            action_row.addStretch(1)
            self.grid_button = QPushButton("Grid view")
            self.grid_button.clicked.connect(self._grid)
            self.grid_button.hide()
            action_row.addWidget(self.grid_button)
            self.status = QLabel("WAITING")
            self.status.setObjectName("Subtitle")
            action_row.addWidget(self.status)
            h.addLayout(action_row)
            root.addWidget(header)

            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(False)
            self.scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            root.addWidget(self.scroll, 1)
            self.wall: VideoWall | None = None
            self._build_wall()

        def _build_wall(self) -> None:
            self.camera_rows = _active_camera_rows()
            self.wall = VideoWall(self.controller, self.camera_rows)
            self.wall.nativeReady.connect(self._start_pipeline)
            self.wall.focusChanged.connect(self._focus_changed)
            self.scroll.setWidget(self.wall)

        def _start_pipeline(self, xid: int) -> None:
            self.status.setText("STARTING")
            self.controller.start(int(xid))

        def _focus_changed(self, source_id) -> None:
            self.grid_button.setVisible(source_id is not None)

        def _grid(self) -> None:
            if self.wall is not None:
                self.wall.set_focus(None)

        def update_status(self, status: UiStatus, metrics: dict) -> None:
            self.status.setText(status.state)
            self.status.setToolTip(status.detail)
            total = int(metrics.get("total_people", 0) or 0)
            online = int(metrics.get("online_cameras", 0) or 0)
            count = int(metrics.get("camera_count", len(self.camera_rows)) or 0)
            self.total_card.value.setText(str(total))
            self.online_card.value.setText(f"{online}/{count}")
            if status.state == "LIVE":
                self.status.setStyleSheet("color:#42d89c;font-weight:700;")
            elif status.state in {"ERROR", "PIPELINE_WARNING"}:
                self.status.setStyleSheet("color:#ff6b7b;font-weight:700;")
            else:
                self.status.setStyleSheet("color:#7890a8;font-weight:700;")

        def reload_pipeline(self) -> None:
            self.controller.stop()
            old = self.wall
            if old is not None:
                old.setParent(None)
                old.deleteLater()
            self._build_wall()
            self.status.setText("RELOADING")

        def close(self) -> None:
            self.controller.stop()

    class WorkerCard(QFrame):
        def __init__(self, row: dict):
            super().__init__()
            self.setObjectName("Card")
            lay = QHBoxLayout(self)
            lay.setContentsMargins(14, 14, 14, 14)
            photo = QLabel()
            photo.setFixedSize(72, 72)
            photo.setAlignment(Qt.AlignCenter)
            path = str(row.get("profile_photo", ""))
            pix = QPixmap(path) if path and Path(path).is_file() else QPixmap()
            if not pix.isNull():
                photo.setPixmap(pix.scaled(72, 72, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            else:
                initials = "".join(part[:1] for part in str(row.get("name", "?")).split()[:2]).upper()
                photo.setText(initials or "?")
                photo.setStyleSheet("background:#163342;color:#55e0d6;font-size:22px;font-weight:800;border-radius:6px;")
            lay.addWidget(photo)
            text = QVBoxLayout()
            name = QLabel(str(row.get("name", "Unknown")))
            name.setStyleSheet("font-size:16px;font-weight:800;color:#edf3f8;")
            person_id = QLabel(str(row.get("id", "")))
            person_id.setObjectName("Subtitle")
            role = QLabel(" · ".join(x for x in (str(row.get("role", "")), str(row.get("department", ""))) if x) or "Worker")
            role.setObjectName("Subtitle")
            tag = QLabel("KNOWN WORKER")
            tag.setStyleSheet("color:#42d89c;font-size:10px;font-weight:800;")
            text.addWidget(name)
            text.addWidget(person_id)
            text.addWidget(role)
            text.addWidget(tag)
            lay.addLayout(text, 1)

    class PeoplePage(QWidget):
        def __init__(self, store: SentinelStore):
            super().__init__()
            self.store = store
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            header, h = page_header("Workers", "Known people only · enrolled worker profiles")
            self.search = QLineEdit()
            self.search.setPlaceholderText("Search worker name, role or department")
            self.search.textChanged.connect(self.refresh)
            h.addWidget(self.search)
            root.addWidget(header)
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.body = QWidget()
            self.grid = QGridLayout(self.body)
            self.grid.setContentsMargins(26, 16, 26, 26)
            self.grid.setSpacing(12)
            self.scroll.setWidget(self.body)
            root.addWidget(self.scroll, 1)
            self.refresh()

        def refresh(self) -> None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            query = self.search.text().strip().lower() if hasattr(self, "search") else ""
            rows = self.store.list_people()
            if query:
                rows = [
                    row for row in rows
                    if query in " ".join(
                        (str(row.get("name", "")), str(row.get("role", "")), str(row.get("department", "")))
                    ).lower()
                ]
            if not rows:
                empty = QLabel("No known workers yet. Use Enrollment to add a worker profile.")
                empty.setObjectName("Subtitle")
                self.grid.addWidget(empty, 0, 0)
                return
            for index, row in enumerate(rows):
                card = WorkerCard(row)
                r, c = divmod(index, 3)
                self.grid.addWidget(card, r, c)
            for c in range(3):
                self.grid.setColumnStretch(c, 1)

    class EventsPage(QWidget):
        def __init__(self, store: SentinelStore):
            super().__init__()
            self.store = store
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            header, h = page_header(
                "Events",
                "Entry / exit snapshots are stored once per event; repeated frames are deduplicated",
            )
            self.type_filter = QComboBox()
            self.type_filter.addItems(["All events", "entry", "exit", "room_change", "restricted", "unknown"])
            self.type_filter.currentTextChanged.connect(self.refresh)
            h.addWidget(self.type_filter)
            root.addWidget(header)
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.body = QWidget()
            self.list_layout = QVBoxLayout(self.body)
            self.list_layout.setContentsMargins(26, 16, 26, 26)
            self.list_layout.setSpacing(10)
            self.scroll.setWidget(self.body)
            root.addWidget(self.scroll, 1)
            self.refresh()

        def refresh(self) -> None:
            while self.list_layout.count():
                item = self.list_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            selected = self.type_filter.currentText() if hasattr(self, "type_filter") else "All events"
            rows = self.store.list_events()
            if selected != "All events":
                rows = [row for row in rows if str(row.get("event_type", "")) == selected]
            if not rows:
                empty = QLabel("No events recorded yet. The event store is ready for the tracking/event hook.")
                empty.setObjectName("Subtitle")
                self.list_layout.addWidget(empty)
                self.list_layout.addStretch(1)
                return
            for row in rows:
                card = QFrame()
                card.setObjectName("Card")
                lay = QHBoxLayout(card)
                lay.setContentsMargins(14, 12, 14, 12)
                shot = QLabel()
                shot.setFixedSize(92, 62)
                shot.setAlignment(Qt.AlignCenter)
                p = str(row.get("snapshot_path", ""))
                pix = QPixmap(p) if p and Path(p).is_file() else QPixmap()
                if not pix.isNull():
                    shot.setPixmap(pix.scaled(92, 62, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
                else:
                    shot.setText("snapshot")
                    shot.setStyleSheet("background:#15212c;color:#6f879b;border-radius:5px;")
                lay.addWidget(shot)
                info = QVBoxLayout()
                event_type = str(row.get("event_type", "event")).replace("_", " ").title()
                who = str(row.get("person_name") or row.get("local_id") or row.get("person_id") or "Person")
                title = QLabel(f"{event_type} · {who}")
                title.setStyleSheet("font-weight:800;font-size:14px;")
                ts = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(float(row.get("created_at", 0))))
                meta = QLabel(f"{row.get('camera_id','')} · {row.get('room','')} · {ts}")
                meta.setObjectName("Subtitle")
                info.addWidget(title)
                info.addWidget(meta)
                lay.addLayout(info, 1)
                self.list_layout.addWidget(card)
            self.list_layout.addStretch(1)

    class RoomCard(QFrame):
        def __init__(self, room: str, cameras: list[dict]):
            super().__init__()
            self.setObjectName("Card")
            self.room = room
            self.cameras = list(cameras)
            self.setMinimumHeight(270)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(18, 16, 18, 16)
            top = QHBoxLayout()
            name = QLabel(room)
            name.setStyleSheet("font-size:18px;font-weight:800;")
            self.count = QLabel("—")
            self.count.setObjectName("BigNumber")
            top.addWidget(name)
            top.addStretch(1)
            top.addWidget(self.count)
            lay.addLayout(top)
            note = QLabel("Unique room occupancy needs Global ID; ReID is intentionally disabled now")
            note.setWordWrap(True)
            note.setObjectName("Subtitle")
            lay.addWidget(note)
            label = QLabel("CAMERAS")
            label.setObjectName("CardTitle")
            lay.addWidget(label)
            self.camera_labels: dict[str, QLabel] = {}
            for row in cameras:
                line = QHBoxLayout()
                cid = str(row.get("id", ""))
                text = QLabel(f"{cid}  ·  {row.get('name', cid)}")
                status = QLabel("waiting")
                status.setObjectName("Subtitle")
                line.addWidget(text)
                line.addStretch(1)
                line.addWidget(status)
                self.camera_labels[cid] = status
                lay.addLayout(line)
            lay.addStretch(1)

        def update_metrics(self, metrics: dict) -> None:
            by_id = {str(row.get("id")): row for row in metrics.get("cameras", [])}
            for cid, label in self.camera_labels.items():
                row = by_id.get(cid, {})
                if row.get("online"):
                    label.setText(f"{float(row.get('fps',0)):.1f} fps")
                    label.setStyleSheet("color:#42d89c;")
                else:
                    label.setText("offline")
                    label.setStyleSheet("color:#ff6b7b;")

    class RoomsPage(QWidget):
        def __init__(self):
            super().__init__()
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            header, _h = page_header("Rooms", "Entrance · Devs · Main Rooms")
            root.addWidget(header)
            self.cards_widget = QWidget()
            self.cards = QHBoxLayout(self.cards_widget)
            self.cards.setContentsMargins(26, 16, 26, 26)
            self.cards.setSpacing(14)
            root.addWidget(self.cards_widget, 1)
            self.room_cards: dict[str, RoomCard] = {}
            self.refresh()

        def refresh(self) -> None:
            while self.cards.count():
                item = self.cards.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            self.room_cards.clear()
            mapping = room_cameras()
            for room in ROOMS:
                card = RoomCard(room, mapping.get(room, []))
                self.room_cards[room] = card
                self.cards.addWidget(card, 1)

        def update_metrics(self, metrics: dict) -> None:
            for card in self.room_cards.values():
                card.update_metrics(metrics)

    class EnrollmentPage(QWidget):
        enrolled = Signal()

        def __init__(self, store: SentinelStore):
            super().__init__()
            self.store = store
            self.images: list[str] = []
            self.profile_index = -1
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            header, _h = page_header("Enrollment", "Select exactly 10 worker face images and one profile photo")
            root.addWidget(header)
            body = QHBoxLayout()
            body.setContentsMargins(26, 16, 26, 26)
            body.setSpacing(16)
            form_card = QFrame()
            form_card.setObjectName("Card")
            form_card.setFixedWidth(330)
            form = QVBoxLayout(form_card)
            form.setContentsMargins(18, 18, 18, 18)
            title = QLabel("Worker information")
            title.setStyleSheet("font-size:17px;font-weight:800;")
            form.addWidget(title)
            self.name = QLineEdit()
            self.name.setPlaceholderText("Full name")
            self.role = QLineEdit()
            self.role.setPlaceholderText("Role")
            self.department = QLineEdit()
            self.department.setPlaceholderText("Department")
            self.notes = QTextEdit()
            self.notes.setPlaceholderText("Notes")
            self.notes.setFixedHeight(90)
            form.addWidget(self.name)
            form.addWidget(self.role)
            form.addWidget(self.department)
            form.addWidget(self.notes)
            self.profile_preview = QLabel("Profile photo not selected")
            self.profile_preview.setAlignment(Qt.AlignCenter)
            self.profile_preview.setFixedHeight(170)
            self.profile_preview.setStyleSheet("background:#0a121b;border:1px dashed #294052;border-radius:6px;color:#6f879b;")
            form.addWidget(self.profile_preview)
            self.summary = QLabel("Images: 0/10")
            self.summary.setObjectName("Subtitle")
            form.addWidget(self.summary)
            enroll = QPushButton("Enroll worker")
            enroll.setObjectName("Primary")
            enroll.clicked.connect(self._enroll)
            form.addWidget(enroll)
            body.addWidget(form_card)

            images_card = QFrame()
            images_card.setObjectName("Card")
            image_lay = QVBoxLayout(images_card)
            image_lay.setContentsMargins(18, 18, 18, 18)
            top = QHBoxLayout()
            t = QLabel("10 face images")
            t.setStyleSheet("font-size:17px;font-weight:800;")
            choose = QPushButton("+ Select 10 images")
            choose.setObjectName("Primary")
            choose.clicked.connect(self._choose)
            top.addWidget(t)
            top.addStretch(1)
            top.addWidget(choose)
            image_lay.addLayout(top)
            hint = QLabel("Use clear images of the same worker from different head/body angles. Click a tile to choose profile photo.")
            hint.setWordWrap(True)
            hint.setObjectName("Subtitle")
            image_lay.addWidget(hint)
            grid = QGridLayout()
            grid.setSpacing(10)
            self.image_buttons: list[QPushButton] = []
            for index in range(10):
                button = QPushButton(f"+\nImage {index + 1}")
                button.setFixedSize(150, 120)
                button.setIconSize(QSize(140, 108))
                button.clicked.connect(lambda _checked=False, i=index: self._select_profile(i))
                self.image_buttons.append(button)
                r, c = divmod(index, 5)
                grid.addWidget(button, r, c)
            image_lay.addLayout(grid)
            image_lay.addStretch(1)
            body.addWidget(images_card, 1)
            root.addLayout(body, 1)

        def _choose(self) -> None:
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select exactly 10 face images",
                "",
                "Images (*.jpg *.jpeg *.png *.webp *.bmp)",
            )
            if not paths:
                return
            if len(paths) != 10:
                QMessageBox.warning(self, "Enrollment", "Select exactly 10 images.")
                return
            self.images = list(paths)
            self.profile_index = -1
            for index, button in enumerate(self.image_buttons):
                pix = QPixmap(self.images[index])
                button.setText("")
                button.setIcon(QIcon(pix))
                button.setStyleSheet("")
            self.profile_preview.clear()
            self.profile_preview.setText("Click one image to use as profile photo")
            self.summary.setText("Images: 10/10 · profile photo: not selected")

        def _select_profile(self, index: int) -> None:
            if len(self.images) != 10:
                return
            self.profile_index = index
            for i, button in enumerate(self.image_buttons):
                button.setStyleSheet(
                    "border:2px solid #42d8ca;" if i == index else ""
                )
            pix = QPixmap(self.images[index])
            if not pix.isNull():
                self.profile_preview.setPixmap(
                    pix.scaled(
                        self.profile_preview.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            self.summary.setText(f"Images: 10/10 · profile photo: Image {index + 1}")

        def _enroll(self) -> None:
            try:
                self.store.enroll_person(
                    name=self.name.text(),
                    role=self.role.text(),
                    department=self.department.text(),
                    notes=self.notes.toPlainText(),
                    image_paths=self.images,
                    profile_index=self.profile_index,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Enrollment", str(exc))
                return
            QMessageBox.information(self, "Enrollment", "Worker profile saved.")
            self.name.clear()
            self.role.clear()
            self.department.clear()
            self.notes.clear()
            self.images = []
            self.profile_index = -1
            self.profile_preview.setPixmap(QPixmap())
            self.profile_preview.setText("Profile photo not selected")
            self.summary.setText("Images: 0/10")
            for index, button in enumerate(self.image_buttons):
                button.setIcon(QIcon())
                button.setText(f"+\nImage {index + 1}")
                button.setStyleSheet("")
            self.enrolled.emit()

    class CameraDialog(QDialog):
        def __init__(self, row: dict | None = None, parent=None):
            super().__init__(parent)
            self.original_id = str(row.get("id")) if row else None
            self.setWindowTitle("Edit camera" if row else "Add camera")
            self.setMinimumWidth(520)
            form = QFormLayout(self)
            self.camera_id = QLineEdit(str(row.get("id")) if row else next_camera_id())
            self.name = QLineEdit(str(row.get("name", "")) if row else "")
            self.uri = QLineEdit(str(row.get("uri", "")) if row else "rtsp://")
            self.codec = QComboBox()
            self.codec.addItems(["h264", "h265"])
            self.codec.setCurrentText(str(row.get("codec", "h264")) if row else "h264")
            self.room = QComboBox()
            self.room.addItems(list(ROOMS))
            if row and str(row.get("room", "")) in ROOMS:
                self.room.setCurrentText(str(row.get("room")))
            self.username = QLineEdit(str(row.get("username", "")) if row else "")
            self.password = QLineEdit(str(row.get("password", "")) if row else "")
            self.password.setEchoMode(QLineEdit.Password)
            self.enabled = QCheckBox("Enabled")
            self.enabled.setChecked(bool(row.get("enabled", True)) if row else True)
            form.addRow("Camera ID", self.camera_id)
            form.addRow("Name", self.name)
            form.addRow("RTSP URL", self.uri)
            form.addRow("Codec", self.codec)
            form.addRow("Room", self.room)
            form.addRow("Username", self.username)
            form.addRow("Password", self.password)
            form.addRow("", self.enabled)
            note = QLabel("Saving restarts only the camera pipeline. UI profiles/events stay intact.")
            note.setWordWrap(True)
            note.setObjectName("Subtitle")
            form.addRow(note)
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            buttons.accepted.connect(self._save)
            buttons.rejected.connect(self.reject)
            form.addRow(buttons)

        def _save(self) -> None:
            try:
                save_camera(
                    {
                        "id": self.camera_id.text().strip(),
                        "name": self.name.text().strip(),
                        "uri": self.uri.text().strip(),
                        "codec": self.codec.currentText(),
                        "room": self.room.currentText(),
                        "username": self.username.text(),
                        "password": self.password.text(),
                        "enabled": self.enabled.isChecked(),
                    },
                    existing_id=self.original_id,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Camera", str(exc))
                return
            self.accept()

    class SettingsPage(QWidget):
        changed = Signal()

        def __init__(self):
            super().__init__()
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            header, h = page_header("Settings", "Camera management · add, edit, disable or remove RTSP cameras")
            action = QHBoxLayout()
            add = QPushButton("+ Add camera")
            add.setObjectName("Primary")
            add.clicked.connect(self._add)
            action.addWidget(add)
            action.addStretch(1)
            h.addLayout(action)
            root.addWidget(header)
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.body = QWidget()
            self.list_layout = QVBoxLayout(self.body)
            self.list_layout.setContentsMargins(26, 16, 26, 26)
            self.list_layout.setSpacing(10)
            self.scroll.setWidget(self.body)
            root.addWidget(self.scroll, 1)
            self.refresh()

        @staticmethod
        def _safe_uri(uri: str) -> str:
            return re.sub(r"(rtsp://)[^/@]+:[^/@]+@", r"\1***:***@", str(uri))

        def refresh(self) -> None:
            while self.list_layout.count():
                item = self.list_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            rows = list_cameras()
            for row in rows:
                card = QFrame()
                card.setObjectName("Card")
                lay = QHBoxLayout(card)
                lay.setContentsMargins(14, 12, 14, 12)
                info = QVBoxLayout()
                title = QLabel(f"{row.get('id')}  ·  {row.get('name', row.get('id'))}")
                title.setStyleSheet("font-size:15px;font-weight:800;")
                meta = QLabel(
                    f"{row.get('room','')} · {row.get('codec','')} · "
                    f"{'enabled' if row.get('enabled',True) else 'disabled'}"
                )
                meta.setObjectName("Subtitle")
                uri = QLabel(self._safe_uri(str(row.get("uri", ""))))
                uri.setObjectName("Subtitle")
                uri.setTextInteractionFlags(Qt.TextSelectableByMouse)
                info.addWidget(title)
                info.addWidget(meta)
                info.addWidget(uri)
                lay.addLayout(info, 1)
                edit = QPushButton("Edit")
                edit.clicked.connect(lambda _checked=False, r=dict(row): self._edit(r))
                remove = QPushButton("Remove")
                remove.setObjectName("Danger")
                remove.clicked.connect(lambda _checked=False, cid=str(row.get("id")): self._remove(cid))
                lay.addWidget(edit)
                lay.addWidget(remove)
                self.list_layout.addWidget(card)
            self.list_layout.addStretch(1)

        def _add(self) -> None:
            dialog = CameraDialog(parent=self)
            if dialog.exec() == QDialog.Accepted:
                self.refresh()
                self.changed.emit()

        def _edit(self, row: dict) -> None:
            dialog = CameraDialog(row, self)
            if dialog.exec() == QDialog.Accepted:
                self.refresh()
                self.changed.emit()

        def _remove(self, camera_id: str) -> None:
            if QMessageBox.question(
                self,
                "Remove camera",
                f"Remove {camera_id}? The live pipeline will restart.",
            ) != QMessageBox.Yes:
                return
            try:
                delete_camera(camera_id)
            except Exception as exc:
                QMessageBox.warning(self, "Camera", str(exc))
                return
            self.refresh()
            self.changed.emit()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SENTINEL VMS")
            self.setMinimumSize(1180, 760)
            self.controller = PipelineController()
            self.store = SentinelStore()

            central = QWidget()
            root = QHBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            sidebar = QFrame()
            sidebar.setObjectName("Sidebar")
            sidebar.setFixedWidth(220)
            side = QVBoxLayout(sidebar)
            side.setContentsMargins(10, 14, 10, 14)
            brand = QLabel("SENTINEL VMS")
            brand.setObjectName("Brand")
            side.addWidget(brand)
            sub = QLabel("local tracking · camera analytics")
            sub.setObjectName("Subtitle")
            side.addWidget(sub)
            side.addSpacing(16)

            self.stack = QStackedWidget()
            self.monitor = MonitoringPage(self.controller)
            self.people = PeoplePage(self.store)
            self.events = EventsPage(self.store)
            self.rooms = RoomsPage()
            self.enrollment = EnrollmentPage(self.store)
            self.settings_page = SettingsPage()
            pages = [
                ("▣  Monitoring", self.monitor),
                ("♙  Workers", self.people),
                ("⌁  Events", self.events),
                ("▥  Rooms", self.rooms),
                ("♙+ Enrollment", self.enrollment),
                ("⚙  Settings", self.settings_page),
            ]
            self.nav_buttons: list[QPushButton] = []
            for index, (text, page) in enumerate(pages):
                button = QPushButton(text)
                button.setObjectName("Nav")
                button.setCheckable(True)
                button.setCursor(Qt.PointingHandCursor)
                button.clicked.connect(lambda checked=False, i=index: self._navigate(i))
                self.nav_buttons.append(button)
                side.addWidget(button)
                self.stack.addWidget(page)
            side.addStretch(1)
            footer = QLabel("build 2026.08 · local-only ReID off")
            footer.setObjectName("Subtitle")
            side.addWidget(footer)
            root.addWidget(sidebar)
            root.addWidget(self.stack, 1)
            self.setCentralWidget(central)

            self.nav_buttons[0].setChecked(True)
            self.enrollment.enrolled.connect(self.people.refresh)
            self.settings_page.changed.connect(self._camera_config_changed)

            self.poll_timer = QTimer(self)
            self.poll_timer.timeout.connect(self._poll)
            self.poll_timer.start(200)

        def _navigate(self, index: int) -> None:
            self.stack.setCurrentIndex(index)
            for i, button in enumerate(self.nav_buttons):
                button.setChecked(i == index)
            if index == 1:
                self.people.refresh()
            elif index == 2:
                self.events.refresh()
            elif index == 3:
                self.rooms.refresh()

        def _camera_config_changed(self) -> None:
            self.rooms.refresh()
            self.monitor.reload_pipeline()

        def _poll(self) -> None:
            status, metrics = self.controller.poll()
            self.monitor.update_status(status, metrics)
            self.rooms.update_metrics(metrics)

        def keyPressEvent(self, event: QKeyEvent) -> None:
            if event.key() == Qt.Key_Escape and self.monitor.wall is not None and self.monitor.wall.focused_source is not None:
                self.monitor.wall.set_focus(None)
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
            self.monitor.close()
            event.accept()

    app = QApplication(sys.argv)
    app.setApplicationName("SENTINEL VMS")
    app.setOrganizationName("Apsidal")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
