from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QWidget

CAMERA_COUNT = 6
GRID_COLUMNS = 2
GRID_ROWS = 3
WALL_WIDTH = 1280
WALL_HEIGHT = 1080


@dataclass(frozen=True)
class UiStatus:
    state: str
    detail: str = ""


def _put_status(status_q, state: str, detail) -> None:
    try:
        status_q.put_nowait((state, detail))
    except Exception:
        pass


def _pipeline_process(window_id: int, command_q, status_q) -> None:
    runtime = None
    try:
        if int(window_id) <= 0:
            raise RuntimeError("invalid Qt native window id")

        os.environ["CAMERA_V2_WALL_WIDTH"] = str(WALL_WIDTH)
        os.environ["CAMERA_V2_WALL_HEIGHT"] = str(WALL_HEIGHT)

        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        from .person_tracking_final import CameraPersonTrackingFinal

        runtime = CameraPersonTrackingFinal()
        if len(runtime.cameras) != CAMERA_COUNT:
            raise RuntimeError(
                f"Sentinel Monitoring expects {CAMERA_COUNT} enabled cameras, "
                f"found {len(runtime.cameras)}"
            )

        runtime.wall_width = WALL_WIDTH
        runtime.wall_height = WALL_HEIGHT
        runtime.tiler.set_property("rows", GRID_ROWS)
        runtime.tiler.set_property("columns", GRID_COLUMNS)
        runtime.tiler.set_property("width", WALL_WIDTH)
        runtime.tiler.set_property("height", WALL_HEIGHT)
        if runtime.tiler.find_property("show-source") is not None:
            runtime.tiler.set_property("show-source", -1)

        current_xid = int(window_id)

        def bind_overlay(overlay, xid: int | None = None) -> None:
            nonlocal current_xid
            target = int(current_xid if xid is None else xid)
            if target <= 0:
                return
            current_xid = target
            GstVideo.VideoOverlay.set_window_handle(overlay, target)
            try:
                GstVideo.VideoOverlay.handle_events(overlay, False)
            except Exception:
                pass

        bind_overlay(runtime.sink, current_xid)

        def on_sync_message(_bus, message, _data=None):
            try:
                prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
            except Exception:
                structure = message.get_structure()
                prepare = bool(
                    structure and structure.get_name() == "prepare-window-handle"
                )
            if not prepare:
                return Gst.BusSyncReply.PASS
            try:
                bind_overlay(message.src)
                _put_status(status_q, "VIDEO_BOUND", f"xid={current_xid}")
                return Gst.BusSyncReply.DROP
            except Exception as exc:
                _put_status(status_q, "ERROR", f"video overlay: {exc}")
                return Gst.BusSyncReply.PASS

        runtime.bus.set_sync_handler(on_sync_message, None)

        def observe_bus(_bus, message):
            if (
                message.type == Gst.MessageType.STATE_CHANGED
                and message.src == runtime.pipeline
            ):
                try:
                    _old, new, _pending = message.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        _put_status(
                            status_q,
                            "LIVE",
                            "6-camera detector/tracker pipeline PLAYING",
                        )
                except Exception:
                    pass
            elif message.type == Gst.MessageType.ERROR:
                try:
                    err, _debug = message.parse_error()
                    src = message.src.get_name() if message.src else "unknown"
                    _put_status(
                        status_q,
                        "PIPELINE_WARNING",
                        f"{src}: {err.message}",
                    )
                except Exception:
                    pass

        runtime.bus.connect("message", observe_bus)

        def poll_commands() -> bool:
            stop_requested = False
            latest_focus = None
            got_focus = False
            latest_bind = None
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
                elif command == "bind":
                    latest_bind = int(value)

            if latest_bind is not None and latest_bind > 0:
                bind_overlay(runtime.sink, latest_bind)
                _put_status(status_q, "VIDEO_BOUND", f"xid={latest_bind}")

            if got_focus and runtime.tiler.find_property("show-source") is not None:
                source_id = (
                    latest_focus
                    if latest_focus is not None
                    and 0 <= latest_focus < CAMERA_COUNT
                    else -1
                )
                runtime.tiler.set_property("show-source", source_id)
                _put_status(status_q, "FOCUS", str(source_id))

            if stop_requested:
                runtime.stop()
                return False
            return True

        last_frames = {camera.camera_id: 0 for camera in runtime.cameras}
        last_seen = {camera.camera_id: 0.0 for camera in runtime.cameras}
        last_metric_t = time.monotonic()

        def publish_metrics() -> bool:
            nonlocal last_metric_t
            now = time.monotonic()
            elapsed = max(0.20, now - last_metric_t)
            last_metric_t = now
            rows = []
            for index, camera in enumerate(runtime.cameras):
                stat = runtime.stats[camera.camera_id]
                previous = int(last_frames.get(camera.camera_id, 0))
                current = int(stat.frames)
                delta = max(0, current - previous)
                last_frames[camera.camera_id] = current
                if delta > 0:
                    last_seen[camera.camera_id] = now
                online = now - last_seen.get(camera.camera_id, 0.0) <= 2.5
                rows.append(
                    {
                        "id": camera.camera_id,
                        "source_id": index,
                        "fps": delta / elapsed,
                        "online": online,
                    }
                )
            _put_status(
                status_q,
                "METRICS",
                {
                    "cameras": rows,
                    "total_people": int(getattr(runtime, "tracked_now", 0)),
                },
            )
            return True

        runtime.GLib.timeout_add(50, poll_commands)
        runtime.GLib.timeout_add(500, publish_metrics)
        _put_status(
            status_q,
            "STARTING",
            "2 columns x 3 rows; live DeepStream wall; detector/tracker unchanged",
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
        self.metrics: dict = {"cameras": [], "total_people": 0}

    def _reset_queues(self) -> None:
        self.command_q = self.ctx.Queue(maxsize=64)
        self.status_q = self.ctx.Queue(maxsize=128)

    def start_or_bind(self, window_id: int) -> None:
        xid = int(window_id)
        if xid <= 0:
            return
        if self.process is not None and self.process.is_alive():
            self.bind(xid)
            return
        self.process = self.ctx.Process(
            target=_pipeline_process,
            args=(xid, self.command_q, self.status_q),
            name="sentinel-live-camera-wall",
            daemon=False,
        )
        self.process.start()
        self.last_status = UiStatus("STARTING")

    def bind(self, window_id: int) -> None:
        try:
            self.command_q.put_nowait(("bind", int(window_id)))
        except queue.Full:
            pass

    def focus(self, source_id: int | None) -> None:
        value = -1 if source_id is None else int(source_id)
        try:
            self.command_q.put_nowait(("focus", value))
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


class LiveVideoWall(QFrame):
    nativeReady = Signal(int)
    cameraDoubleClicked = Signal(int)

    def __init__(self, cameras, people, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cameras = list(cameras)
        self.people = list(people)
        self._native_emitted = False
        self._metrics: dict = {"cameras": []}
        self.setObjectName("panel")
        self.setMinimumSize(640, 540)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setAttribute(Qt.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        _ = int(self.winId())

        self.camera_labels: list[QLabel] = []
        self.status_labels: list[QLabel] = []
        self.occupancy_labels: list[QLabel] = []
        inside = [person for person in self.people if getattr(person, "in_building", False)]
        for index, camera in enumerate(self.cameras[:CAMERA_COUNT]):
            camera_label = QLabel(f"CAM-{index + 1:02d}", self)
            camera_label.setStyleSheet(
                "background:rgba(8,14,20,220);color:#e7edf3;"
                "border:1px solid rgba(80,105,125,120);border-radius:4px;"
                "padding:3px 7px;font-weight:700;"
            )
            camera_label.adjustSize()
            self.camera_labels.append(camera_label)

            status_label = QLabel("CONNECTING", self)
            status_label.setStyleSheet(
                "background:rgba(8,14,20,205);color:#7e8c99;"
                "border-radius:4px;padding:3px 6px;font:10px 'DejaVu Sans Mono';"
            )
            status_label.adjustSize()
            self.status_labels.append(status_label)

            room_id = getattr(camera, "room_id", None)
            occupancy = len(
                [person for person in inside if getattr(person, "room_id", None) == room_id]
            )
            occupancy_label = QLabel(f"●  {occupancy}", self)
            occupancy_label.setStyleSheet(
                "background:rgba(8,14,20,220);color:#39d9c5;"
                "border:1px solid rgba(57,217,197,75);border-radius:4px;"
                "padding:3px 7px;font:700 10px 'DejaVu Sans Mono';"
            )
            occupancy_label.adjustSize()
            self.occupancy_labels.append(occupancy_label)

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
            QTimer.singleShot(100, lambda: self.nativeReady.emit(xid))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_overlays()

    def _tile_rect(self, source_id: int) -> tuple[int, int, int, int]:
        col = source_id % GRID_COLUMNS
        row = source_id // GRID_COLUMNS
        left = round(col * self.width() / GRID_COLUMNS)
        right = round((col + 1) * self.width() / GRID_COLUMNS)
        top = round(row * self.height() / GRID_ROWS)
        bottom = round((row + 1) * self.height() / GRID_ROWS)
        return left, top, max(1, right - left), max(1, bottom - top)

    def _layout_overlays(self) -> None:
        for sid in range(min(CAMERA_COUNT, len(self.camera_labels))):
            left, top, width, height = self._tile_rect(sid)
            cam = self.camera_labels[sid]
            stat = self.status_labels[sid]
            occ = self.occupancy_labels[sid]
            cam.move(left + 10, top + 10)
            stat.adjustSize()
            stat.move(left + width - stat.width() - 10, top + 10)
            occ.move(left + 10, top + height - occ.height() - 10)
            cam.raise_()
            stat.raise_()
            occ.raise_()

    def source_at(self, pos: QPoint) -> int | None:
        if self.width() <= 0 or self.height() <= 0:
            return None
        x, y = pos.x(), pos.y()
        if x < 0 or y < 0 or x >= self.width() or y >= self.height():
            return None
        col = min(GRID_COLUMNS - 1, int(x * GRID_COLUMNS / self.width()))
        row = min(GRID_ROWS - 1, int(y * GRID_ROWS / self.height()))
        sid = row * GRID_COLUMNS + col
        return sid if 0 <= sid < CAMERA_COUNT else None

    def mouseDoubleClickEvent(self, event):
        sid = self.source_at(event.position().toPoint())
        if sid is not None:
            self.cameraDoubleClicked.emit(sid)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def update_metrics(self, metrics: dict) -> None:
        self._metrics = dict(metrics or {})
        by_source = {
            int(row.get("source_id", -1)): row
            for row in self._metrics.get("cameras", [])
            if isinstance(row, dict)
        }
        for sid, status in enumerate(self.status_labels):
            row = by_source.get(sid)
            if not row:
                text = "CONNECTING"
                color = "#7e8c99"
            elif row.get("online"):
                text = f"{float(row.get('fps', 0.0)):.1f} fps"
                color = "#3ddc97"
            else:
                text = "OFFLINE"
                color = "#f06464"
            status.setText(text)
            status.setStyleSheet(
                f"background:rgba(8,14,20,205);color:{color};"
                "border-radius:4px;padding:3px 6px;font:10px 'DejaVu Sans Mono';"
            )
            status.adjustSize()
        self._layout_overlays()
