from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QWidget

CAMERA_COUNT = 6
GRID_COLUMNS = 2
GRID_ROWS = 3
WALL_WIDTH = 1600
WALL_HEIGHT = 1350


@dataclass(frozen=True)
class UiStatus:
    state: str
    detail: str = ""


def _put_status(status_q, state: str, detail) -> None:
    try:
        status_q.put_nowait((state, detail))
    except Exception:
        pass


class PipelineController:
    """Queue/process lifecycle shared by the production Pro controller.

    The old base controller launched an obsolete ReID runtime. Production now has
    one launcher only: ProPipelineController in sentinel_video_pro.py.
    """

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
        raise NotImplementedError("Use ProPipelineController for the Sentinel production runtime")

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
        self._status_cache: dict[int, tuple[str, str]] = {}

        # This QWidget is not a normal Qt-painted panel: nveglglessink owns its
        # native X11 surface through GstVideoOverlay. Do not let the application
        # QSS or Qt's background erase race with EGL video frames.
        self.setObjectName("nativeVideoSurface")
        self.setFrameShape(QFrame.NoFrame)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setMinimumSize(640, 540)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.setAttribute(Qt.WA_DontCreateNativeAncestors, True)
        self.setAttribute(Qt.WA_NativeWindow, True)
        _ = int(self.winId())

        self.camera_labels: list[QLabel] = []
        self.status_labels: list[QLabel] = []
        self.occupancy_labels: list[QLabel] = []

        for index, _camera in enumerate(self.cameras[:CAMERA_COUNT]):
            camera_label = QLabel(f"CAM-{index + 1:02d}", self)
            camera_label.setStyleSheet(
                "background:rgba(8,14,20,224);color:#e7edf3;"
                "border:1px solid rgba(80,105,125,120);border-radius:4px;"
                "padding:3px 7px;font-weight:700;"
            )
            camera_label.adjustSize()
            self.camera_labels.append(camera_label)

            status_label = QLabel("CONNECTING", self)
            status_label.setStyleSheet(
                "background:transparent;border:0;color:#7e8c99;"
                "padding:0;font:700 9px 'DejaVu Sans Mono';"
            )
            status_label.adjustSize()
            self.status_labels.append(status_label)
            self._status_cache[index] = ("CONNECTING", "#7e8c99")

        self._layout_overlays()

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
            left, top, width, _height = self._tile_rect(sid)
            cam = self.camera_labels[sid]
            stat = self.status_labels[sid]
            cam.move(left + 10, top + 10)
            stat.adjustSize()
            stat.move(left + width - stat.width() - 10, top + 10)
            cam.raise_()
            stat.raise_()

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

        layout_needed = False
        for sid, status in enumerate(self.status_labels):
            row = by_source.get(sid)
            if not row:
                text = "CONNECTING"
                color = "#7e8c99"
            elif row.get("online"):
                # The display does not need sub-frame FPS precision. Integer FPS
                # prevents a 19.9/20.0/20.1 label from repainting the native video
                # surface several times per second while the actual stream is fine.
                text = f"{int(round(float(row.get('fps', 0.0))))} fps"
                color = "#3ddc97"
            else:
                text = "OFFLINE"
                color = "#f06464"

            render_key = (text, color)
            if self._status_cache.get(sid) == render_key:
                continue

            self._status_cache[sid] = render_key
            status.setText(text)
            status.setStyleSheet(
                f"background:transparent;border:0;color:{color};"
                "padding:0;font:700 9px 'DejaVu Sans Mono';"
            )
            status.adjustSize()
            layout_needed = True

        if layout_needed:
            self._layout_overlays()
