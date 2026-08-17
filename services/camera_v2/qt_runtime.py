from __future__ import annotations

import csv
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path


WALL_WIDTH = 1024
WALL_HEIGHT = 864
GRID_ROWS = 3
GRID_COLUMNS = 2
CAMERA_COUNT = 6


def _put_latest(q, payload) -> None:
    try:
        q.put_nowait(payload)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except Exception:
        pass
    try:
        q.put_nowait(payload)
    except Exception:
        pass


def _pipeline_process(window_id: int, command_q, state_q, status_q) -> None:
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

        from .sentinel_live_runtime import SentinelLiveRuntime

        runtime = SentinelLiveRuntime()
        if len(runtime.cameras) != CAMERA_COUNT:
            raise RuntimeError(f"Sentinel UI requires 6 cameras, found {len(runtime.cameras)}")

        runtime.tiler.set_property("rows", GRID_ROWS)
        runtime.tiler.set_property("columns", GRID_COLUMNS)
        runtime.tiler.set_property("width", WALL_WIDTH)
        runtime.tiler.set_property("height", WALL_HEIGHT)
        if runtime.tiler.find_property("show-source") is not None:
            runtime.tiler.set_property("show-source", -1)

        current_xid = [int(window_id)]

        def bind_overlay(overlay, xid: int | None = None) -> None:
            target = int(current_xid[0] if xid is None else xid)
            if target <= 0:
                raise RuntimeError("invalid video window id")
            current_xid[0] = target
            GstVideo.VideoOverlay.set_window_handle(overlay, target)
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
                _put_latest(status_q, ("VIDEO_BOUND", f"xid={current_xid[0]}"))
                return Gst.BusSyncReply.DROP
            except Exception as exc:
                _put_latest(status_q, ("ERROR", f"video overlay: {exc}"))
                return Gst.BusSyncReply.PASS

        runtime.bus.set_sync_handler(on_sync_message, None)

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
                try:
                    bind_overlay(runtime.sink, latest_bind)
                    _put_latest(status_q, ("VIDEO_BOUND", f"xid={latest_bind}"))
                except Exception as exc:
                    _put_latest(status_q, ("ERROR", f"rebind failed: {exc}"))

            if got_focus and runtime.tiler.find_property("show-source") is not None:
                source_id = latest_focus if 0 <= latest_focus < CAMERA_COUNT else -1
                runtime.tiler.set_property("show-source", source_id)

            if stop_requested:
                runtime.stop()
                return False
            return True

        def publish_snapshot() -> bool:
            try:
                _put_latest(state_q, runtime.ui_snapshot())
            except Exception as exc:
                _put_latest(status_q, ("STATE_WARNING", f"{type(exc).__name__}: {exc}"))
            return not runtime._stopping

        runtime.GLib.timeout_add(50, poll_commands)
        runtime.GLib.timeout_add(200, publish_snapshot)
        _put_latest(
            status_q,
            (
                "STARTING",
                f"DeepStream 2x3 wall {WALL_WIDTH}x{WALL_HEIGHT}; YOLO26m + NvDCF; frame_copy=0 mjpeg=0",
            ),
        )

        rc = runtime.run()
        _put_latest(status_q, ("STOPPED", f"exit={rc}"))
    except BaseException as exc:
        _put_latest(status_q, ("ERROR", f"{type(exc).__name__}: {exc}"))
        try:
            if runtime is not None:
                runtime.stop()
                runtime.pipeline.set_state(runtime.Gst.State.NULL)
        except Exception:
            pass


class CameraQtController:
    """Process-isolated controller used by the exact Sentinel Qt shell.

    Qt stays in the parent process. The proven DeepStream/YOLO/NvDCF pipeline runs
    in a spawned child and renders directly into the Qt X11 WId with GstVideoOverlay.
    UI state crosses the process boundary only as small dictionaries.
    """

    def __init__(self) -> None:
        self.ctx = mp.get_context("spawn")
        self.command_q = self.ctx.Queue(maxsize=32)
        self.state_q = self.ctx.Queue(maxsize=1)
        self.status_q = self.ctx.Queue(maxsize=8)
        self.process: mp.Process | None = None
        self._status = "WAITING"
        self._error = ""
        self._focus_source: int | None = None
        self._window_handle = 0
        self._heat_enabled = [False] * CAMERA_COUNT
        self._last_snapshot = self._offline_snapshot()
        self._last_snapshot_mono = 0.0

    @staticmethod
    def _offline_snapshot() -> dict:
        return {
            "timestamp": time.time(),
            "cameras": [
                {
                    "source_id": i,
                    "camera_id": f"CAM-{i + 1:02d}",
                    "room_id": i // 2 + 1,
                    "fps": 0.0,
                    "online": False,
                    "count": 0,
                }
                for i in range(CAMERA_COUNT)
            ],
            "tracks": [],
            "events": [],
            "rooms": [
                {"room_id": 1, "name": "Lobbi", "count": 0, "camera_counts": [0, 0], "camera_ids": ["CAM-01", "CAM-02"]},
                {"room_id": 2, "name": "Ofis", "count": 0, "camera_counts": [0, 0], "camera_ids": ["CAM-03", "CAM-04"]},
                {"room_id": 3, "name": "Ombor", "count": 0, "camera_counts": [0, 0], "camera_ids": ["CAM-05", "CAM-06"]},
            ],
            "detector": {"ready": False, "error": "", "batch_ms": 0.0, "result_age_ms": 0.0, "tracked_now": 0},
        }

    @property
    def camera_count(self) -> int:
        return CAMERA_COUNT

    @property
    def status(self) -> str:
        self._drain_status()
        return self._status

    @property
    def error(self) -> str:
        self._drain_status()
        return self._error

    def _drain_status(self) -> None:
        latest = None
        while True:
            try:
                latest = self.status_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            state, detail = str(latest[0]), str(latest[1])
            self._status = state
            self._error = detail if state in {"ERROR", "STATE_WARNING"} else ""

        process = self.process
        if process is not None and not process.is_alive() and self._status not in {"ERROR", "STOPPED"}:
            self._status = "ERROR"
            self._error = f"camera process exited: {process.exitcode}"

    def start(self, win_id: int) -> None:
        win_id = int(win_id)
        if win_id <= 0:
            self._status = "ERROR"
            self._error = "invalid Qt video window id"
            return

        self._window_handle = win_id
        if self.process is not None and self.process.is_alive():
            self.bind_window(win_id)
            return

        self.process = self.ctx.Process(
            target=_pipeline_process,
            args=(win_id, self.command_q, self.state_q, self.status_q),
            name="sentinel-camera-v2",
            daemon=False,
        )
        self.process.start()
        self._status = "STARTING"
        self._error = ""

    def bind_window(self, win_id: int) -> None:
        win_id = int(win_id)
        if win_id <= 0:
            return
        self._window_handle = win_id
        if self.process is not None and self.process.is_alive():
            try:
                self.command_q.put_nowait(("bind", win_id))
            except queue.Full:
                pass

    def set_focus_source(self, source_id: int | None) -> None:
        self._focus_source = source_id
        value = -1 if source_id is None else int(source_id)
        if self.process is not None and self.process.is_alive():
            try:
                self.command_q.put_nowait(("focus", value))
            except queue.Full:
                pass

    def focus_source(self) -> int | None:
        return self._focus_source

    # Compatibility with the historical shell. The supplied exact UI contains no
    # heatmap control, so these do not alter the native video.
    def set_heatmap_enabled(self, source_id: int, enabled: bool) -> None:
        source_id = int(source_id)
        if 0 <= source_id < CAMERA_COUNT:
            self._heat_enabled[source_id] = bool(enabled)

    def heatmap_enabled(self, source_id: int) -> bool:
        source_id = int(source_id)
        return 0 <= source_id < CAMERA_COUNT and self._heat_enabled[source_id]

    def heat_points(self, _source_id: int, max_points: int = 30):
        _ = max_points
        return []

    def snapshot(self) -> dict:
        latest = None
        while True:
            try:
                latest = self.state_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._last_snapshot = latest
            self._last_snapshot_mono = time.monotonic()

        self._drain_status()

        if self.process is not None and not self.process.is_alive():
            fallback = self._offline_snapshot()
            fallback["events"] = list(self._last_snapshot.get("events", []))
            self._last_snapshot = fallback
        elif self._last_snapshot_mono and time.monotonic() - self._last_snapshot_mono > 3.0:
            stale = dict(self._last_snapshot)
            stale["cameras"] = [dict(c, online=False, fps=0.0) for c in stale.get("cameras", [])]
            self._last_snapshot = stale

        return {
            key: ([dict(row) for row in value] if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
            for key, value in self._last_snapshot.items()
        }

    def export_events_csv(self, path: str | Path) -> None:
        events = self.snapshot().get("events", [])
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "type", "camera_id", "room_id", "person_id", "label", "message"])
            for event in events:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(event.get("time") or 0.0))),
                    event.get("type", ""),
                    event.get("camera_id", ""),
                    event.get("room_id", ""),
                    event.get("person_id", ""),
                    event.get("label", ""),
                    event.get("message", ""),
                ])

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
        self._status = "STOPPED"
