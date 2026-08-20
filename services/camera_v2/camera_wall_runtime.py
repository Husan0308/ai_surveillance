from __future__ import annotations

"""Process-isolated controller for the production 2x3 camera wall."""

import gc
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass

CAMERA_COUNT = 6
GRID_COLUMNS = 2
GRID_ROWS = 3
WALL_WIDTH = 1600
# nvmultistreamtiler/NVMM aligns the requested 1350 surface to 1352 on the
# target Pascal stack. Keep the production caps geometry aligned with the
# negotiated surface so downstream caps do not fail with not-negotiated (-4).
WALL_HEIGHT = 1352
FOCUS_WIDTH = 1920
FOCUS_HEIGHT = 1080


@dataclass(frozen=True)
class CameraWallStatus:
    state: str
    detail: object = ""


def _put_latest(q, state: str, detail: object = "") -> None:
    item = (str(state), detail)
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def _run_backend(
    window_id: int,
    command_q,
    status_q,
    display_backend: str,
    initial_focus: int = -1,
) -> tuple[int, bool, int]:
    runtime = None
    current_focus = int(initial_focus)
    try:
        xid = int(window_id)
        if xid <= 0:
            raise RuntimeError("invalid Qt native XID")

        os.environ["CAMERA_V2_PASCAL_SAFE"] = "1"
        os.environ["CAMERA_V2_HEATMAP"] = "0"
        os.environ["CAMERA_V2_WALL_WIDTH"] = str(WALL_WIDTH)
        os.environ["CAMERA_V2_WALL_HEIGHT"] = str(WALL_HEIGHT)
        os.environ["CAMERA_V2_DISPLAY_BACKEND"] = display_backend

        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        from .pascal_safe_pipeline import CameraPascalSafeRuntime

        runtime = CameraPascalSafeRuntime()
        if len(runtime.cameras) != CAMERA_COUNT:
            raise RuntimeError(
                f"production wall requires {CAMERA_COUNT} cameras, found {len(runtime.cameras)}"
            )

        def set_focus(source_id: int) -> None:
            nonlocal current_focus
            sid = int(source_id)
            if not 0 <= sid < CAMERA_COUNT:
                sid = -1
            current_focus = sid
            if runtime.tiler.find_property("show-source") is None:
                return
            if sid >= 0:
                runtime.tiler.set_property("rows", 1)
                runtime.tiler.set_property("columns", 1)
                runtime.tiler_rows = 1
                runtime.tiler_columns = 1
                runtime.tiler.set_property("show-source", sid)
                runtime.set_wall_output_geometry(FOCUS_WIDTH, FOCUS_HEIGHT)
                print(f"CAMERA_FOCUS source={sid} mode=fullscreen", flush=True)
            else:
                runtime.tiler.set_property("rows", GRID_ROWS)
                runtime.tiler.set_property("columns", GRID_COLUMNS)
                runtime.tiler_rows = GRID_ROWS
                runtime.tiler_columns = GRID_COLUMNS
                runtime.tiler.set_property("show-source", -1)
                runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)
                print("CAMERA_FOCUS source=-1 mode=grid", flush=True)
            _put_latest(status_q, "FOCUS", {"source": sid})

        runtime.tiler_rows = GRID_ROWS
        runtime.tiler_columns = GRID_COLUMNS
        runtime.tiler.set_property("rows", GRID_ROWS)
        runtime.tiler.set_property("columns", GRID_COLUMNS)
        if runtime.tiler.find_property("show-source") is not None:
            runtime.tiler.set_property("show-source", -1)
        runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)
        if runtime.sink.find_property("force-aspect-ratio") is not None:
            runtime.sink.set_property("force-aspect-ratio", True)
        if current_focus >= 0:
            set_focus(current_focus)

        current_xid = xid
        bound_xid = 0

        def bind_overlay(overlay, target_xid: int | None = None) -> bool:
            nonlocal current_xid, bound_xid
            target = int(current_xid if target_xid is None else target_xid)
            if target <= 0:
                return False
            current_xid = target
            if target == bound_xid:
                return False
            GstVideo.VideoOverlay.set_window_handle(overlay, target)
            try:
                GstVideo.VideoOverlay.handle_events(overlay, False)
            except Exception:
                pass
            bound_xid = target
            print(
                f"CAMERA_VIDEO_BIND xid={target} action=bind backend={display_backend}",
                flush=True,
            )
            return True

        bind_overlay(runtime.sink, current_xid)

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
                _put_latest(
                    status_q,
                    "VIDEO_BOUND",
                    {"backend": display_backend, "xid": current_xid},
                )
                return Gst.BusSyncReply.DROP
            except Exception as exc:
                print(f"CAMERA_VIDEO_BIND_ERROR {type(exc).__name__}: {exc}", flush=True)
                _put_latest(status_q, "ERROR", "Camera display bind failed")
                return Gst.BusSyncReply.PASS

        runtime.bus.set_sync_handler(on_sync_message, None)

        def observe_bus(_bus, message):
            if message.type == Gst.MessageType.STATE_CHANGED and message.src == runtime.pipeline:
                try:
                    _old, new, _pending = message.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        _put_latest(status_q, "LIVE", {"backend": display_backend})
                except Exception:
                    pass
            elif message.type == Gst.MessageType.ERROR:
                try:
                    err, _debug = message.parse_error()
                    source = message.src.get_name() if message.src else "unknown"
                    print(
                        f"CAMERA_PIPELINE_ERROR source={source} backend={display_backend} "
                        f"message={err.message}",
                        flush=True,
                    )
                    _put_latest(
                        status_q,
                        "PIPELINE_WARNING",
                        {"backend": display_backend, "message": err.message},
                    )
                except Exception:
                    pass

        runtime.bus.connect("message", observe_bus)

        def poll_commands() -> bool:
            latest_bind = None
            latest_focus = None
            stop_requested = False
            while True:
                try:
                    command, value = command_q.get_nowait()
                except queue.Empty:
                    break
                if command == "stop":
                    stop_requested = True
                elif command == "bind":
                    try:
                        latest_bind = int(value)
                    except Exception:
                        pass
                elif command == "focus":
                    try:
                        latest_focus = int(value)
                    except Exception:
                        pass

            if latest_bind is not None and latest_bind > 0:
                if bind_overlay(runtime.sink, latest_bind):
                    _put_latest(
                        status_q,
                        "VIDEO_BOUND",
                        {"backend": display_backend, "xid": latest_bind},
                    )
            if latest_focus is not None:
                set_focus(latest_focus)
            if stop_requested:
                runtime.stop()
                return False
            return True

        runtime.GLib.timeout_add(50, poll_commands)
        _put_latest(status_q, "STARTING", {"backend": display_backend})
        rc = runtime.run()
        failover = bool(getattr(runtime, "display_failover_requested", False))
        return int(rc), failover, current_focus
    finally:
        if runtime is not None:
            try:
                runtime.stop()
            except Exception:
                pass
            try:
                runtime.pipeline.set_state(runtime.Gst.State.NULL)
            except Exception:
                pass
        runtime = None
        gc.collect()


def _camera_wall_process(window_id: int, command_q, status_q) -> None:
    try:
        requested = os.environ.get("CAMERA_V2_DISPLAY_BACKEND", "egl").strip().lower()
        backend = requested if requested in {"egl", "x11"} else "egl"
        focus = -1

        rc, failover, focus = _run_backend(
            window_id, command_q, status_q, backend, focus
        )
        if backend == "egl" and failover:
            print("CAMERA_DISPLAY_FAILOVER action=restart backend=x11", flush=True)
            _put_latest(status_q, "DISPLAY_FAILOVER", {"from": "egl", "to": "x11"})
            rc, _, focus = _run_backend(
                window_id, command_q, status_q, "x11", focus
            )
            backend = "x11"

        _put_latest(status_q, "STOPPED", {"backend": backend, "rc": rc})
    except BaseException as exc:
        print(f"CAMERA_RUNTIME_ERROR {type(exc).__name__}: {exc}", flush=True)
        _put_latest(status_q, "ERROR", f"{type(exc).__name__}: {exc}")


class CameraWallController:
    def __init__(self) -> None:
        self.ctx = mp.get_context("spawn")
        self.command_q = self.ctx.Queue(maxsize=12)
        self.status_q = self.ctx.Queue(maxsize=32)
        self.process = None
        self.last_status = CameraWallStatus("STOPPED")
        self._last_requested_xid = 0
        self._focus_source = -1

    def start_or_bind(self, window_id: int) -> None:
        xid = int(window_id)
        if xid <= 0:
            return
        if self.process is not None and self.process.is_alive():
            if xid == self._last_requested_xid:
                return
            self._last_requested_xid = xid
            try:
                self.command_q.put_nowait(("bind", xid))
            except queue.Full:
                pass
            return

        self.stop()
        self._last_requested_xid = xid
        self.process = self.ctx.Process(
            target=_camera_wall_process,
            args=(xid, self.command_q, self.status_q),
            name="sentinel-camera-wall",
            daemon=False,
        )
        self.process.start()
        self.last_status = CameraWallStatus("STARTING")

    def focus(self, source_id: int) -> None:
        sid = int(source_id)
        sid = sid if 0 <= sid < CAMERA_COUNT else -1
        self._focus_source = sid
        if self.process is None or not self.process.is_alive():
            return
        try:
            self.command_q.put_nowait(("focus", sid))
        except queue.Full:
            pass

    def poll(self):
        latest = None
        while True:
            try:
                latest = self.status_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            state, detail = latest
            self.last_status = CameraWallStatus(str(state), detail)
        return self.last_status, {}

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive():
            try:
                self.command_q.put_nowait(("stop", None))
            except queue.Full:
                pass
            process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        try:
            process.close()
        except Exception:
            pass
        self.process = None
        self._last_requested_xid = 0
        self.last_status = CameraWallStatus("STOPPED")
