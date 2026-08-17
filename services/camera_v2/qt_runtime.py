from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("CAMERA_V2_QT_UI", "1")


class CameraQtController:
    """Qt owner for the existing CameraPersonHeatmap pipeline.

    The camera architecture is not duplicated or replaced here. This class only:
      * creates the existing runtime lazily after the Qt native WId exists;
      * attaches nveglglessink to that WId through GstVideoOverlay;
      * starts the same YOLO scheduler/process used by the CLI runtime;
      * exposes realtime metadata snapshots to Qt.
    """

    def __init__(self) -> None:
        self.runtime = None
        self._loop_thread: threading.Thread | None = None
        self._started = False
        self._starting = False
        self._stopped = False
        self._window_handle = 0
        self._focus_source: int | None = None
        self._heat_enabled = [False] * 6
        self._lock = threading.RLock()
        self._status = "WAITING"
        self._error = ""
        self._last_bus_error = ""
        self._started_mono = 0.0
        self._overlay_bound = False
        self._bus_observer_connected = False

        self._heat_lock = threading.RLock()
        self._heat = [[[0.0 for _ in range(32)] for _ in range(18)] for _ in range(6)]
        self._heat_tracks: dict[tuple[int, int], tuple[float, float, float]] = {}
        self._heat_last_decay = time.monotonic()
        self._last_snapshot: dict = {"cameras": [], "tracks": [], "events": [], "rooms": []}

    @property
    def camera_count(self) -> int:
        return 6

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def _set_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._status = status
            self._error = error

    def _prepare_runtime(self):
        if self.runtime is not None:
            return self.runtime
        from .person_heatmap import CameraPersonHeatmap

        runtime = CameraPersonHeatmap()
        if len(runtime.cameras) != 6:
            raise RuntimeError(f"Sentinel Qt expects 6 cameras, found {len(runtime.cameras)}")
        self.runtime = runtime
        return runtime

    def _observe_bus_message(self, _bus, message) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        Gst = runtime.Gst
        if message.type == Gst.MessageType.ERROR:
            try:
                err, debug = message.parse_error()
                src = message.src.get_name() if message.src else "unknown"
                text = f"{src}: {err.message}"
                if debug:
                    text += f" | {debug}"
            except Exception as exc:
                text = f"GStreamer error: {exc}"
            self._last_bus_error = text
            print(f"CAMERA_QT BUS_ERROR {text}", flush=True)
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == runtime.pipeline:
            try:
                old, new, pending = message.parse_state_changed()
                print(
                    f"CAMERA_QT pipeline_state {old.value_nick}->{new.value_nick} "
                    f"pending={pending.value_nick}",
                    flush=True,
                )
            except Exception:
                pass

    def _install_video_overlay_handler(self) -> None:
        runtime = self.runtime
        if runtime is None:
            return

        import gi
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        handle = int(self._window_handle)
        if handle <= 0:
            raise RuntimeError("Qt video window handle is not valid")

        # GStreamer calls this sync handler from the streaming thread exactly when
        # the video sink needs its native window. Do not call any Qt functions here;
        # only use the WId already cached on the Qt GUI thread.
        def on_sync_message(_bus, message, _data=None):
            try:
                is_prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
            except Exception:
                structure = message.get_structure()
                is_prepare = bool(structure and structure.get_name() == "prepare-window-handle")
            if not is_prepare:
                return Gst.BusSyncReply.PASS
            try:
                GstVideo.VideoOverlay.set_window_handle(message.src, handle)
                GstVideo.VideoOverlay.handle_events(message.src, False)
                self._overlay_bound = True
                print(
                    f"CAMERA_QT overlay_bound sink={message.src.get_name()} wid={handle}",
                    flush=True,
                )
                return Gst.BusSyncReply.DROP
            except Exception as exc:
                self._set_status("VIDEO ERROR", str(exc))
                print(f"CAMERA_QT overlay_bind_error {exc}", flush=True)
                return Gst.BusSyncReply.PASS

        runtime.bus.set_sync_handler(on_sync_message, None)

        # Setting the known sink before PLAYING is also valid and avoids an internal
        # fallback window on sinks that do not emit prepare-window-handle on every run.
        GstVideo.VideoOverlay.set_window_handle(runtime.sink, handle)
        try:
            GstVideo.VideoOverlay.handle_events(runtime.sink, False)
        except Exception:
            pass
        self._overlay_bound = True
        print(f"CAMERA_QT overlay_prebound sink={runtime.sink.get_name()} wid={handle}", flush=True)

        if not self._bus_observer_connected:
            runtime.bus.connect("message", self._observe_bus_message)
            self._bus_observer_connected = True

    def start(self, win_id: int) -> None:
        if self._started or self._starting or self._stopped:
            return
        self._starting = True
        self._window_handle = int(win_id)
        self._set_status("STARTING")
        try:
            runtime = self._prepare_runtime()
            self._install_video_overlay_handler()

            from .detection import _yolo_worker

            ctx = mp.get_context("spawn")
            runtime.job_q = ctx.Queue(maxsize=1)
            runtime.result_q = ctx.Queue(maxsize=2)
            runtime.worker = ctx.Process(
                target=_yolo_worker,
                args=(runtime.job_q, runtime.result_q),
                daemon=True,
            )
            runtime.worker.start()
            runtime.scheduler_thread = threading.Thread(
                target=runtime._scheduler,
                name="camera-v2-yolo-scheduler",
                daemon=True,
            )
            runtime.scheduler_thread.start()

            result = runtime.pipeline.set_state(runtime.Gst.State.PLAYING)
            if result == runtime.Gst.StateChangeReturn.FAILURE:
                runtime.pipeline.set_state(runtime.Gst.State.NULL)
                self._stop_detector_sidecar()
                raise RuntimeError("Camera V2 pipeline failed to enter PLAYING")

            self._loop_thread = threading.Thread(
                target=runtime.loop.run,
                name="camera-v2-glib-loop",
                daemon=True,
            )
            self._loop_thread.start()
            self._started = True
            self._started_mono = time.monotonic()
            self._set_status("CONNECTING")
            print(
                "CAMERA_QT started: Sentinel UI + EXISTING Camera V2 pipeline; "
                "6xRTSP/NVDEC->mux->YOLO26m/NvDCF->2x3 tiler->OSD->EGL; "
                "video_copy=0 mjpeg=0 architecture_changed=0",
                flush=True,
            )
        except Exception as exc:
            self._set_status("ERROR", f"{type(exc).__name__}: {exc}")
            print(f"CAMERA_QT START ERROR: {type(exc).__name__}: {exc}", flush=True)
            try:
                if self.runtime is not None:
                    self.runtime.pipeline.set_state(self.runtime.Gst.State.NULL)
            except Exception:
                pass
            self._stop_detector_sidecar()
        finally:
            self._starting = False

    def _stop_detector_sidecar(self) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        try:
            runtime.det_stop.set()
            runtime._clear_requests()
        except Exception:
            pass
        try:
            runtime.mailbox.close()
        except Exception:
            pass
        if getattr(runtime, "job_q", None) is not None:
            try:
                runtime.job_q.put_nowait(None)
            except Exception:
                pass
        thread = getattr(runtime, "scheduler_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        worker = getattr(runtime, "worker", None)
        if worker is not None:
            worker.join(timeout=3.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        runtime = self.runtime
        if runtime is None:
            return
        try:
            runtime.stop()
        except Exception:
            pass
        if self._loop_thread is not None and self._loop_thread is not threading.current_thread():
            self._loop_thread.join(timeout=2.0)
        try:
            runtime.pipeline.set_state(runtime.Gst.State.NULL)
        except Exception:
            pass
        self._stop_detector_sidecar()
        for tee, pad in getattr(runtime, "tee_request_pads", []):
            try:
                tee.release_request_pad(pad)
            except Exception:
                pass
        for pad in getattr(runtime, "_request_pads", []):
            try:
                runtime.mux.release_request_pad(pad)
            except Exception:
                pass
        self._set_status("STOPPED")

    def set_focus_source(self, source_id: int | None) -> None:
        self._focus_source = source_id
        runtime = self.runtime
        if runtime is None:
            return
        runtime.tiler.set_property("show-source", -1 if source_id is None else int(source_id))
        try:
            import gi
            gi.require_version("GstVideo", "1.0")
            from gi.repository import GstVideo
            GstVideo.VideoOverlay.expose(runtime.sink)
        except Exception:
            pass

    def focus_source(self) -> int | None:
        return self._focus_source

    def set_heatmap_enabled(self, source_id: int, enabled: bool) -> None:
        source_id = int(source_id)
        if 0 <= source_id < len(self._heat_enabled):
            self._heat_enabled[source_id] = bool(enabled)

    def heatmap_enabled(self, source_id: int) -> bool:
        source_id = int(source_id)
        return 0 <= source_id < len(self._heat_enabled) and self._heat_enabled[source_id]

    def _decay_heat(self, now: float) -> None:
        dt = max(0.0, now - self._heat_last_decay)
        if dt < 0.05:
            return
        factor = math.pow(0.9975, dt * 10.0)
        with self._heat_lock:
            for source in self._heat:
                for row in source:
                    for x in range(len(row)):
                        row[x] *= factor
                        if row[x] < 0.004:
                            row[x] = 0.0
        self._heat_last_decay = now

    def _deposit(self, source_id: int, gx: float, gy: float, amount: float) -> None:
        if not 0 <= source_id < len(self._heat):
            return
        cx = max(0, min(31, int(round(gx))))
        cy = max(0, min(17, int(round(gy))))
        kernel = (
            (0.08, 0.18, 0.08),
            (0.18, 1.0, 0.18),
            (0.08, 0.18, 0.08),
        )
        grid = self._heat[source_id]
        for ky in range(-1, 2):
            y = cy + ky
            if not 0 <= y < 18:
                continue
            for kx in range(-1, 2):
                x = cx + kx
                if 0 <= x < 32:
                    grid[y][x] = min(
                        1.0,
                        grid[y][x] + amount * kernel[ky + 1][kx + 1],
                    )

    def _update_heat(self, snapshot: dict) -> None:
        now = time.monotonic()
        self._decay_heat(now)
        runtime = self.runtime
        frame_w = max(1.0, float(getattr(runtime, "frame_width", 1280)))
        frame_h = max(1.0, float(getattr(runtime, "frame_height", 720)))
        active_keys: set[tuple[int, int]] = set()
        with self._heat_lock:
            for track in snapshot.get("tracks", []):
                sid = int(track.get("source_id", -1))
                oid = int(track.get("object_id", -1))
                if not 0 <= sid < 6 or oid < 0:
                    continue
                key = (sid, oid)
                active_keys.add(key)
                foot_x = float(track.get("left", 0.0)) + float(track.get("width", 0.0)) * 0.5
                foot_y = float(track.get("top", 0.0)) + float(track.get("height", 0.0)) * 0.98
                gx = max(0.0, min(31.0, foot_x / frame_w * 31.0))
                gy = max(0.0, min(17.0, foot_y / frame_h * 17.0))
                previous = self._heat_tracks.get(key)
                self._heat_tracks[key] = (gx, gy, now)
                if previous is None:
                    continue
                dx, dy = gx - previous[0], gy - previous[1]
                dist = math.hypot(dx, dy)
                if 0.22 <= dist <= 4.5:
                    steps = max(1, min(10, int(dist * 2.2)))
                    amount = min(0.020, 0.0035 + dist * 0.003)
                    for step in range(1, steps + 1):
                        t = step / steps
                        self._deposit(
                            sid,
                            previous[0] + dx * t,
                            previous[1] + dy * t,
                            amount,
                        )
            for key in list(self._heat_tracks):
                if key not in active_keys and now - self._heat_tracks[key][2] > 1.5:
                    self._heat_tracks.pop(key, None)

    def snapshot(self) -> dict:
        runtime = self.runtime
        if runtime is None:
            cameras = [
                {
                    "source_id": i,
                    "camera_id": f"CAM-{i + 1:02d}",
                    "fps": 0.0,
                    "online": False,
                    "count": 0,
                }
                for i in range(6)
            ]
            return {"cameras": cameras, "tracks": [], "events": [], "rooms": []}

        snap = runtime.ui_snapshot()
        self._update_heat(snap)
        self._last_snapshot = snap

        online = sum(1 for camera in snap.get("cameras", []) if camera.get("online"))
        if self._started:
            if online == 6:
                self._set_status("LIVE")
            elif online > 0:
                self._set_status(f"DEGRADED · {online}/6", self._last_bus_error)
            elif time.monotonic() - self._started_mono > 7.0:
                detail = self._last_bus_error or "No RTSP/NVDEC frames received yet"
                self._set_status("NO VIDEO", detail)

        if online and self._overlay_bound:
            try:
                import gi
                gi.require_version("GstVideo", "1.0")
                from gi.repository import GstVideo
                GstVideo.VideoOverlay.expose(runtime.sink)
            except Exception:
                pass
        return snap

    def heat_points(
        self,
        source_id: int,
        max_points: int = 30,
    ) -> list[tuple[float, float, float]]:
        if not self.heatmap_enabled(source_id):
            return []
        with self._heat_lock:
            candidates = []
            for y, row in enumerate(self._heat[int(source_id)]):
                for x, value in enumerate(row):
                    if value >= 0.012:
                        candidates.append((value, x + 0.5, y + 0.5))
        candidates.sort(reverse=True)
        return [
            (x / 32.0, y / 18.0, value)
            for value, x, y in candidates[:max_points]
        ]

    def export_events_csv(self, path: str | Path) -> None:
        events = self.snapshot().get("events", [])
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "type", "camera_id", "label", "message"])
            for event in events:
                writer.writerow(
                    [
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(event.get("time", 0)),
                        ),
                        event.get("type", ""),
                        event.get("camera_id", ""),
                        event.get("label", ""),
                        event.get("message", ""),
                    ]
                )
