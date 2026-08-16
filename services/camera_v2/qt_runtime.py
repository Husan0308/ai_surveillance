from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("CAMERA_V2_QT_UI", "1")

from .detection import _yolo_worker
from .person_heatmap import CameraPersonHeatmap


class CameraQtController:
    """Non-blocking owner for CameraPersonHeatmap used by the Qt event loop."""

    def __init__(self) -> None:
        self.runtime = CameraPersonHeatmap()
        self._loop_thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._focus_source: int | None = None
        self._heat_enabled = [False] * len(self.runtime.cameras)

        # Tiny UI-side movement map: no frame copies, only tracker coordinates.
        self._heat_lock = threading.RLock()
        self._heat = [[[0.0 for _ in range(32)] for _ in range(18)] for _ in self.runtime.cameras]
        self._heat_tracks: dict[tuple[int, int], tuple[float, float, float]] = {}
        self._heat_last_decay = time.monotonic()
        self._last_snapshot: dict = {"cameras": [], "tracks": [], "events": [], "rooms": []}

    @property
    def camera_count(self) -> int:
        return len(self.runtime.cameras)

    def bind_window(self, win_id: int) -> None:
        import gi
        gi.require_version("GstVideo", "1.0")
        from gi.repository import GstVideo
        GstVideo.VideoOverlay.set_window_handle(self.runtime.sink, int(win_id))
        try:
            GstVideo.VideoOverlay.handle_events(self.runtime.sink, False)
        except Exception:
            pass

    def start(self) -> None:
        if self._started:
            return
        ctx = mp.get_context("spawn")
        self.runtime.job_q = ctx.Queue(maxsize=1)
        self.runtime.result_q = ctx.Queue(maxsize=2)
        self.runtime.worker = ctx.Process(
            target=_yolo_worker,
            args=(self.runtime.job_q, self.runtime.result_q),
            daemon=True,
        )
        self.runtime.worker.start()
        self.runtime.scheduler_thread = threading.Thread(
            target=self.runtime._scheduler,
            name="camera-v2-yolo-scheduler",
            daemon=True,
        )
        self.runtime.scheduler_thread.start()

        result = self.runtime.pipeline.set_state(self.runtime.Gst.State.PLAYING)
        if result == self.runtime.Gst.StateChangeReturn.FAILURE:
            self.runtime.pipeline.set_state(self.runtime.Gst.State.NULL)
            self._stop_detector_sidecar()
            raise RuntimeError("Camera V2 Qt pipeline failed to enter PLAYING")

        self._loop_thread = threading.Thread(
            target=self.runtime.loop.run,
            name="camera-v2-glib-loop",
            daemon=True,
        )
        self._loop_thread.start()
        self._started = True
        print("CAMERA_QT started: Sentinel UI + native DeepStream wall + YOLO26m + NvDCF", flush=True)

    def _stop_detector_sidecar(self) -> None:
        self.runtime.det_stop.set()
        self.runtime._clear_requests()
        try:
            self.runtime.mailbox.close()
        except Exception:
            pass
        if self.runtime.job_q is not None:
            try:
                self.runtime.job_q.put_nowait(None)
            except Exception:
                pass
        if self.runtime.scheduler_thread is not None:
            self.runtime.scheduler_thread.join(timeout=2.0)
        if self.runtime.worker is not None:
            self.runtime.worker.join(timeout=3.0)
            if self.runtime.worker.is_alive():
                self.runtime.worker.terminate()
                self.runtime.worker.join(timeout=1.0)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.runtime.stop()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        try:
            self.runtime.pipeline.set_state(self.runtime.Gst.State.NULL)
        except Exception:
            pass
        self._stop_detector_sidecar()
        for tee, pad in getattr(self.runtime, "tee_request_pads", []):
            try:
                tee.release_request_pad(pad)
            except Exception:
                pass
        for pad in getattr(self.runtime, "_request_pads", []):
            try:
                self.runtime.mux.release_request_pad(pad)
            except Exception:
                pass

    def set_focus_source(self, source_id: int | None) -> None:
        value = -1 if source_id is None else int(source_id)
        self.runtime.tiler.set_property("show-source", value)
        self._focus_source = source_id

    def focus_source(self) -> int | None:
        return self._focus_source

    def set_heatmap_enabled(self, source_id: int, enabled: bool) -> None:
        self._heat_enabled[int(source_id)] = bool(enabled)

    def heatmap_enabled(self, source_id: int) -> bool:
        return self._heat_enabled[int(source_id)]

    def _decay_heat(self, now: float) -> None:
        dt = max(0.0, now - self._heat_last_decay)
        if dt < 0.05:
            return
        factor = math.pow(0.985, dt * 10.0)
        with self._heat_lock:
            for source in self._heat:
                for row in source:
                    for x in range(len(row)):
                        row[x] *= factor
                        if row[x] < 0.01:
                            row[x] = 0.0
        self._heat_last_decay = now

    def _deposit(self, source_id: int, gx: float, gy: float, amount: float) -> None:
        cx = max(0, min(31, int(round(gx))))
        cy = max(0, min(17, int(round(gy))))
        kernel = ((0.12, 0.25, 0.12), (0.25, 1.0, 0.25), (0.12, 0.25, 0.12))
        grid = self._heat[source_id]
        for ky in range(-1, 2):
            y = cy + ky
            if not 0 <= y < 18:
                continue
            for kx in range(-1, 2):
                x = cx + kx
                if 0 <= x < 32:
                    grid[y][x] = min(1.0, grid[y][x] + amount * kernel[ky + 1][kx + 1])

    def _update_heat(self, snapshot: dict) -> None:
        now = time.monotonic()
        self._decay_heat(now)
        active_keys: set[tuple[int, int]] = set()
        with self._heat_lock:
            for track in snapshot.get("tracks", []):
                sid = int(track["source_id"])
                oid = int(track["object_id"])
                key = (sid, oid)
                active_keys.add(key)
                frame_w = max(1.0, float(self.runtime.frame_width))
                frame_h = max(1.0, float(self.runtime.frame_height))
                foot_x = float(track["left"]) + float(track["width"]) * 0.5
                foot_y = float(track["top"]) + float(track["height"]) * 0.98
                gx = max(0.0, min(31.0, foot_x / frame_w * 31.0))
                gy = max(0.0, min(17.0, foot_y / frame_h * 17.0))
                previous = self._heat_tracks.get(key)
                self._heat_tracks[key] = (gx, gy, now)
                if previous is None:
                    continue
                dx, dy = gx - previous[0], gy - previous[1]
                dist = math.hypot(dx, dy)
                if 0.16 <= dist <= 4.5:
                    steps = max(1, min(8, int(dist * 2.0)))
                    for step in range(1, steps + 1):
                        t = step / steps
                        self._deposit(sid, previous[0] + dx * t, previous[1] + dy * t, min(0.08, 0.018 + dist * 0.012))
            for key in list(self._heat_tracks):
                if key not in active_keys and now - self._heat_tracks[key][2] > 1.5:
                    self._heat_tracks.pop(key, None)

    def snapshot(self) -> dict:
        snap = self.runtime.ui_snapshot()
        self._update_heat(snap)
        self._last_snapshot = snap
        return snap

    def heat_points(self, source_id: int, max_points: int = 36) -> list[tuple[float, float, float]]:
        if not self.heatmap_enabled(source_id):
            return []
        with self._heat_lock:
            candidates = []
            for y, row in enumerate(self._heat[int(source_id)]):
                for x, value in enumerate(row):
                    if value >= 0.025:
                        candidates.append((value, x + 0.5, y + 0.5))
        candidates.sort(reverse=True)
        return [(x / 32.0, y / 18.0, value) for value, x, y in candidates[:max_points]]

    def export_events_csv(self, path: str | Path) -> None:
        events = self.snapshot().get("events", [])
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "type", "camera_id", "label", "message"])
            for event in events:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.get("time", 0))),
                    event.get("type", ""), event.get("camera_id", ""),
                    event.get("label", ""), event.get("message", ""),
                ])
