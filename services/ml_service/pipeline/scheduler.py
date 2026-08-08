"""Thread-safe, event-driven fresh-frame scheduler with no inference code."""
import threading
import time
from .batch import BatchOutput

class BatchScheduler:
    def __init__(self, max_frame_age_ms=250, on_batch=None, metrics_interval_sec=2,
                 starved_after_ms=500, batch_collect_window_ms=5):
        self.max_frame_age_ms = max(1.0, float(max_frame_age_ms))
        self.starved_after_ms = max(1.0, float(starved_after_ms))
        self.metrics_interval = max(.25, float(metrics_interval_sec))
        self.collect_window = max(0.0, float(batch_collect_window_ms)) / 1000
        self._on_batch, self._condition = on_batch, threading.Condition(threading.RLock())
        self._buffers, self._states, self._last_dropped = {}, {}, {}
        self._thread, self._running, self._last_batch = None, False, None
        self._batch_id = self._stale_drops = 0
        self._window_at = time.monotonic()
        self._window_batches = self._window_processed = self._window_stale = 0

    def notify_frame_available(self):
        with self._condition: self._condition.notify()

    def register_camera(self, camera_id, buffer):
        with self._condition:
            self._buffers[camera_id] = buffer
            self._states.setdefault(camera_id, {"used_frame_id": None, "frame_age_ms": 0.0,
                "duplicate_count": 0, "stale_drops": 0, "starved_count": 0,
                "starved": False, "last_seen": 0.0})
            self._last_dropped.setdefault(camera_id, 0); self._condition.notify()

    def unregister_camera(self, camera_id):
        with self._condition:
            self._buffers.pop(camera_id, None); self._states.pop(camera_id, None)
            self._last_dropped.pop(camera_id, None); self._condition.notify()

    def start(self):
        with self._condition:
            if self._thread and self._thread.is_alive(): return
            self._running = True
            self._thread = threading.Thread(target=self.run, name="batch-scheduler", daemon=False)
            self._thread.start()

    def run(self):
        while True:
            with self._condition:
                if not self._running: break
                if not any(len(buffer) for buffer in self._buffers.values()):
                    self._condition.wait(timeout=min(self.metrics_interval, self.starved_after_ms / 1000)); continue
                deadline = time.monotonic() + self.collect_window
                while self._running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: break
                    self._condition.wait(timeout=remaining)
                if not self._running: break
                batch = self._collect_locked()
            if batch is not None and self._on_batch is not None: self._on_batch(batch)

    def _collect_locked(self):
        now_wall, now_mono, packets = time.time(), time.monotonic(), []
        for camera_id, buffer in tuple(self._buffers.items()):
            packet = buffer.take()
            if packet is None: continue
            state = self._states[camera_id]
            age = max(0.0, (now_wall - packet.receive_timestamp) * 1000)
            state["last_seen"], state["frame_age_ms"], state["starved"] = now_mono, age, False
            if state["used_frame_id"] is not None and packet.frame_id <= state["used_frame_id"]:
                state["duplicate_count"] += 1; continue
            if age > self.max_frame_age_ms:
                state["stale_drops"] += 1; self._stale_drops += 1; self._window_stale += 1; continue
            state["used_frame_id"] = packet.frame_id; packets.append(packet)
        if not packets: return None
        self._batch_id += 1; self._window_batches += 1; self._window_processed += len(packets)
        self._last_batch = BatchOutput(self._batch_id, now_wall, tuple(packets))
        return self._last_batch

    def snapshot_metrics(self, reader_metrics=None):
        now, reader_metrics = time.monotonic(), reader_metrics or {}
        elapsed = max(1e-6, now - self._window_at)
        with self._condition:
            cameras, dropped = {}, self._window_stale
            for camera_id, state in self._states.items():
                source = dict(reader_metrics.get(camera_id, {}))
                starved = not source.get("online", False) or bool(state["last_seen"] and now - state["last_seen"] > self.starved_after_ms / 1000)
                if starved and not state["starved"]: state["starved_count"] += 1; state["starved"] = True
                for key in ("used_frame_id", "frame_age_ms", "duplicate_count", "starved_count"): source[key] = state[key]
                source["stale_drops"], source["is_starved"] = state["stale_drops"], starved
                for key, default in (("recv_frame_id", 0), ("source_fps", 0.0), ("interarrival_ms", 0.0), ("max_interarrival_ms", 0.0), ("dropped_old", 0)): source.setdefault(key, default)
                current = source["dropped_old"]; dropped += max(0, current - self._last_dropped.get(camera_id, 0))
                self._last_dropped[camera_id] = current; cameras[camera_id] = source
            output = {"cameras": cameras, "batch_size": len(self._last_batch.frames) if self._last_batch else 0,
                "batch_cameras": list(self._last_batch.camera_ids) if self._last_batch else [],
                "batch_rate": self._window_batches / elapsed, "stale_drops": self._stale_drops,
                "source_total_fps": sum(c["source_fps"] for c in cameras.values()),
                "processed_total_fps": self._window_processed / elapsed, "dropped_total_fps": dropped / elapsed}
            self._window_at = now; self._window_batches = self._window_processed = self._window_stale = 0
            return output

    def stop(self):
        with self._condition: self._running = False; self._condition.notify_all()
    def join(self, timeout=None):
        if self._thread: self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()
