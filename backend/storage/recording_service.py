import os
import time
import queue
import threading
from datetime import datetime

import cv2

from PySide6.QtCore import QObject, Signal

from backend.core.logger import get_logger

log = get_logger("storage.recording")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RecordingService(QObject):
    """
    Thread-safe video recording service.

    write_frame() faqat queue ga qo'yadi.
    Yozish alohida threadda bajariladi.
    """

    recording_started = Signal(str, str)
    recording_stopped = Signal(str)
    message = Signal(str, str)

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.enabled = bool(config.get("storage.recordings_enabled", False))

        recordings_dir = config.get("storage.recordings_dir", "recordings")

        if not os.path.isabs(recordings_dir):
            recordings_dir = os.path.join(BASE_DIR, recordings_dir)

        self.dir = recordings_dir
        os.makedirs(self.dir, exist_ok=True)

        self.format = str(config.get("storage.recording_format", "mp4")).lower()
        self.segment_minutes = int(config.get("storage.segment_minutes", 5))

        self.writers = {}
        self.locks = {}
        self.camera_enabled = {}

        self.frame_queue = queue.Queue(maxsize=300)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        log.info("RecordingService started: %s", self.dir)

    # ---------------- controls ----------------
    def set_global_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

        if not self.enabled:
            self.stop_all()

    def set_camera_recording(self, camera_id: str, enabled: bool):
        self.camera_enabled[camera_id] = bool(enabled)

        if not enabled:
            self.stop_recording(camera_id)

    def is_recording(self, camera_id: str) -> bool:
        return camera_id in self.writers

    # ---------------- paths ----------------
    def _camera_dir(self, camera_id: str) -> str:
        safe = "".join(c for c in str(camera_id) if c.isalnum() or c in ("-", "_"))
        path = os.path.join(self.dir, safe)
        os.makedirs(path, exist_ok=True)
        return path

    def _new_path(self, camera_id: str) -> str:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{ts}.{self.format}"
        return os.path.join(self._camera_dir(camera_id), filename)

    # ---------------- queue input ----------------
    def write_frame(self, camera_id: str, bgr):
        if not self.enabled:
            return

        if not self.camera_enabled.get(camera_id, False):
            return

        if bgr is None:
            return

        try:
            self.frame_queue.put_nowait((camera_id, bgr))
        except queue.Full:
            pass

    # ---------------- writer thread ----------------
    def _run(self):
        while self._running:
            try:
                camera_id, bgr = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_frame(camera_id, bgr)
            except Exception as e:
                log.error("recording process frame error: %s", e)

    def _process_frame(self, camera_id: str, bgr):
        info = self.writers.get(camera_id)

        if info is None:
            h, w = bgr.shape[:2]
            self.start_recording(camera_id, fps=25, size=(w, h))
            info = self.writers.get(camera_id)

            if info is None:
                return

        lock = self.locks.get(camera_id)

        if lock is None:
            return

        with lock:
            elapsed = time.time() - info["start_time"]

            if elapsed >= self.segment_minutes * 60:
                self._rotate(camera_id, bgr)
                return

            h, w = bgr.shape[:2]

            if (w, h) != info["size"]:
                bgr = cv2.resize(bgr, info["size"])

            info["writer"].write(bgr)

    # ---------------- start / stop ----------------
    def start_recording(self, camera_id: str, fps: int = 25, size=(640, 360)):
        if not self.enabled:
            return False

        if not self.camera_enabled.get(camera_id, False):
            return False

        if camera_id in self.writers:
            return True

        try:
            path = self._new_path(camera_id)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, float(fps), size)

            if not writer.isOpened():
                log.error("Cannot open video writer: %s", path)
                self.message.emit(f"Recording failed: {camera_id}", "error")
                return False

            self.locks[camera_id] = threading.Lock()

            self.writers[camera_id] = {
                "writer": writer,
                "path": path,
                "start_time": time.time(),
                "fps": fps,
                "size": size,
            }

            self.recording_started.emit(camera_id, path)
            log.info("Recording started: %s -> %s", camera_id, path)

            return True

        except Exception as e:
            log.error("start_recording error: %s", e)
            return False

    def stop_recording(self, camera_id: str):
        info = self.writers.pop(camera_id, None)

        if info is None:
            return

        lock = self.locks.pop(camera_id, None)

        try:
            if lock is not None:
                with lock:
                    info["writer"].release()
            else:
                info["writer"].release()

        except Exception as e:
            log.error("stop_recording error: %s", e)

        self.recording_stopped.emit(camera_id)
        log.info("Recording stopped: %s", camera_id)

    def stop_all(self):
        for camera_id in list(self.writers.keys()):
            self.stop_recording(camera_id)

    def _rotate(self, camera_id: str, bgr):
        try:
            info = self.writers.get(camera_id)

            if info is None:
                return

            info["writer"].release()

            path = self._new_path(camera_id)
            fps = info.get("fps", 25)
            size = info.get("size", (640, 360))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, float(fps), size)

            if not writer.isOpened():
                log.error("Rotate writer failed: %s", path)
                self.writers.pop(camera_id, None)
                return

            self.writers[camera_id] = {
                "writer": writer,
                "path": path,
                "start_time": time.time(),
                "fps": fps,
                "size": size,
            }

            writer.write(bgr)

            self.recording_started.emit(camera_id, path)
            log.info("Recording rotated: %s -> %s", camera_id, path)

        except Exception as e:
            log.error("_rotate error: %s", e)

    # ---------------- shutdown ----------------
    def shutdown(self):
        self._running = False

        try:
            self._thread.join(timeout=3.0)
        except Exception:
            pass

        self.stop_all()
        log.info("RecordingService stopped")