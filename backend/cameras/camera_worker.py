import os
import time
import cv2

from PySide6.QtCore import QThread, Signal

from backend.cameras.utils import build_source_url, is_int_source
from backend.cameras.frame_buffer import FrameBuffer
from backend.cameras.camera_health import CameraHealth
from backend.core.logger import get_logger

log = get_logger("camera.worker")


class CameraWorker(QThread):
    """
    Har kamera uchun alohida thread.

    - RTSP / USB / Laptop kamerani ochadi
    - Frame o'qiydi
    - FrameBuffer ga qo'yadi
    - Auto reconnect qiladi
    - FPS / latency / packet loss hisoblaydi
    """

    status_changed = Signal(str, bool)       # camera_id, online
    frame_captured = Signal(str)             # camera_id
    health_updated = Signal(str, dict)  
    frame_bgr_ready = Signal(str, object)   # camera_id, BGR frame     # camera_id, metrics

    def __init__(self, cam_cfg: dict, target_size=(640, 360)):
        super().__init__()

        self.cam_id = cam_cfg.get("id", "CAM-XX")
        self.cfg = cam_cfg
        self.target_size = target_size

        self.target_fps = int(cam_cfg.get("fps", 25) or 25)
        self.reconnect_interval = int(cam_cfg.get("reconnect_interval", 10) or 10)
        self.connection_timeout = int(cam_cfg.get("connection_timeout", 5) or 5)

        self.fail_limit = max(5, self.connection_timeout * 5)

        self.buffer = FrameBuffer()
        self.health = CameraHealth()

        self._running = False

    def stop(self):
        self._running = False

    def _open_capture(self):
        src = build_source_url(
            self.cfg.get("source"),
            self.cfg.get("username"),
            self.cfg.get("password"),
        )

        if is_int_source(src):
            api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(src), api)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)

            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.connection_timeout * 1000))
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.connection_timeout * 1000))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        return cap

    def _wait_reconnect(self):
        end = time.time() + self.reconnect_interval

        while self._running and time.time() < end:
            time.sleep(0.5)

    def run(self):
        self._running = True
        log.info("CameraWorker started: %s", self.cam_id)

        while self._running:
            cap = self._open_capture()

            if cap is None or not cap.isOpened():
                log.warning("Camera cannot open: %s", self.cam_id)

                self.health.online = False
                self.status_changed.emit(self.cam_id, False)

                if cap is not None:
                    cap.release()

                self._wait_reconnect()
                continue

            self.health.online = True
            self.status_changed.emit(self.cam_id, True)
            log.info("Camera connected: %s", self.cam_id)

            fail = 0
            frames = 0
            last_fps_time = time.time()

            while self._running:
                loop_t = time.time()
                ret, frame = cap.read()
                latency = (time.time() - loop_t) * 1000.0

                self.health.record_read(bool(ret), latency)

                if not ret or frame is None:
                    fail += 1

                    if fail >= self.fail_limit:
                        log.warning("Camera read failed: %s (%s)", self.cam_id, fail)
                        break

                    time.sleep(0.1)
                    continue

                fail = 0

                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)

                self.buffer.put(frame)
                self.frame_bgr_ready.emit(self.cam_id, frame)
                self.frame_captured.emit(self.cam_id)

                frames += 1
                now = time.time()

                if now - last_fps_time >= 1.0:
                    self.health.set_fps(frames / max(1e-6, now - last_fps_time))
                    frames = 0
                    last_fps_time = now
                    self.health_updated.emit(self.cam_id, self.health.metrics())

                elapsed = time.time() - loop_t
                interval = 1.0 / max(1, self.target_fps)

                if elapsed < interval:
                    time.sleep(interval - elapsed)

            self.health.online = False
            self.status_changed.emit(self.cam_id, False)

            cap.release()

            if self._running:
                self._wait_reconnect()

        log.info("CameraWorker stopped: %s", self.cam_id)