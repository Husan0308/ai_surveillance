import os
import time
import threading
import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal

from backend.cameras.utils import build_source_url, is_int_source
from backend.cameras.frame_buffer import FrameBuffer
from backend.cameras.camera_health import CameraHealth
from backend.core.logger import get_logger

log = get_logger("camera.worker")

_GST_INIT_LOCK = threading.Lock()
_GST = None

def _load_gstreamer():
    """Initialize GI/GStreamer once for all concurrent camera threads."""
    global _GST
    with _GST_INIT_LOCK:
        if _GST is None:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)
            _GST = Gst
    return _GST

def _configure_project_gstreamer_runtime():
    """Expose project-local plugins before GStreamer scans its registry."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lib_dir = os.path.join(root, ".runtime", "gstreamer", "usr", "lib", "x86_64-linux-gnu")
    plugin_dir = os.path.join(lib_dir, "gstreamer-1.0")
    if os.path.isdir(plugin_dir):
        current_plugins = os.environ.get("GST_PLUGIN_PATH", "")
        os.environ["GST_PLUGIN_PATH"] = plugin_dir + (os.pathsep + current_plugins if current_plugins else "")
        current_libs = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + current_libs if current_libs else "")


class GstVideoCapture:
    """Small cv2.VideoCapture-compatible wrapper around a GStreamer appsink."""

    def __init__(self, description, timeout_sec=5):
        self._opened = False
        self._pipeline = None
        self._sink = None
        self._bus = None
        self._timeout_sec = max(1, int(timeout_sec))
        try:
            Gst = _load_gstreamer()
            self.Gst = Gst
            self._pipeline = Gst.parse_launch(description)
            self._sink = self._pipeline.get_by_name("framesink")
            self._bus = self._pipeline.get_bus()
            state = self._pipeline.set_state(Gst.State.PLAYING)
            self._opened = state != Gst.StateChangeReturn.FAILURE and self._sink is not None
        except Exception as exc:
            log.error("GStreamer capture init failed: %s", exc)
            self.release()

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        msg = self._bus.timed_pop_filtered(0, self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if msg is not None:
            if msg.type == self.Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                log.error("GStreamer stream error: %s (%s)", err, debug or "")
            self.release()
            return False, None
        # Flush stale queued frames and pull ONLY the latest frame
        sample = None
        while True:
            s = self._sink.emit("try-pull-sample", 0)
            if s is None:
                break
            sample = s
        if sample is None:
            sample = self._sink.emit("try-pull-sample", self._timeout_sec * self.Gst.SECOND)
        if sample is None:
            return False, None
        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        buf = sample.get_buffer()
        ok, mapped = buf.map(self.Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            arr = np.frombuffer(mapped.data, dtype=np.uint8).reshape(height, width, 3)
            frame = np.ascontiguousarray(arr)
        finally:
            buf.unmap(mapped)
        return True, frame

    def set(self, *_args):
        return False

    def get(self, *_args):
        return 0.0

    def release(self):
        self._opened = False
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self.Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = self._sink = self._bus = None




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

    def __init__(self, cam_cfg: dict, target_size=(1280, 720), use_deepstream=False, use_gstreamer=False):
        super().__init__()
        self.use_deepstream = bool(use_deepstream)
        self.use_gstreamer = bool(use_gstreamer)

        self.cam_id = cam_cfg.get("id", "CAM-XX")
        self.cfg = cam_cfg

        # Target resolution — parse from config (may be string "1280x720", list or tuple)
        _raw_size = self.cfg.get("target_size", target_size)
        if isinstance(_raw_size, str) and "x" in _raw_size:
            _w, _h = _raw_size.lower().split("x")
            self.target_size = (int(_w), int(_h))
        elif isinstance(_raw_size, (list, tuple)) and len(_raw_size) == 2:
            self.target_size = (int(_raw_size[0]), int(_raw_size[1]))
        else:
            self.target_size = target_size  # fallback to constructor arg

        # Target FPS — default 25 for smooth RTSP video playback
        _cfg_fps = int(self.cfg.get("max_fps", self.cfg.get("fps", 25) or 25))
        self.target_fps = min(_cfg_fps, 30)

        self.reconnect_interval = int(self.cfg.get("reconnect_interval", 10) or 10)
        self.connection_timeout = int(self.cfg.get("connection_timeout", 5) or 5)
        self.fail_limit = max(5, self.connection_timeout * 5)

        self.buffer = FrameBuffer()
        self.health = CameraHealth()

        self._running = False

    def stop(self):
        self._running = False

    def _open_deepstream_capture(self, src):
        """NVIDIA NVDEC hardware decode — zero CPU decoding, ultra-low latency."""
        _configure_project_gstreamer_runtime()
        if is_int_source(src):
            source = f"v4l2src device=/dev/video{int(src)}"
        else:
            escaped = str(src).replace("\\", "\\\\").replace('"', '\\"')
            latency = int(self.cfg.get("latency_ms", 20) or 20)
            codec = str(self.cfg.get("codec", "h264")).lower()
            if codec in ("h265", "hevc"):
                depay = "rtph265depay"
            else:
                depay = "rtph264depay"
            source = (
                f'rtspsrc location="{escaped}" latency={latency} drop-on-latency=true '
                f'buffer-mode=slave protocols=tcp+udp '
                f'! {depay} '
                f'! nvv4l2decoder low-latency-mode=true drop-frame-interval=0 num-extra-surfaces=1'
            )
        out_w, out_h = self.target_size if self.target_size else (1280, 720)
        pipeline = (
            f"{source} ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            f"nvvideoconvert "
            f"! video/x-raw,format=BGRx,width={out_w},height={out_h} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=framesink sync=false max-buffers=1 drop=true"
        )
        log.info("DeepStream pipeline for %s: %dx%d via NVDEC", self.cam_id, out_w, out_h)
        return GstVideoCapture(pipeline, timeout_sec=self.connection_timeout)

    def _open_software_gstreamer_capture(self, src):
        """Low-latency GStreamer decode with a dropping appsink."""
        _configure_project_gstreamer_runtime()
        width, height = self.target_size or (1280, 720)
        if is_int_source(src):
            source = f"v4l2src device=/dev/video{int(src)}"
        else:
            escaped = str(src).replace("\\", "\\\\").replace('"', '\\"')
            latency = int(self.cfg.get("latency_ms", 20) or 20)
            codec = str(self.cfg.get("codec", "h265")).lower()
            if codec in ("h264", "avc"):
                depay, parser, decoder = "rtph264depay", "h264parse", "avdec_h264"
            else:
                depay, parser, decoder = "rtph265depay", "h265parse", "avdec_h265"
            source = (
                f'rtspsrc location="{escaped}" latency={latency} protocols=tcp+udp drop-on-latency=true '
                f'! {depay} ! {parser} ! {decoder} max-threads=1 output-corrupt=false'
            )
        pipeline = (
            f"{source} ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            f"videoconvert ! videoscale ! video/x-raw,format=BGR,width={width},height={height} ! "
            "appsink name=framesink sync=false max-buffers=1 drop=true"
        )
        return GstVideoCapture(pipeline, timeout_sec=self.connection_timeout)

    def _open_capture(self):
        src = build_source_url(
            self.cfg.get("source"),
            self.cfg.get("username"),
            self.cfg.get("password"),
        )

        if self.use_deepstream:
            cap = self._open_deepstream_capture(src)
            if cap.isOpened():
                log.info("Camera %s using DeepStream/GStreamer", self.cam_id)
                return cap
            cap.release()
            log.warning("Camera %s DeepStream failed; using OpenCV fallback", self.cam_id)

        if self.use_gstreamer:
            cap = self._open_software_gstreamer_capture(src)
            if cap.isOpened():
                log.info("Camera %s using low-latency GStreamer", self.cam_id)
                return cap
            cap.release()
            log.warning("Camera %s GStreamer failed; using OpenCV fallback", self.cam_id)

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

            # ─── Stream sifat sozlamalari ───
            target_w = int(self.cfg.get("width", self.cfg.get("resolution_w", 1920)))
            target_h = int(self.cfg.get("height", self.cfg.get("resolution_h", 1080)))
            target_fps = int(self.cfg.get("fps", 25))

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Haqiqiy qiymatlarni log qilish
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            log.info(
                "Camera %s stream: %dx%d @ %.1f fps (requested %dx%d @ %d)",
                self.cam_id, actual_w, actual_h, actual_fps,
                target_w, target_h, target_fps
            )

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

                # Frame ni AIWorker uchun optimal o'lchamga resize
                current_size = (frame.shape[1], frame.shape[0])
                if self.target_size and current_size != tuple(self.target_size):
                    frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

                # Buffer ga qo'yish (AIWorker shu frame ni oladi)
                self.buffer.put(frame)
                # Recording service uchun ham shu frame ni yuborish
                self.frame_bgr_ready.emit(self.cam_id, frame)
                self.frame_captured.emit(self.cam_id)

                frames += 1
                now = time.time()

                if now - last_fps_time >= 1.0:
                    self.health.set_fps(frames / max(1e-6, now - last_fps_time))
                    frames = 0
                    last_fps_time = time.time()
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