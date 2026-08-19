from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit, urlunsplit

import gi
import numpy as np


gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _redact(value: str) -> str:
    return re.sub(r"(?i)(rtsps?://)[^@/\s]+@", r"\1***:***@", value)


def _uri_with_credentials(uri: str, username: str, password: str) -> str:
    if not username:
        return uri
    parsed = urlsplit(uri)
    if parsed.username:
        return uri
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    userinfo = quote(username, safe="")
    if password:
        userinfo += ":" + quote(password, safe="")
    netloc = f"{userinfo}@{host}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _owned_bgr(mapped_data, width: int, height: int) -> np.ndarray:
    array = np.frombuffer(mapped_data, dtype=np.uint8).reshape((height, width, 4))
    return array[..., :3].copy()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class DeepStreamCapture:
    """Latest-frame RTSP capture owned by DeepStream nvurisrcbin.

    nvurisrcbin performs RTSP negotiation, H.264/H.265 codec discovery and NVIDIA
    decode internally. One post-convert BGRx stream is split into two latest-only
    consumers: the in-process appsink used by detection and an optional shmsink
    used by the frontend native renderer. MJPEG remains a fallback path elsewhere.
    """

    backend = "deepstream-nvurisrcbin"

    def __init__(
        self,
        camera_id: str,
        uri: str,
        config,
        transport: str | None = None,
        username: str = "",
        password: str = "",
    ) -> None:
        self.camera_id = camera_id
        self.uri = uri
        self.config = config
        self.transport = (transport or config.rtsp_transport).lower()
        self.username = username
        self.password = password
        self._opened = False
        self._last_error = ""
        self._last_warning = ""
        self._frame_intervals_ms = deque(maxlen=300)
        self._last_frame_mono: float | None = None

        self.shm_enabled = _env_bool("ML_SHM_VIDEO_ENABLED", True)
        self.shm_dir = Path(os.getenv("ML_SHM_VIDEO_DIR", "/tmp/ai-surveillance")).expanduser()
        self.shm_socket = self.shm_dir / f"{self.camera_id}.sock"
        frame_bytes = int(config.display_width) * int(config.display_height) * 4
        requested_mb = max(2, int(os.getenv("ML_SHM_VIDEO_SIZE_MB", "8")))
        self.shm_size_bytes = max(requested_mb * 1024 * 1024, frame_bytes * 4)

        required_plugins = ["nvurisrcbin", "nvvideoconvert", "appsink"]
        if self.shm_enabled:
            required_plugins.append("shmsink")
        for plugin in required_plugins:
            if Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"required DeepStream/GStreamer plugin missing: {plugin}")

        if self.shm_enabled:
            self.shm_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.shm_socket.unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeError(f"{camera_id}: cannot clear shm socket {self.shm_socket}: {exc}") from exc

        self.pipeline_text = self._build_pipeline()
        self.pipeline = Gst.parse_launch(self.pipeline_text)
        self.bus = self.pipeline.get_bus()
        self.sink = self.pipeline.get_by_name("sink")
        self.latest_queue = self.pipeline.get_by_name("latest_queue")
        self.shm_sink = self.pipeline.get_by_name("shm_sink") if self.shm_enabled else None
        if self.sink is None:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"{camera_id}: appsink was not created")
        if self.shm_enabled and self.shm_sink is None:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"{camera_id}: shmsink was not created")

        result = self.pipeline.set_state(Gst.State.PLAYING)
        self._opened = result != Gst.StateChangeReturn.FAILURE
        if not self._opened:
            detail = self._consume_bus()
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"{camera_id}: PLAYING failed: {detail or 'no detail'}")

    def _build_pipeline(self) -> str:
        c = self.config
        queue_buffers = max(1, int(c.postdecode_queue_buffers))
        reconnect_interval = max(1, int(round(c.reconnect_delay_sec)))
        rtp_protocol = 4 if self.transport == "tcp" else 0
        source_uri = _uri_with_credentials(self.uri, self.username, self.password)

        parts = [
            "nvurisrcbin",
            "name=source",
            f"uri={_gst_quote(source_uri)}",
            f"gpu-id={int(c.gpu_id)}",
            f"cudadec-memtype={int(c.cudadec_memtype)}",
            f"num-extra-surfaces={max(1, int(c.decoder_extra_surfaces))}",
            f"select-rtp-protocol={rtp_protocol}",
            f"latency={max(1, int(c.latency_ms))}",
            f"drop-on-latency={'true' if c.drop_on_latency else 'false'}",
            f"udp-buffer-size={max(1, int(c.udp_buffer_size))}",
            f"rtsp-reconnect-interval={reconnect_interval}",
            "rtsp-reconnect-attempts=-1",
            "disable-audio=true",
            "async-handling=true",
            "message-forward=true",
            "!",
            "queue",
            "name=latest_queue",
            f"max-size-buffers={queue_buffers}",
            "max-size-bytes=0",
            "max-size-time=0",
            "leaky=downstream",
            "silent=true",
            "!",
            "nvvideoconvert",
            "name=converter",
            f"gpu-id={int(c.gpu_id)}",
            "!",
            f"video/x-raw,width={int(c.display_width)},height={int(c.display_height)},format=BGRx",
        ]

        if not self.shm_enabled:
            parts.extend(
                [
                    "!",
                    "appsink",
                    "name=sink",
                    "drop=true",
                    "max-buffers=1",
                    "sync=false",
                    "wait-on-eos=false",
                    "enable-last-sample=false",
                ]
            )
            return " ".join(parts)

        parts.extend(
            [
                "!",
                "tee",
                "name=frame_tee",
                "frame_tee.",
                "!",
                "queue",
                "name=app_queue",
                "max-size-buffers=1",
                "max-size-bytes=0",
                "max-size-time=0",
                "leaky=downstream",
                "silent=true",
                "!",
                "appsink",
                "name=sink",
                "drop=true",
                "max-buffers=1",
                "sync=false",
                "wait-on-eos=false",
                "enable-last-sample=false",
                "frame_tee.",
                "!",
                "queue",
                "name=shm_queue",
                "max-size-buffers=1",
                "max-size-bytes=0",
                "max-size-time=0",
                "leaky=downstream",
                "silent=true",
                "!",
                "shmsink",
                "name=shm_sink",
                f"socket-path={_gst_quote(str(self.shm_socket))}",
                f"shm-size={int(self.shm_size_bytes)}",
                "wait-for-connection=false",
                "sync=false",
                "async=false",
                "qos=false",
            ]
        )
        return " ".join(parts)

    def _consume_bus(self) -> str | None:
        terminal = None
        while self.bus is not None:
            message = self.bus.pop_filtered(
                Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS
            )
            if message is None:
                break
            source = message.src.get_name() if message.src is not None else "unknown"
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                self._last_error = f"{source}: {err.message} | {debug or ''}"
                terminal = self._last_error
            elif message.type == Gst.MessageType.WARNING:
                err, debug = message.parse_warning()
                self._last_warning = f"{source}: {err.message} | {debug or ''}"
            elif message.type == Gst.MessageType.EOS:
                terminal = f"EOS from {source}"
        return terminal

    def is_opened(self) -> bool:
        return self._opened

    def last_error(self) -> str:
        self._consume_bus()
        return self._last_error

    def current_queue_buffers(self) -> int | None:
        if self.latest_queue is None:
            return None
        try:
            return int(self.latest_queue.get_property("current-level-buffers"))
        except Exception:
            return None

    def read(self):
        if not self._opened:
            return False, None
        timeout_ns = max(100, int(self.config.capture_timeout_ms)) * Gst.MSECOND
        sample = self.sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            terminal = self._consume_bus()
            if terminal:
                self._opened = False
                raise RuntimeError(terminal)
            return False, None

        caps = sample.get_caps()
        buffer = sample.get_buffer()
        if caps is None or buffer is None or caps.get_size() == 0:
            return False, None
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        pixel_format = str(structure.get_value("format"))
        if pixel_format != "BGRx":
            raise RuntimeError(f"{self.camera_id}: unexpected appsink format {pixel_format}")

        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            image = _owned_bgr(mapped.data, width, height)
        finally:
            buffer.unmap(mapped)

        now = time.monotonic()
        if self._last_frame_mono is not None:
            self._frame_intervals_ms.append((now - self._last_frame_mono) * 1000.0)
        self._last_frame_mono = now
        return True, image

    def debug_info(self) -> dict:
        self._consume_bus()
        return {
            "backend": self.backend,
            "transport": self.transport,
            "auth_configured": bool(self.username),
            "pipeline": _redact(self.pipeline_text),
            "last_error": self._last_error,
            "last_warning": self._last_warning,
            "queue_buffers": self.current_queue_buffers(),
            "shm_enabled": self.shm_enabled,
            "shm_socket": str(self.shm_socket) if self.shm_enabled else "",
            "shm_size_bytes": int(self.shm_size_bytes) if self.shm_enabled else 0,
        }

    def close(self) -> None:
        self._opened = False
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        if self.shm_enabled:
            try:
                self.shm_socket.unlink(missing_ok=True)
            except OSError:
                pass
