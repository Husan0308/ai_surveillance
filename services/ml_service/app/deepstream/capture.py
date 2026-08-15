from __future__ import annotations

from collections import deque
import re
import time

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

_CODEC = {
    "h264": ("H264", "rtph264depay", "h264parse"),
    "h265": ("H265", "rtph265depay", "h265parse"),
}


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _redact(value: str) -> str:
    value = re.sub(r"(?i)(rtsps?://)[^@/\s]+@", r"\1***:***@", value)
    value = re.sub(r'(?i)user-id="[^"]*"', 'user-id="***"', value)
    value = re.sub(r'(?i)user-pw="[^"]*"', 'user-pw="***"', value)
    return value


def _owned_bgr(mapped_data, width: int, height: int) -> np.ndarray:
    array = np.frombuffer(mapped_data, dtype=np.uint8).reshape((height, width, 4))
    return array[..., :3].copy()


class DeepStreamCapture:
    """Explicit RTSP + NVIDIA NVDEC capture for one camera.

    RTSP negotiation is owned by rtspsrc. NVIDIA owns decode/scale via
    nvv4l2decoder + nvvideoconvert. The downstream queue and appsink each keep
    only the newest frame, so no presentation backlog can grow.
    """

    backend = "rtspsrc-nvv4l2decoder"

    def __init__(
        self,
        camera_id: str,
        uri: str,
        codec: str,
        config,
        transport: str | None = None,
        username: str = "",
        password: str = "",
    ) -> None:
        self.camera_id = camera_id
        self.uri = uri
        self.codec = codec.lower()
        self.config = config
        self.transport = (transport or config.rtsp_transport).lower()
        self.username = username
        self.password = password
        self._opened = False
        self._last_error = ""
        self._last_warning = ""
        self._frame_intervals_ms = deque(maxlen=300)
        self._last_frame_mono: float | None = None

        if self.codec not in _CODEC:
            raise ValueError(f"{camera_id}: unsupported codec {codec}")
        for plugin in ("rtspsrc", "nvv4l2decoder", "nvvideoconvert", "appsink"):
            if Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"required GStreamer/NVIDIA plugin missing: {plugin}")

        self.pipeline_text = self._build_pipeline()
        self.pipeline = Gst.parse_launch(self.pipeline_text)
        self.bus = self.pipeline.get_bus()
        self.sink = self.pipeline.get_by_name("sink")
        self.latest_queue = self.pipeline.get_by_name("latest_queue")
        if self.sink is None:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"{camera_id}: appsink was not created")

        result = self.pipeline.set_state(Gst.State.PLAYING)
        self._opened = result != Gst.StateChangeReturn.FAILURE
        if not self._opened:
            detail = self._consume_bus()
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"{camera_id}: PLAYING failed: {detail or 'no detail'}")

    def _build_pipeline(self) -> str:
        c = self.config
        encoding, depay, parser = _CODEC[self.codec]
        queue_buffers = max(1, int(c.postdecode_queue_buffers))
        source_options = [
            f"location={_gst_quote(self.uri)}",
            f"latency={max(1, c.latency_ms)}",
            f"drop-on-latency={'true' if c.drop_on_latency else 'false'}",
            "buffer-mode=auto",
        ]

        # rtspsrc natively handles RTSP Basic/Digest challenges via these
        # properties; secrets never need to appear inside the committed URI.
        if self.username:
            source_options.append(f"user-id={_gst_quote(self.username)}")
            source_options.append(f"user-pw={_gst_quote(self.password)}")

        # 'auto' preserves GStreamer's normal UDP->TCP negotiation. The old
        # working project used auto; force TCP/UDP only when explicitly asked.
        if self.transport in {"tcp", "udp"}:
            source_options.append(f"protocols={self.transport}")
        if self.transport in {"udp", "auto"} and c.udp_buffer_size > 0:
            source_options.append(f"udp-buffer-size={c.udp_buffer_size}")

        return " ".join(
            [
                "rtspsrc", "name=source", *source_options,
                "!", f"application/x-rtp,media=video,encoding-name={encoding}",
                "!", depay, "name=depay",
                "!", parser, "name=parser",
                "!", "nvv4l2decoder", "name=decoder",
                f"num-extra-surfaces={max(1, c.decoder_extra_surfaces)}",
                "!", "queue", "name=latest_queue",
                f"max-size-buffers={queue_buffers}",
                "max-size-bytes=0", "max-size-time=0", "leaky=downstream", "silent=true",
                "!", "nvvideoconvert", "name=converter", f"gpu-id={c.gpu_id}",
                "!", f"video/x-raw,width={c.display_width},height={c.display_height},format=BGRx",
                "!", "appsink", "name=sink", "drop=true", "max-buffers=1", "sync=false",
                "wait-on-eos=false", "enable-last-sample=false",
            ]
        )

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
            "codec": self.codec,
            "auth_configured": bool(self.username),
            "pipeline": _redact(self.pipeline_text),
            "last_error": self._last_error,
            "last_warning": self._last_warning,
            "queue_buffers": self.current_queue_buffers(),
        }

    def close(self) -> None:
        self._opened = False
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
