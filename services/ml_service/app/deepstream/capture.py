from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import time
import os

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from services.ml_service.app.deepstream.decoder_sample import BoundedDecoderSample

Gst.init(None)

_CODEC = {
    "h264": ("H264", "rtph264depay", "h264parse"),
    "h265": ("H265", "rtph265depay", "h265parse"),
}


@dataclass(frozen=True)
class CaptureTiming:
    decoder_reference_ns: int
    decoder_out_ns: int
    appsink_receive_ns: int
    pts_ns: int
    dts_ns: int


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


def _decoder_extra_surfaces_for_camera(
    camera_id: str, configured: int, test_overrides: str | None = None
) -> int:
    """Resolve an opt-in per-camera decoder-surface experiment value."""
    value = max(0, int(configured))
    if not test_overrides:
        return value
    for entry in test_overrides.split(","):
        camera, separator, raw_value = entry.strip().partition("=")
        if not separator or not re.fullmatch(r"CAM-\d{2}", camera):
            raise ValueError("decoder surface override must use CAM-XX=0..24 entries")
        try:
            candidate = int(raw_value)
        except ValueError as exc:
            raise ValueError("decoder surface override must use CAM-XX=0..24 entries") from exc
        if not 0 <= candidate <= 24:
            raise ValueError("decoder extra surfaces must be in the dGPU-supported range 0..24")
        if camera == camera_id:
            value = candidate
    return value


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
        output_bgrx: bool = False,
        latency_ms: int | None = None,
        low_latency_mode: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.uri = uri
        self.codec = codec.lower()
        self.config = config
        self.transport = (transport or config.rtsp_transport).lower()
        self.username = username
        self.password = password
        self.output_bgrx = bool(output_bgrx)
        self.low_latency_mode = bool(low_latency_mode)
        self.low_latency_mode_effective: bool | None = None
        self.latency_ms = int(config.latency_ms if latency_ms is None else latency_ms)
        if self.latency_ms < 1:
            raise ValueError(f"{camera_id}: latency_ms must be >= 1")
        self._opened = False
        self._last_error = ""
        self._last_warning = ""
        self._frame_intervals_ms = deque(maxlen=300)
        self._last_frame_mono: float | None = None
        self.last_timing = CaptureTiming(0, 0, 0, 0, 0)
        self._decoder_input_times: dict[int, int] = {}
        self._decoder_output_times: dict[int, tuple[int, int]] = {}
        self._decoder_input_order: deque[int] = deque(maxlen=256)
        self._decoder_output_order: deque[int] = deque(maxlen=256)
        self._decoder_timing_started_ns = 0
        self._decoder_input_buffers = 0
        self._decoder_output_buffers = 0
        self._last_decoder_input_mono_ns = 0
        self._last_decoder_output_mono_ns = 0
        self._decoder_input_intervals_ns: deque[int] = deque(maxlen=512)
        self._decoder_output_intervals_ns: deque[int] = deque(maxlen=512)
        self.decoder_input_pts_backsteps = 0
        self.decoder_input_max_pts_backstep_ns = 0
        self.decoder_output_pts_backsteps = 0
        self.decoder_output_max_pts_backstep_ns = 0
        self._last_decoder_input_pts = -1
        self._last_decoder_output_pts = -1
        self.decoder_dts_samples = 0
        dump_directory = os.getenv("MV3DT_DECODER_AU_DUMP_DIR")
        self._decoder_sample = (
            BoundedDecoderSample(dump_directory, camera_id, self.codec)
            if dump_directory else None
        )

        if self.codec not in _CODEC:
            raise ValueError(f"{camera_id}: unsupported codec {codec}")
        for plugin in ("rtspsrc", "nvv4l2decoder", "nvvideoconvert", "appsink"):
            if Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"required GStreamer/NVIDIA plugin missing: {plugin}")

        self.pipeline_text = self._build_pipeline()
        self.pipeline = Gst.parse_launch(self.pipeline_text)
        self.bus = self.pipeline.get_bus()
        self.sink = self.pipeline.get_by_name("sink")
        self.decoder = self.pipeline.get_by_name("decoder")
        self.latest_queue = self.pipeline.get_by_name("latest_queue")
        if self.sink is None:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"{camera_id}: appsink was not created")
        if self.decoder is not None:
            low_latency_property = self.decoder.find_property("low-latency-mode")
            if low_latency_property is None and self.low_latency_mode:
                self.pipeline.set_state(Gst.State.NULL)
                raise RuntimeError(
                    f"{camera_id}: nvv4l2decoder does not expose low-latency-mode"
                )
            if low_latency_property is not None:
                self.low_latency_mode_effective = bool(
                    self.decoder.get_property("low-latency-mode")
                )
            decoder_sink = self.decoder.get_static_pad("sink")
            decoder_src = self.decoder.get_static_pad("src")
            if decoder_sink is not None:
                decoder_sink.add_probe(Gst.PadProbeType.BUFFER, self._decoder_input_probe)
            if decoder_src is not None:
                decoder_src.add_probe(Gst.PadProbeType.BUFFER, self._decoder_output_probe)

        result = self.pipeline.set_state(Gst.State.PLAYING)
        self._opened = result != Gst.StateChangeReturn.FAILURE
        if not self._opened:
            detail = self._consume_bus()
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"{camera_id}: PLAYING failed: {detail or 'no detail'}")

    @staticmethod
    def _buffer_pts(buffer) -> int:
        return int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else -1

    @staticmethod
    def _remember(mapping: dict, order: deque, key: int, value) -> None:
        if key < 0:
            return
        if len(order) == order.maxlen:
            mapping.pop(order.popleft(), None)
        order.append(key)
        mapping[key] = value

    def _decoder_input_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        now_ns = time.monotonic_ns()
        if not self._decoder_timing_started_ns:
            self._decoder_timing_started_ns = now_ns
        self._decoder_input_buffers += 1
        if self._last_decoder_input_mono_ns:
            self._decoder_input_intervals_ns.append(now_ns - self._last_decoder_input_mono_ns)
        self._last_decoder_input_mono_ns = now_ns
        pts = self._buffer_pts(buffer)
        dts = int(buffer.dts) if buffer.dts != Gst.CLOCK_TIME_NONE else 0
        if pts >= 0:
            if self._last_decoder_input_pts >= 0 and pts < self._last_decoder_input_pts:
                backstep = self._last_decoder_input_pts - pts
                self.decoder_input_pts_backsteps += 1
                self.decoder_input_max_pts_backstep_ns = max(
                    self.decoder_input_max_pts_backstep_ns, backstep
                )
            self._last_decoder_input_pts = pts
        if dts > 0:
            self.decoder_dts_samples += 1
        if self._decoder_sample is not None:
            mapped_ok, mapped = buffer.map(Gst.MapFlags.READ)
            if mapped_ok:
                try:
                    self._decoder_sample.append(mapped.data)
                finally:
                    buffer.unmap(mapped)
        self._remember(
            self._decoder_input_times, self._decoder_input_order,
            pts, (now_ns, dts),
        )
        return Gst.PadProbeReturn.OK

    def _decoder_output_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        pts = self._buffer_pts(buffer)
        out_ns = time.monotonic_ns()
        if not self._decoder_timing_started_ns:
            self._decoder_timing_started_ns = out_ns
        self._decoder_output_buffers += 1
        if self._last_decoder_output_mono_ns:
            self._decoder_output_intervals_ns.append(out_ns - self._last_decoder_output_mono_ns)
        self._last_decoder_output_mono_ns = out_ns
        if pts >= 0:
            if self._last_decoder_output_pts >= 0 and pts < self._last_decoder_output_pts:
                backstep = self._last_decoder_output_pts - pts
                self.decoder_output_pts_backsteps += 1
                self.decoder_output_max_pts_backstep_ns = max(
                    self.decoder_output_max_pts_backstep_ns, backstep
                )
            self._last_decoder_output_pts = pts
        input_timing = self._decoder_input_times.pop(pts, None)
        if input_timing is None:
            # Missing input evidence is unknown, not zero decoder latency.
            # Keep it invalid so the paint auditor cannot manufacture a PASS.
            reference_ns, dts_ns = 0, 0
        else:
            reference_ns, dts_ns = input_timing
        self._remember(
            self._decoder_output_times, self._decoder_output_order,
            pts, (reference_ns, out_ns, dts_ns),
        )
        return Gst.PadProbeReturn.OK

    def _build_pipeline(self) -> str:
        c = self.config
        encoding, depay, parser = _CODEC[self.codec]
        queue_buffers = max(1, int(c.postdecode_queue_buffers))
        decoder_extra_surfaces = _decoder_extra_surfaces_for_camera(
            self.camera_id,
            c.decoder_extra_surfaces,
            os.getenv("MV3DT_TEST_DECODER_EXTRA_SURFACES_BY_CAMERA"),
        )
        source_options = [
            f"location={_gst_quote(self.uri)}",
            f"latency={self.latency_ms}",
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
                *( ["low-latency-mode=true"] if self.low_latency_mode else [] ),
                f"num-extra-surfaces={decoder_extra_surfaces}",
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
            if self.output_bgrx:
                image = np.frombuffer(mapped.data, dtype=np.uint8).reshape((height, width, 4)).copy()
            else:
                image = _owned_bgr(mapped.data, width, height)
        finally:
            buffer.unmap(mapped)

        receive_ns = time.monotonic_ns()
        pts_ns = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else -1
        dts_ns = int(buffer.dts) if buffer.dts != Gst.CLOCK_TIME_NONE else -1
        decoder_reference_ns, decoder_out_ns, decoder_dts_ns = self._decoder_output_times.pop(
            pts_ns, (0, 0, 0)
        )
        self.last_timing = CaptureTiming(
            decoder_reference_ns, decoder_out_ns, receive_ns, pts_ns,
            decoder_dts_ns or dts_ns,
        )
        now = receive_ns / 1e9
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
            "latency_ms": self.latency_ms,
            "auth_configured": bool(self.username),
            "pipeline": _redact(self.pipeline_text),
            "last_error": self._last_error,
            "last_warning": self._last_warning,
            "queue_buffers": self.current_queue_buffers(),
            "decoder_pts_order": self.decoder_pts_order_diagnostics(),
            "decoder_sample_bytes": self._decoder_sample.bytes_written if self._decoder_sample else 0,
        }

    def decoder_pts_order_diagnostics(self) -> dict:
        """Return bounded counters proving whether decoder input PTS reorders."""
        elapsed_ns = (
            max(0, time.monotonic_ns() - self._decoder_timing_started_ns)
            if self._decoder_timing_started_ns else 0
        )

        def interval_stats(samples: deque[int]) -> dict:
            ordered = sorted(samples)
            if not ordered:
                return {"p50_ms": None, "p95_ms": None}
            p50 = ordered[int((len(ordered) - 1) * 0.50)]
            p95 = ordered[int((len(ordered) - 1) * 0.95)]
            return {"p50_ms": p50 / 1e6, "p95_ms": p95 / 1e6}

        return {
            "input_pts_backsteps": self.decoder_input_pts_backsteps,
            "input_max_pts_backstep_ms": self.decoder_input_max_pts_backstep_ns / 1e6,
            "output_pts_backsteps": self.decoder_output_pts_backsteps,
            "output_max_pts_backstep_ms": self.decoder_output_max_pts_backstep_ns / 1e6,
            "input_buffers_with_dts": self.decoder_dts_samples,
            "input_buffers": self._decoder_input_buffers,
            "output_buffers": self._decoder_output_buffers,
            "input_fps": self._decoder_input_buffers * 1e9 / elapsed_ns if elapsed_ns else 0.0,
            "output_fps": self._decoder_output_buffers * 1e9 / elapsed_ns if elapsed_ns else 0.0,
            "input_interval": interval_stats(self._decoder_input_intervals_ns),
            "output_interval": interval_stats(self._decoder_output_intervals_ns),
        }

    def close(self) -> None:
        self._opened = False
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        if self._decoder_sample is not None:
            self._decoder_sample.close()
