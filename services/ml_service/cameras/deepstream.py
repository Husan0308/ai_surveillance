"""DeepStream RTSP capture backend for Core-v1.

The backend deliberately uses DeepStream only for URI ingest, NVDEC and color
conversion. Detection stays in the existing isolated PyTorch CUDA process so we
do not depend on TensorRT/nvinfer support on Pascal GPUs.

Pipeline invariants:
- RTSP jitter is bounded with drop-on-latency.
- decoder gets extra surfaces to avoid starvation.
- only one decoded frame may wait downstream.
- appsink keeps only the newest frame.
- no nvstreammux is used for display/capture, so a slow camera cannot stall the
  other five cameras.
"""
from __future__ import annotations

from collections import deque
import time

from .gstreamer import (
    _gstreamer,
    _gst_quote,
    authenticated_source,
    jitter_nanoseconds_to_ms,
    owned_bgr_from_mapped,
    redacted_pipeline,
)


def deepstream_available() -> bool:
    try:
        Gst = _gstreamer()
        return Gst.ElementFactory.find("nvurisrcbin") is not None
    except Exception:
        return False


def deepstream_rtsp_pipeline(config: dict) -> str:
    source = authenticated_source(config)
    if not isinstance(source, str) or not source.lower().startswith(("rtsp://", "rtsps://")):
        raise ValueError(f"camera {config.get('id')} DeepStream backend requires an RTSP source")

    latency = max(1, int(config.get("latency_ms", 150)))
    extra_surfaces = max(1, int(config.get("decoder_extra_surfaces", 4)))
    udp_buffer_size = max(65536, int(config.get("udp_buffer_size", 1048576)))
    reconnect_interval = max(1, int(config.get("deepstream_reconnect_interval_sec", 2)))
    reconnect_attempts = int(config.get("deepstream_reconnect_attempts", -1))
    drop_on_latency = bool(config.get("drop_on_latency", True))
    postdecode_queue_buffers = max(1, int(config.get("postdecode_queue_buffers", 1)))

    transport = str(config.get("rtsp_transport", "auto")).strip().lower()
    # nvurisrcbin: 0 = UDP/UDP-mcast/TCP, 4 = TCP only.
    select_rtp_protocol = 4 if transport == "tcp" else 0

    source_options = [
        f"uri={_gst_quote(source)}",
        "disable-audio=true",
        f"select-rtp-protocol={select_rtp_protocol}",
        f"latency={latency}",
        f"drop-on-latency={'true' if drop_on_latency else 'false'}",
        f"num-extra-surfaces={extra_surfaces}",
        "cudadec-memtype=0",
        f"udp-buffer-size={udp_buffer_size}",
        f"rtsp-reconnect-interval={reconnect_interval}",
        f"rtsp-reconnect-attempts={reconnect_attempts}",
    ]

    postdecode_queue = (
        f"queue name=latest_queue max-size-buffers={postdecode_queue_buffers} "
        "max-size-bytes=0 max-size-time=0 leaky=downstream silent=true"
    )

    # nvurisrcbin already owns uridecodebin + nvv4l2decoder. Keep frames in the
    # NVIDIA path until the final conversion required by the existing numpy
    # detector/publisher code.
    return (
        f"nvurisrcbin name=source {' '.join(source_options)} ! "
        f"{postdecode_queue} ! "
        "nvvideoconvert name=converter ! video/x-raw,format=BGRx ! "
        "appsink name=sink drop=true max-buffers=1 sync=false wait-on-eos=false"
    )


class DeepStreamCapture:
    backend = "deepstream-nvurisrcbin"

    def __init__(self, config: dict):
        Gst = _gstreamer()
        if Gst.ElementFactory.find("nvurisrcbin") is None:
            raise RuntimeError("DeepStream nvurisrcbin plugin is not available")

        self.Gst = Gst
        self.config = dict(config)
        self.pipeline = deepstream_rtsp_pipeline(config)
        self._pull_timeout_ns = max(100, int(config.get("capture_timeout_ms", 1000))) * Gst.MSECOND
        self._last_bus_error = ""
        self._last_bus_warning = ""
        self._last_pipeline_lag_ms = None
        self._pipeline_lag_ms = deque(maxlen=2000)
        self._map_copy_ms = deque(maxlen=2000)
        self._frame_intervals_ms = deque(maxlen=2000)
        self._last_frame_mono = None
        self._state_change_result = "UNKNOWN"
        self._source_runtime = {
            "backend": self.backend,
            "transport": str(config.get("rtsp_transport", "auto")),
            "latency_ms": int(config.get("latency_ms", 150)),
            "drop_on_latency": bool(config.get("drop_on_latency", True)),
            "decoder_extra_surfaces": int(config.get("decoder_extra_surfaces", 4)),
            "cudadec_memtype": 0,
            "udp_buffer_size": int(config.get("udp_buffer_size", 1048576)),
            "postdecode_queue_buffers": int(config.get("postdecode_queue_buffers", 1)),
            "rtsp_reconnect_interval_sec": int(config.get("deepstream_reconnect_interval_sec", 2)),
        }

        self._pipeline = Gst.parse_launch(self.pipeline)
        self._bus = self._pipeline.get_bus()
        self._source = self._pipeline.get_by_name("source")
        self._latest_queue = self._pipeline.get_by_name("latest_queue")
        self._sink = self._pipeline.get_by_name("sink") or self._pipeline.get_by_name("appsink0")
        if self._sink is None:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("DeepStream pipeline has no appsink")

        result = self._pipeline.set_state(Gst.State.PLAYING)
        self._state_change_result = str(result.value_nick if hasattr(result, "value_nick") else result)
        self._opened = result != Gst.StateChangeReturn.FAILURE
        if not self._opened:
            terminal = self._consume_bus_messages()
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"DeepStream set_state(PLAYING) failed: {terminal or 'no bus detail'}")

    def _consume_bus_messages(self):
        terminal = None
        while self._bus is not None:
            message = self._bus.pop_filtered(
                self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS
            )
            if message is None:
                break
            source = message.src.get_name() if message.src is not None else "unknown"
            if message.type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                self._last_bus_error = f"source={source} message={err.message} debug={debug or ''}"
                terminal = self._last_bus_error
            elif message.type == self.Gst.MessageType.WARNING:
                err, debug = message.parse_warning()
                self._last_bus_warning = f"source={source} message={err.message} debug={debug or ''}"
            elif message.type == self.Gst.MessageType.EOS:
                terminal = f"EOS from {source}"
        return terminal

    def isOpened(self):
        return self._opened

    def last_error(self):
        return self._last_bus_error

    def current_pipeline_lag_ms(self):
        return self._last_pipeline_lag_ms

    def current_queue_buffers(self):
        if self._latest_queue is None:
            return None
        try:
            return int(self._latest_queue.get_property("current-level-buffers"))
        except Exception:
            return None

    def source_runtime(self):
        return dict(self._source_runtime)

    def _measure_pipeline_lag(self, buffer):
        try:
            pts = buffer.pts
            if pts == self.Gst.CLOCK_TIME_NONE:
                return
            clock = self._pipeline.get_clock()
            if clock is None:
                return
            base_time = self._pipeline.get_base_time()
            clock_time = clock.get_time()
            if base_time == self.Gst.CLOCK_TIME_NONE or clock_time < base_time:
                return
            running_time = clock_time - base_time
            lag_ms = max(0.0, float(running_time - pts) / float(self.Gst.MSECOND)) if running_time >= pts else 0.0
            self._last_pipeline_lag_ms = lag_ms
            self._pipeline_lag_ms.append(lag_ms)
        except Exception:
            pass

    def read(self):
        if not self._opened or self._sink is None:
            return False, None
        sample = self._sink.emit("try-pull-sample", self._pull_timeout_ns)
        if sample is None:
            terminal = self._consume_bus_messages()
            if terminal:
                self._opened = False
                raise RuntimeError(f"DeepStream terminal error: {terminal}")
            return False, None

        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        pixel_format = str(caps.get_value("format"))
        buffer = sample.get_buffer()
        self._measure_pipeline_lag(buffer)

        now = time.monotonic()
        if self._last_frame_mono is not None:
            self._frame_intervals_ms.append(max(0.0, (now - self._last_frame_mono) * 1000.0))
        self._last_frame_mono = now

        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return False, None
        copied = time.perf_counter()
        try:
            frame = owned_bgr_from_mapped(mapped.data, width, height, pixel_format)
        finally:
            buffer.unmap(mapped)
        self._map_copy_ms.append((time.perf_counter() - copied) * 1000.0)
        return True, frame

    @staticmethod
    def _stats(values):
        ordered = sorted(values)
        if not ordered:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        def pct(p):
            return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]
        return {"count": len(ordered), "p50": pct(.50), "p95": pct(.95), "max": ordered[-1]}

    def stage_metrics(self):
        self._consume_bus_messages()
        return {
            "backend": self.backend,
            "frame_interval_ms": self._stats(self._frame_intervals_ms),
            "map_copy_ms": self._stats(self._map_copy_ms),
            "pipeline_lag_ms": self._stats(self._pipeline_lag_ms),
            "current_pipeline_lag_ms": self._last_pipeline_lag_ms,
            "current_postdecode_queue_buffers": self.current_queue_buffers(),
            "source_runtime": dict(self._source_runtime),
            "state_change_result": self._state_change_result,
            "last_bus_error": self._last_bus_error,
            "last_bus_warning": self._last_bus_warning,
            "pipeline": redacted_pipeline(self.pipeline),
        }

    def interrupt(self):
        self._opened = False

    def release(self):
        if getattr(self, "_pipeline", None) is not None:
            self._pipeline.set_state(self.Gst.State.NULL)
        self._opened = False
