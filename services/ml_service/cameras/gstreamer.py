"""NVIDIA/GStreamer RTSP capture adapter with latest-sample semantics."""
from __future__ import annotations

from urllib.parse import quote, urlsplit, urlunsplit
import re
import threading
import time
from collections import deque

SUPPORTED_CODECS = {"h264": ("rtph264depay", "h264parse"), "h265": ("rtph265depay", "h265parse"), "hevc": ("rtph265depay", "h265parse")}
_GST_INIT_LOCK = threading.Lock()
_GST = None


def jitter_nanoseconds_to_ms(value):
    if value is None:
        return None
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _gstreamer():
    global _GST
    with _GST_INIT_LOCK:
        if _GST is None:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)
            _GST = Gst
        return _GST


def authenticated_source(config: dict) -> str | int:
    source = config.get("source", config.get("rtsp_url"))
    if isinstance(source, str) and source.isdigit():
        return int(source)
    user, password = config.get("username"), config.get("password")
    if not isinstance(source, str) or not user or not password:
        return source
    parsed = urlsplit(source)
    if "@" in parsed.netloc:
        return source
    credentials = f"{quote(str(user), safe='')}:{quote(str(password), safe='')}@"
    return urlunsplit((parsed.scheme, credentials + parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def redacted_pipeline(value: str) -> str:
    return re.sub(r'(?i)(location="?rtsps?://)[^@"\s]+@', r'\1***:***@', str(value))


def owned_bgr_from_mapped(data, width, height, pixel_format):
    import numpy as np
    channels = 4 if str(pixel_format) == "BGRx" else 3
    mapped = np.frombuffer(data, dtype=np.uint8).reshape((int(height), int(width), channels))
    return mapped[..., :3].copy()


def nvidia_rtsp_pipeline(config: dict) -> str:
    codec = str(config.get("codec", "")).lower()
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"camera {config.get('id')} requires codec=h264 or codec=h265")
    source = authenticated_source(config)
    if not isinstance(source, str) or not source.lower().startswith(("rtsp://", "rtsps://")):
        raise ValueError(f"camera {config.get('id')} NVIDIA backend requires an RTSP source")

    depay, parser = SUPPORTED_CODECS[codec]
    latency = max(0, int(config.get("latency_ms", 50)))
    decoder_backend = str(config.get("decoder_backend", "nvv4l2decoder")).lower()
    extra_surfaces = max(0, int(config.get("decoder_extra_surfaces", 2)))
    low_latency = bool(config.get("decoder_low_latency_mode", False))

    if decoder_backend == "nvcodec":
        decoder = "nvh264dec name=decoder max-display-delay=0" if codec == "h264" else "nvh265dec name=decoder max-display-delay=0"
        conversion = "video/x-raw,format=NV12 ! videoconvert name=converter ! video/x-raw,format=BGR"
    elif decoder_backend == "nvv4l2decoder":
        decoder = f"nvv4l2decoder name=decoder num-extra-surfaces={extra_surfaces}"
        if low_latency:
            decoder += " low-latency-mode=true"
        conversion = "nvvideoconvert name=converter ! video/x-raw,format=BGRx"
    else:
        raise ValueError(f"unsupported NVIDIA decoder backend: {decoder_backend}")

    encoding = "H264" if codec == "h264" else "H265"
    transport = str(config.get("rtsp_transport", "tcp")).strip().lower()
    if transport not in {"tcp", "udp", "auto"}:
        raise ValueError(f"unsupported RTSP transport: {transport}")

    buffer_mode = str(config.get("rtsp_buffer_mode", "auto")).strip().lower()
    if buffer_mode not in {"none", "slave", "buffer", "auto", "synced"}:
        raise ValueError(f"unsupported rtspsrc buffer-mode: {buffer_mode}")

    source_options = [
        f"location={_gst_quote(source)}",
        f"latency={latency}",
        f"buffer-mode={buffer_mode}",
        f"drop-on-latency={'true' if bool(config.get('drop_on_latency', False)) else 'false'}",
    ]
    if transport != "auto":
        source_options.append(f"protocols={transport}")
    udp_buffer_size = int(config.get("udp_buffer_size", 0) or 0)
    if transport in {"udp", "auto"} and udp_buffer_size > 0:
        source_options.append(f"udp-buffer-size={udp_buffer_size}")

    return (
        f"rtspsrc name=source {' '.join(source_options)} ! "
        f"application/x-rtp,media=video,encoding-name={encoding} ! "
        f"{depay} name=depay ! {parser} name=parser ! {decoder} ! {conversion} ! "
        "appsink name=sink drop=true max-buffers=1 sync=false wait-on-eos=false"
    )


class GStreamerCapture:
    backend = "gstreamer-nvdec"

    def __init__(self, config: dict):
        Gst = _gstreamer()
        self.Gst = Gst
        self.config = dict(config)
        self.pipeline = nvidia_rtsp_pipeline(config)
        self._pull_timeout_ns = max(100, int(config.get("capture_timeout_ms", 1000))) * Gst.MSECOND
        self._last_bus_error = ""
        self._last_bus_warning = ""
        self._state_change_result = "UNKNOWN"
        self._last_pipeline_lag_ms = None
        self._pipeline_lag_ms = deque(maxlen=2000)
        self._source_runtime = {}

        self._pipeline = Gst.parse_launch(self.pipeline)
        self._bus = self._pipeline.get_bus()
        self._source = self._pipeline.get_by_name("source")
        self._sink = self._pipeline.get_by_name("sink") or self._pipeline.get_by_name("appsink0")

        # GStreamer 1.24.10+ exposes tcp-timestamp. When available it timestamps
        # every RTP-over-TCP packet at receive time, preventing sender/client
        # clock drift from silently growing end-to-end latency. Older 1.24 builds
        # simply skip the property and continue with the configured buffer mode.
        if self._source is not None:
            tcp_timestamp_requested = bool(config.get("tcp_timestamp", True))
            pspec = self._source.find_property("tcp-timestamp")
            if pspec is not None:
                self._source.set_property("tcp-timestamp", tcp_timestamp_requested)
                self._source_runtime["tcp_timestamp_supported"] = True
                self._source_runtime["tcp_timestamp"] = tcp_timestamp_requested
            else:
                self._source_runtime["tcp_timestamp_supported"] = False
                self._source_runtime["tcp_timestamp"] = None
            self._source_runtime["buffer_mode"] = str(config.get("rtsp_buffer_mode", "auto"))
            self._source_runtime["transport"] = str(config.get("rtsp_transport", "tcp"))
            self._source_runtime["latency_ms"] = int(config.get("latency_ms", 50))
            self._source_runtime["drop_on_latency"] = bool(config.get("drop_on_latency", False))

        stages = ("rtp_receive", "jitterbuffer_input", "jitterbuffer_output", "depay_output", "parser_input", "parser_output", "decoder_input", "decoder_output", "conversion_output", "appsink")
        self._stage_last = {}
        self._stage_intervals = {name: deque(maxlen=2000) for name in stages}
        self._map_copy_ms = deque(maxlen=2000)
        self._jitterbuffers = []

        def probe(name):
            def callback(_pad, _info):
                now = time.monotonic()
                previous = self._stage_last.get(name)
                self._stage_last[name] = now
                if previous is not None:
                    self._stage_intervals[name].append(max(0.0, (now - previous) * 1000.0))
                return Gst.PadProbeReturn.OK
            return callback

        def attach_jitterbuffer(element):
            factory = element.get_factory() if element is not None else None
            if factory is None or factory.get_name() != "rtpjitterbuffer" or element in self._jitterbuffers:
                return
            self._jitterbuffers.append(element)
            for pad_name, stage in (("sink", "jitterbuffer_input"), ("src", "jitterbuffer_output")):
                pad = element.get_static_pad(pad_name)
                if pad is not None:
                    pad.add_probe(Gst.PadProbeType.BUFFER, probe(stage))

        self._pipeline.connect("deep-element-added", lambda *_args: attach_jitterbuffer(_args[-1]))
        depay = self._pipeline.get_by_name("depay")
        parser = self._pipeline.get_by_name("parser")
        decoder = self._pipeline.get_by_name("decoder")
        converter = self._pipeline.get_by_name("converter")
        elements = (
            (depay, "sink", "rtp_receive"),
            (depay, "src", "depay_output"),
            (parser, "sink", "parser_input"),
            (parser, "src", "parser_output"),
            (decoder, "sink", "decoder_input"),
            (decoder, "src", "decoder_output"),
            (converter, "src", "conversion_output"),
            (self._sink, "sink", "appsink"),
        )
        for element, pad_name, stage in elements:
            if element is not None:
                pad = element.get_static_pad(pad_name)
                if pad is not None:
                    pad.add_probe(Gst.PadProbeType.BUFFER, probe(stage))

        result = self._pipeline.set_state(Gst.State.PLAYING)
        self._state_change_result = str(result.value_nick if hasattr(result, "value_nick") else result)
        self._opened = result != Gst.StateChangeReturn.FAILURE
        if not self._opened:
            terminal = self._consume_bus_messages()
            raise RuntimeError(f"GStreamer set_state(PLAYING) failed: {terminal or 'no bus detail'}")

    def _consume_bus_messages(self):
        terminal = None
        while self._bus is not None:
            message = self._bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS)
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
                raise RuntimeError(f"GStreamer terminal error: {terminal}")
            return False, None

        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        pixel_format = str(caps.get_value("format"))
        buffer = sample.get_buffer()
        self._measure_pipeline_lag(buffer)
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

    def stage_metrics(self):
        self._consume_bus_messages()

        def stats(values):
            ordered = sorted(values)
            def pct(p):
                return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))] if ordered else 0.0
            return {"count": len(ordered), "p50": pct(.50), "p95": pct(.95), "max": ordered[-1] if ordered else 0.0}

        jitter = {"available": False, "lost": None, "late": None, "duplicates": None, "average_jitter_ms": None}
        for element in self._jitterbuffers:
            try:
                structure = element.get_property("stats")
                def field(*names):
                    for name in names:
                        if structure.has_field(name):
                            return structure.get_value(name)
                average_jitter = field("avg-jitter", "average-jitter")
                jitter = {
                    "available": True,
                    "lost": field("num-lost", "lost"),
                    "late": field("num-late", "late"),
                    "duplicates": field("num-duplicates", "duplicates"),
                    "average_jitter_ms": jitter_nanoseconds_to_ms(average_jitter),
                }
                break
            except Exception:
                continue

        return {
            **{f"{name}_interval_ms": stats(values) for name, values in self._stage_intervals.items()},
            "rtp_jitterbuffer": jitter,
            "map_copy_ms": stats(self._map_copy_ms),
            "pipeline_lag_ms": stats(self._pipeline_lag_ms),
            "current_pipeline_lag_ms": self._last_pipeline_lag_ms,
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
