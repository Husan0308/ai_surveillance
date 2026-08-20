from __future__ import annotations

"""Stage 4 diagnostic: Stage 3 plus exactly one nvvideoconvert element.

Graph:
    nvurisrcbin/NVDEC -> queue -> nvstreammux(batch=1)
    -> nvmultistreamtiler(1x1) -> nvvideoconvert -> fakesink

No caps forcing, OSD, detector, tracker, display, EGL or Qt is present. This
proves only that the converter can consume and release the tiler's NVMM buffers.
"""

import os
import signal
import sys
import time

from services.ml_service.app.config import load_settings


def _set_if(element, name: str, value) -> bool:
    if element.find_property(name) is None:
        return False
    element.set_property(name, value)
    return True


def _request_pad(element, name: str):
    request_simple = getattr(element, "request_pad_simple", None)
    pad = request_simple(name) if request_simple is not None else None
    if pad is None:
        pad = element.get_request_pad(name)
    if pad is None:
        raise RuntimeError(f"{element.get_name()}: could not request {name}")
    return pad


def main() -> int:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import GLib, Gst

    Gst.init(None)
    settings = load_settings()
    requested_id = os.environ.get("CAMERA_V2_STAGE4_CAMERA", "CAM-01").strip()
    camera = next((c for c in settings.cameras if c.camera_id == requested_id), None)
    if camera is None:
        known = ", ".join(c.camera_id for c in settings.cameras)
        raise RuntimeError(f"unknown CAMERA_V2_STAGE4_CAMERA={requested_id!r}; known={known}")

    latency_ms = int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "250"))
    width = int(os.environ.get("CAMERA_V2_STAGE4_WIDTH", "2560"))
    height = int(os.environ.get("CAMERA_V2_STAGE4_HEIGHT", "1440"))
    gpu_id = int(settings.deepstream.gpu_id)

    pipeline = Gst.Pipeline.new("camera-stage4-convert")
    if pipeline is None:
        raise RuntimeError("could not create stage4 pipeline")

    def make(factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"required GStreamer element unavailable: {factory}")
        return element

    source = make("nvurisrcbin", "stage4_source")
    queue = make("queue", "stage4_queue")
    mux = make("nvstreammux", "stage4_mux")
    tiler = make("nvmultistreamtiler", "stage4_tiler")
    convert = make("nvvideoconvert", "stage4_convert")
    sink = make("fakesink", "stage4_sink")

    source.set_property("uri", camera.uri)
    _set_if(source, "disable-audio", True)
    _set_if(source, "select-rtp-protocol", 4)
    _set_if(source, "latency", latency_ms)
    _set_if(source, "drop-on-latency", True)
    _set_if(source, "num-extra-surfaces", 4)
    _set_if(source, "cudadec-memtype", 0)
    _set_if(source, "rtsp-reconnect-interval", 0)
    _set_if(source, "message-forward", True)
    _set_if(source, "async-handling", True)
    _set_if(source, "gpu-id", gpu_id)

    _set_if(queue, "max-size-buffers", 2)
    _set_if(queue, "max-size-bytes", 0)
    _set_if(queue, "max-size-time", 0)
    _set_if(queue, "leaky", 2)

    mux.set_property("batch-size", 1)
    _set_if(mux, "live-source", True)
    _set_if(mux, "batched-push-timeout", 50000)
    _set_if(mux, "width", width)
    _set_if(mux, "height", height)
    _set_if(mux, "enable-padding", False)
    _set_if(mux, "gpu-id", gpu_id)
    _set_if(mux, "nvbuf-memory-type", 2)
    _set_if(mux, "sync-inputs", False)

    tiler.set_property("rows", 1)
    tiler.set_property("columns", 1)
    tiler.set_property("width", width)
    tiler.set_property("height", height)
    _set_if(tiler, "gpu-id", gpu_id)

    _set_if(convert, "gpu-id", gpu_id)
    _set_if(convert, "compute-hw", 1)

    _set_if(sink, "sync", False)
    _set_if(sink, "async", False)
    _set_if(sink, "enable-last-sample", False)

    for element in (source, queue, mux, tiler, convert, sink):
        pipeline.add(element)

    mux_sink = _request_pad(mux, "sink_0")
    qsrc = queue.get_static_pad("src")
    if qsrc is None:
        raise RuntimeError("stage4 queue src pad missing")
    if qsrc.link(mux_sink) != Gst.PadLinkReturn.OK:
        raise RuntimeError("stage4 queue -> nvstreammux sink_0 link failed")
    if not mux.link(tiler):
        raise RuntimeError("stage4 nvstreammux -> tiler link failed")
    if not tiler.link(convert):
        raise RuntimeError("stage4 tiler -> nvvideoconvert link failed")
    if not convert.link(sink):
        raise RuntimeError("stage4 nvvideoconvert -> fakesink link failed")

    counters = {"source": 0, "mux": 0, "tiler": 0, "convert": 0, "sink": 0}
    started = time.monotonic()

    def probe(name):
        def _probe(_pad, _info):
            counters[name] += 1
            return Gst.PadProbeReturn.OK
        return _probe

    qsrc.add_probe(Gst.PadProbeType.BUFFER, probe("source"))
    mux.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("mux"))
    tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("tiler"))
    convert.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("convert"))
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("sink"))

    def configure_rtsp_child(_bin, _sub_bin, element) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if camera.username:
            _set_if(element, "user-id", camera.username)
            _set_if(element, "user-pw", camera.password)
        _set_if(element, "protocols", 4)
        _set_if(element, "latency", latency_ms)
        _set_if(element, "drop-on-latency", True)
        _set_if(element, "do-rtsp-keep-alive", True)
        print(f"STAGE4_RTSP camera={camera.camera_id} auth={'yes' if camera.username else 'no'} transport=tcp latency={latency_ms}ms", flush=True)

    source.connect("deep-element-added", configure_rtsp_child)

    def source_pad_added(_source, pad) -> None:
        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        caps = pad.get_current_caps()
        if caps is None or caps.get_size() == 0:
            caps = pad.query_caps(None)
        caps_text = caps.to_string() if caps is not None else "pending"
        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            media = caps.get_structure(0).get_name()
            if media and not media.startswith("video/"):
                return
        result = pad.link(sink_pad)
        print(f"STAGE4_LINK result={result} caps={caps_text}", flush=True)

    source.connect("pad-added", source_pad_added)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus(_bus, message) -> None:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src is not None else "unknown"
            print(f"STAGE4_ERROR source={src} message={err.message} debug={debug or ''}", file=sys.stderr, flush=True)
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("STAGE4_EOS", flush=True)
            loop.quit()
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
            old, new, pending = message.parse_state_changed()
            print(f"STAGE4_STATE old={old.value_nick} new={new.value_nick} pending={pending.value_nick}", flush=True)

    bus.connect("message", on_bus)

    def stats_tick() -> bool:
        elapsed = max(0.001, time.monotonic() - started)
        print(
            f"STAGE4_STATS camera={camera.camera_id} source_frames={counters['source']} "
            f"mux_buffers={counters['mux']} tiler_buffers={counters['tiler']} "
            f"convert_buffers={counters['convert']} sink_buffers={counters['sink']} "
            f"fps={counters['sink'] / elapsed:.1f} batch=1 layout=1x1 size={width}x{height}",
            flush=True,
        )
        return True

    GLib.timeout_add_seconds(2, stats_tick)

    def stop_handler(_signum, _frame) -> None:
        loop.quit()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    print(
        f"STAGE4_START camera={camera.camera_id} graph=nvurisrcbin/NVDEC->queue->nvstreammux(batch=1)->"
        f"nvmultistreamtiler(1x1,{width}x{height})->nvvideoconvert->fakesink "
        "capsforce=0 osd=0 detector=0 tracker=0 display=0 qt=0",
        flush=True,
    )

    result = pipeline.set_state(Gst.State.PLAYING)
    if result == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        mux.release_request_pad(mux_sink)
        raise RuntimeError("stage4 pipeline refused PLAYING")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        try:
            mux.release_request_pad(mux_sink)
        except Exception:
            pass

    print(
        f"STAGE4_STOP source_frames={counters['source']} mux_buffers={counters['mux']} "
        f"tiler_buffers={counters['tiler']} convert_buffers={counters['convert']} sink_buffers={counters['sink']}",
        flush=True,
    )
    return 0 if counters["convert"] > 0 and counters["sink"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
