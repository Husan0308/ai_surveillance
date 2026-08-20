from __future__ import annotations

"""Stage 1 diagnostic: one RTSP camera, no mux, no detector, no Qt.

This module intentionally does not import the production wall runtime. Its only
job is to prove the smallest useful media path on the target machine:

    nvurisrcbin/NVDEC -> queue -> nvvideoconvert -> sink

Use CAMERA_V2_STAGE1_SINK=fake to prove decode/buffer flow without any window
system dependency, then CAMERA_V2_STAGE1_SINK=x11 to prove visible X11 output.
Nothing from this file is imported by the production launcher.
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


def main() -> int:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import GLib, Gst

    Gst.init(None)
    settings = load_settings()
    requested_id = os.environ.get("CAMERA_V2_STAGE1_CAMERA", "CAM-01").strip()
    camera = next((c for c in settings.cameras if c.camera_id == requested_id), None)
    if camera is None:
        known = ", ".join(c.camera_id for c in settings.cameras)
        raise RuntimeError(f"unknown CAMERA_V2_STAGE1_CAMERA={requested_id!r}; known={known}")

    sink_mode = os.environ.get("CAMERA_V2_STAGE1_SINK", "fake").strip().lower()
    if sink_mode not in {"fake", "x11"}:
        raise RuntimeError("CAMERA_V2_STAGE1_SINK must be fake or x11")

    latency_ms = int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "250"))
    pipeline = Gst.Pipeline.new("camera-stage1")
    if pipeline is None:
        raise RuntimeError("could not create stage1 pipeline")

    def make(factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"required GStreamer element unavailable: {factory}")
        return element

    source = make("nvurisrcbin", "stage1_source")
    queue = make("queue", "stage1_queue")
    convert = make("nvvideoconvert", "stage1_convert")
    capsfilter = make("capsfilter", "stage1_caps")
    sink = make("fakesink" if sink_mode == "fake" else "ximagesink", "stage1_sink")

    source.set_property("uri", camera.uri)
    _set_if(source, "disable-audio", True)
    _set_if(source, "select-rtp-protocol", 4)  # RTSP/RTP interleaved TCP only
    _set_if(source, "latency", latency_ms)
    _set_if(source, "drop-on-latency", True)
    _set_if(source, "num-extra-surfaces", 4)
    _set_if(source, "cudadec-memtype", 0)
    _set_if(source, "rtsp-reconnect-interval", 0)  # fail once; no reconnect noise in diagnosis
    _set_if(source, "message-forward", True)
    _set_if(source, "async-handling", True)
    _set_if(source, "gpu-id", int(settings.deepstream.gpu_id))

    _set_if(queue, "max-size-buffers", 2)
    _set_if(queue, "max-size-bytes", 0)
    _set_if(queue, "max-size-time", 0)
    _set_if(queue, "leaky", 2)

    _set_if(convert, "gpu-id", int(settings.deepstream.gpu_id))
    if sink_mode == "fake":
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
        )
        _set_if(sink, "sync", False)
        _set_if(sink, "async", False)
    else:
        # System-memory BGRx + ximagesink is deliberate for AnyDesk/X11 diagnosis.
        # EGL/Qt is introduced only in later stages after this path is proven.
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRx"))
        _set_if(sink, "sync", False)
        _set_if(sink, "async", False)
        _set_if(sink, "force-aspect-ratio", True)

    for element in (source, queue, convert, capsfilter, sink):
        pipeline.add(element)
    if not queue.link(convert):
        raise RuntimeError("stage1 queue -> nvvideoconvert link failed")
    if not convert.link(capsfilter):
        raise RuntimeError("stage1 nvvideoconvert -> caps link failed")
    if not capsfilter.link(sink):
        raise RuntimeError("stage1 caps -> sink link failed")

    counters = {"source": 0, "sink": 0}
    started = time.monotonic()
    last_caps = {"text": "pending"}

    def probe_source(_pad, _info):
        counters["source"] += 1
        return Gst.PadProbeReturn.OK

    def probe_sink(_pad, _info):
        counters["sink"] += 1
        return Gst.PadProbeReturn.OK

    queue.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe_source)
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe_sink)

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
        print(
            f"STAGE1_RTSP camera={camera.camera_id} auth={'yes' if camera.username else 'no'} "
            f"transport=tcp latency={latency_ms}ms",
            flush=True,
        )

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
        print(f"STAGE1_LINK result={result} caps={caps_text}", flush=True)
        if result != Gst.PadLinkReturn.OK:
            return
        try:
            current = pad.get_current_caps()
            if current is not None:
                last_caps["text"] = current.to_string()
        except Exception:
            pass

    source.connect("pad-added", source_pad_added)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus(_bus, message) -> None:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src is not None else "unknown"
            print(
                f"STAGE1_ERROR source={src} message={err.message} debug={debug or ''}",
                file=sys.stderr,
                flush=True,
            )
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("STAGE1_EOS", flush=True)
            loop.quit()
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
            old, new, pending = message.parse_state_changed()
            print(
                f"STAGE1_STATE old={old.value_nick} new={new.value_nick} pending={pending.value_nick}",
                flush=True,
            )

    bus.connect("message", on_bus)

    def stats_tick() -> bool:
        elapsed = max(0.001, time.monotonic() - started)
        print(
            f"STAGE1_STATS camera={camera.camera_id} sink={sink_mode} "
            f"source_frames={counters['source']} sink_buffers={counters['sink']} "
            f"avg_fps={counters['sink'] / elapsed:.1f} caps={last_caps['text']}",
            flush=True,
        )
        return True

    GLib.timeout_add_seconds(2, stats_tick)

    def stop_handler(_signum, _frame) -> None:
        loop.quit()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    print(
        f"STAGE1_START camera={camera.camera_id} sink={sink_mode} "
        "graph=nvurisrcbin/NVDEC->queue->nvvideoconvert->sink "
        "mux=0 tiler=0 osd=0 detector=0 tracker=0 qt=0",
        flush=True,
    )
    result = pipeline.set_state(Gst.State.PLAYING)
    if result == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("stage1 pipeline refused PLAYING")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()

    print(
        f"STAGE1_STOP source_frames={counters['source']} sink_buffers={counters['sink']}",
        flush=True,
    )
    return 0 if counters["sink"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
