from __future__ import annotations

"""Stage 10 diagnostic: proven six-camera wall with only process isolation added.

Parent process owns the native Qt QWidget/XID. A multiprocessing 'spawn' child
owns the exact media stack already proven by Stage 9:

  6x nvurisrcbin/NVDEC -> queues -> nvstreammux(batch=6)
  -> nvmultistreamtiler(2x3) -> nvvideoconvert -> NVMM RGBA
  -> nvdsosd -> nveglglessink -> parent Qt XID

No detector, tracker, analysis tee/branch, focus controller, or production runtime
is imported. This isolates cross-process XID/GStreamer ownership only.
"""

import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from services.ml_service.app.config import load_settings


def _set_if(element, name: str, value) -> bool:
    if element.find_property(name) is None:
        return False
    element.set_property(name, value)
    return True


def _request_pad(element, name: str):
    fn = getattr(element, "request_pad_simple", None)
    pad = fn(name) if fn is not None else None
    if pad is None:
        pad = element.get_request_pad(name)
    if pad is None:
        raise RuntimeError(f"{element.get_name()}: could not request {name}")
    return pad


def _put_latest(q, item) -> None:
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def _child_main(xid: int, stop_event, status_q) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import GLib, Gst, GstVideo

    Gst.init(None)
    settings = load_settings()
    count = 6
    if len(settings.cameras) < count:
        raise RuntimeError(f"stage10 requires 6 enabled cameras, got {len(settings.cameras)}")
    cameras = list(settings.cameras[:count])
    latency_ms = int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "250"))
    gpu_id = int(settings.deepstream.gpu_id)
    mux_width = int(os.environ.get("CAMERA_V2_STAGE10_MUX_WIDTH", "2560"))
    mux_height = int(os.environ.get("CAMERA_V2_STAGE10_MUX_HEIGHT", "1440"))
    columns = 2
    rows = int(math.ceil(count / columns))
    wall_width = 1600
    wall_height = 1350

    pipeline = Gst.Pipeline.new("camera-stage10-process-qt")
    if pipeline is None:
        raise RuntimeError("could not create stage10 pipeline")

    def make(factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"required GStreamer element unavailable: {factory}")
        return element

    mux = make("nvstreammux", "stage10_mux")
    tiler = make("nvmultistreamtiler", "stage10_tiler")
    convert = make("nvvideoconvert", "stage10_convert")
    capsfilter = make("capsfilter", "stage10_rgba_caps")
    osd = make("nvdsosd", "stage10_osd")
    sink = make("nveglglessink", "stage10_sink")

    mux.set_property("batch-size", count)
    _set_if(mux, "live-source", True)
    _set_if(mux, "batched-push-timeout", 50000)
    _set_if(mux, "width", mux_width)
    _set_if(mux, "height", mux_height)
    _set_if(mux, "enable-padding", False)
    _set_if(mux, "gpu-id", gpu_id)
    _set_if(mux, "nvbuf-memory-type", 2)
    _set_if(mux, "sync-inputs", False)
    _set_if(mux, "buffer-pool-size", 8)

    tiler.set_property("rows", rows)
    tiler.set_property("columns", columns)
    tiler.set_property("width", wall_width)
    tiler.set_property("height", wall_height)
    _set_if(tiler, "gpu-id", gpu_id)

    _set_if(convert, "gpu-id", gpu_id)
    _set_if(convert, "compute-hw", 1)
    capsfilter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
    )
    _set_if(osd, "process-mode", 1)
    _set_if(osd, "display-bbox", True)
    _set_if(osd, "display-text", False)
    _set_if(osd, "display-mask", False)
    _set_if(osd, "gpu-id", gpu_id)
    _set_if(sink, "sync", False)
    _set_if(sink, "async", False)
    _set_if(sink, "qos", False)
    _set_if(sink, "force-aspect-ratio", True)

    for element in (mux, tiler, convert, capsfilter, osd, sink):
        pipeline.add(element)
    for left, right, label in (
        (mux, tiler, "mux -> tiler"),
        (tiler, convert, "tiler -> convert"),
        (convert, capsfilter, "convert -> RGBA"),
        (capsfilter, osd, "RGBA -> OSD"),
        (osd, sink, "OSD -> EGL"),
    ):
        if not left.link(right):
            raise RuntimeError(f"stage10 {label} link failed")

    counters = {"mux": 0, "tiler": 0, "rgba": 0, "osd": 0, "sink": 0}
    source_counts = {camera.camera_id: 0 for camera in cameras}
    request_pads = []
    started = time.monotonic()

    def make_probe(name: str):
        def _probe(_pad, _info):
            counters[name] += 1
            return Gst.PadProbeReturn.OK
        return _probe

    mux.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("mux"))
    tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("tiler"))
    capsfilter.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("rgba"))
    osd.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("osd"))
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, make_probe("sink"))

    for index, camera in enumerate(cameras):
        source = make("nvurisrcbin", f"stage10_source_{index}")
        q = make("queue", f"stage10_queue_{index}")
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
        _set_if(q, "max-size-buffers", 2)
        _set_if(q, "max-size-bytes", 0)
        _set_if(q, "max-size-time", 0)
        _set_if(q, "leaky", 2)
        pipeline.add(source)
        pipeline.add(q)

        mux_sink = _request_pad(mux, f"sink_{index}")
        request_pads.append(mux_sink)
        qsrc = q.get_static_pad("src")
        if qsrc is None or qsrc.link(mux_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera.camera_id}: queue -> mux sink_{index} failed")

        def source_probe(_pad, _info, cid=camera.camera_id):
            source_counts[cid] += 1
            return Gst.PadProbeReturn.OK

        qsrc.add_probe(Gst.PadProbeType.BUFFER, source_probe)

        def configure_rtsp_child(_bin, _sub_bin, element, cam=camera) -> None:
            factory = element.get_factory()
            if factory is None or factory.get_name() != "rtspsrc":
                return
            if cam.username:
                _set_if(element, "user-id", cam.username)
                _set_if(element, "user-pw", cam.password)
            _set_if(element, "protocols", 4)
            _set_if(element, "latency", latency_ms)
            _set_if(element, "drop-on-latency", True)
            _set_if(element, "do-rtsp-keep-alive", True)
            print(
                f"STAGE10_RTSP camera={cam.camera_id} auth={'yes' if cam.username else 'no'} "
                f"transport=tcp latency={latency_ms}ms pid={os.getpid()}",
                flush=True,
            )

        source.connect("deep-element-added", configure_rtsp_child)

        def source_pad_added(_source, pad, qq=q, cam=camera) -> None:
            qsink = qq.get_static_pad("sink")
            if qsink is None or qsink.is_linked():
                return
            caps = pad.get_current_caps()
            if caps is None or caps.get_size() == 0:
                caps = pad.query_caps(None)
            caps_text = caps.to_string() if caps is not None else "pending"
            if caps is not None and caps.get_size() > 0 and not caps.is_any():
                media = caps.get_structure(0).get_name()
                if media and not media.startswith("video/"):
                    return
            result = pad.link(qsink)
            print(
                f"STAGE10_LINK camera={cam.camera_id} result={result} caps={caps_text} pid={os.getpid()}",
                flush=True,
            )

        source.connect("pad-added", source_pad_added)

    GstVideo.VideoOverlay.set_window_handle(sink, int(xid))
    try:
        GstVideo.VideoOverlay.handle_events(sink, False)
    except Exception:
        pass
    print(f"STAGE10_BIND event=preplay xid={xid} child_pid={os.getpid()}", flush=True)

    bus = pipeline.get_bus()

    def sync_handler(_bus, message, _data=None):
        try:
            prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
        except Exception:
            structure = message.get_structure()
            prepare = bool(structure and structure.get_name() == "prepare-window-handle")
        if not prepare:
            return Gst.BusSyncReply.PASS
        GstVideo.VideoOverlay.set_window_handle(message.src, int(xid))
        try:
            GstVideo.VideoOverlay.handle_events(message.src, False)
        except Exception:
            pass
        print(
            f"STAGE10_BIND event=prepare-window-handle xid={xid} child_pid={os.getpid()}",
            flush=True,
        )
        return Gst.BusSyncReply.DROP

    bus.set_sync_handler(sync_handler, None)
    bus.add_signal_watch()
    loop = GLib.MainLoop()

    def on_bus(_bus, message) -> None:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src is not None else "unknown"
            text = f"source={src} message={err.message} debug={debug or ''}"
            print(f"STAGE10_ERROR {text}", file=sys.stderr, flush=True)
            _put_latest(status_q, ("ERROR", text))
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("STAGE10_EOS", flush=True)
            _put_latest(status_q, ("EOS", ""))
            loop.quit()
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
            old, new, pending = message.parse_state_changed()
            print(
                f"STAGE10_STATE old={old.value_nick} new={new.value_nick} pending={pending.value_nick} pid={os.getpid()}",
                flush=True,
            )
            if new == Gst.State.PLAYING:
                _put_latest(status_q, ("PLAYING", os.getpid()))

    bus.connect("message", on_bus)

    def stats_tick() -> bool:
        elapsed = max(0.001, time.monotonic() - started)
        sources = " ".join(f"{cid}:{value}" for cid, value in source_counts.items())
        print(
            f"STAGE10_STATS sources=[{sources}] mux_buffers={counters['mux']} "
            f"tiler_buffers={counters['tiler']} rgba_buffers={counters['rgba']} "
            f"osd_buffers={counters['osd']} sink_buffers={counters['sink']} "
            f"fps={counters['sink'] / elapsed:.1f} xid={xid} child_pid={os.getpid()}",
            flush=True,
        )
        _put_latest(
            status_q,
            (
                "STATS",
                {
                    "sources": dict(source_counts),
                    "mux": counters["mux"],
                    "tiler": counters["tiler"],
                    "rgba": counters["rgba"],
                    "osd": counters["osd"],
                    "sink": counters["sink"],
                },
            ),
        )
        return True

    def stop_tick() -> bool:
        if stop_event.is_set():
            loop.quit()
            return False
        return True

    GLib.timeout_add_seconds(2, stats_tick)
    GLib.timeout_add(100, stop_tick)

    print(
        f"STAGE10_CHILD_START cameras=6 batch=6 layout=2x3 wall={wall_width}x{wall_height} "
        f"xid={xid} parent={os.getppid()} child={os.getpid()} detector=0 tracker=0 analysis=0",
        flush=True,
    )
    result = pipeline.set_state(Gst.State.PLAYING)
    if result == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("stage10 pipeline refused PLAYING")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        for pad in request_pads:
            try:
                mux.release_request_pad(pad)
            except Exception:
                pass
        print(
            f"STAGE10_CHILD_STOP sink_buffers={counters['sink']} child_pid={os.getpid()}",
            flush=True,
        )


class NativeVideoHost(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setStyleSheet("background:#000000;")

    def paintEngine(self):  # noqa: N802
        return None


class Stage10Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stage 10 - 6 cameras / process isolated Qt-XID")
        self.resize(1600, 900)
        self.host = NativeVideoHost()
        self.setCentralWidget(self.host)
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.status_q = self.ctx.Queue(maxsize=16)
        self.process = None
        self.xid = 0
        self._stopping = False
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(200)
        self.poll_timer.timeout.connect(self._poll_child)
        QTimer.singleShot(0, self._start_child)

    def _start_child(self) -> None:
        self.host.winId()
        QApplication.processEvents()
        xid = int(self.host.winId())
        if xid <= 0:
            raise RuntimeError("stage10 parent Qt native XID is invalid")
        self.xid = xid
        print(
            f"STAGE10_PARENT_XID xid={xid} parent_pid={os.getpid()} "
            f"platform={QApplication.platformName()} display={os.environ.get('DISPLAY', 'unset')}",
            flush=True,
        )
        self.process = self.ctx.Process(
            target=_child_main,
            args=(xid, self.stop_event, self.status_q),
            name="stage10-camera-wall",
            daemon=False,
        )
        self.process.start()
        print(
            f"STAGE10_PARENT_START parent_pid={os.getpid()} child_pid={self.process.pid} "
            "cameras=6 controller=spawn detector=0 tracker=0 analysis=0",
            flush=True,
        )
        self.poll_timer.start()

    def _poll_child(self) -> None:
        while True:
            try:
                state, detail = self.status_q.get_nowait()
            except queue.Empty:
                break
            print(f"STAGE10_PARENT_STATUS state={state} detail={detail}", flush=True)
        if self.process is not None and not self.process.is_alive() and not self._stopping:
            code = self.process.exitcode
            print(f"STAGE10_PARENT_CHILD_EXIT code={code}", flush=True)
            QApplication.quit()

    def _stop_child(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.poll_timer.stop()
        self.stop_event.set()
        process = self.process
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            print(f"STAGE10_PARENT_STOP child_exit={process.exitcode}", flush=True)
        self.process = None

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_child()
        super().closeEvent(event)


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    app = QApplication.instance() or QApplication(sys.argv)
    window = Stage10Window()
    window.show()

    def stop_signal(_signum, _frame) -> None:
        QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    rc = app.exec()
    window._stop_child()
    return int(rc)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
