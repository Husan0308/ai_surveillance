from __future__ import annotations

"""Stage 11 diagnostic: Stage 10 plus only a post-mux tee and two queues.

Parent owns Qt/XID. Spawn child owns six-camera GStreamer pipeline.
New compared with Stage 10:

    nvstreammux -> tee
      -> display queue -> proven tiler/RGBA/OSD/EGL chain
      -> analysis queue -> fakesink

No analysis tiler, conversion, appsink, detector or tracker is present. This
isolates tee/request-pad/branch scheduling and buffer release only.
"""

import math
import multiprocessing as mp
import os
import queue
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
        raise RuntimeError(f"stage11 requires 6 enabled cameras, got {len(settings.cameras)}")
    cameras = list(settings.cameras[:count])
    latency_ms = int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "250"))
    gpu_id = int(settings.deepstream.gpu_id)
    mux_width = int(os.environ.get("CAMERA_V2_STAGE11_MUX_WIDTH", "2560"))
    mux_height = int(os.environ.get("CAMERA_V2_STAGE11_MUX_HEIGHT", "1440"))
    columns = 2
    rows = int(math.ceil(count / columns))
    wall_width = 1600
    wall_height = 1350

    pipeline = Gst.Pipeline.new("camera-stage11-process-tee")
    if pipeline is None:
        raise RuntimeError("could not create stage11 pipeline")

    def make(factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"required GStreamer element unavailable: {factory}")
        return element

    mux = make("nvstreammux", "stage11_mux")
    tee = make("tee", "stage11_tee")
    display_q = make("queue", "stage11_display_q")
    analysis_q = make("queue", "stage11_analysis_q")
    analysis_sink = make("fakesink", "stage11_analysis_sink")
    tiler = make("nvmultistreamtiler", "stage11_tiler")
    convert = make("nvvideoconvert", "stage11_convert")
    capsfilter = make("capsfilter", "stage11_rgba_caps")
    osd = make("nvdsosd", "stage11_osd")
    sink = make("nveglglessink", "stage11_sink")

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

    for q in (display_q, analysis_q):
        _set_if(q, "max-size-buffers", 2 if q is display_q else 1)
        _set_if(q, "max-size-bytes", 0)
        _set_if(q, "max-size-time", 0)
        _set_if(q, "leaky", 2)
        _set_if(q, "silent", True)

    _set_if(analysis_sink, "sync", False)
    _set_if(analysis_sink, "async", False)
    _set_if(analysis_sink, "enable-last-sample", False)

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

    for element in (
        mux, tee, display_q, analysis_q, analysis_sink,
        tiler, convert, capsfilter, osd, sink,
    ):
        pipeline.add(element)

    if not mux.link(tee):
        raise RuntimeError("stage11 mux -> tee failed")
    tee_display = _request_pad(tee, "src_%u")
    tee_analysis = _request_pad(tee, "src_%u")
    if tee_display.link(display_q.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise RuntimeError("stage11 tee -> display_q failed")
    if tee_analysis.link(analysis_q.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise RuntimeError("stage11 tee -> analysis_q failed")
    if not display_q.link(tiler):
        raise RuntimeError("stage11 display_q -> tiler failed")
    if not analysis_q.link(analysis_sink):
        raise RuntimeError("stage11 analysis_q -> fakesink failed")
    for left, right, label in (
        (tiler, convert, "tiler -> convert"),
        (convert, capsfilter, "convert -> RGBA"),
        (capsfilter, osd, "RGBA -> OSD"),
        (osd, sink, "OSD -> EGL"),
    ):
        if not left.link(right):
            raise RuntimeError(f"stage11 {label} failed")

    counters = {
        "mux": 0,
        "tee": 0,
        "display_q": 0,
        "analysis_q": 0,
        "analysis_sink": 0,
        "tiler": 0,
        "rgba": 0,
        "osd": 0,
        "sink": 0,
    }
    source_counts = {camera.camera_id: 0 for camera in cameras}
    mux_request_pads = []
    started = time.monotonic()

    def probe(name: str):
        def _probe(_pad, _info):
            counters[name] += 1
            return Gst.PadProbeReturn.OK
        return _probe

    mux.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("mux"))
    tee.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("tee"))
    display_q.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("display_q"))
    analysis_q.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("analysis_q"))
    analysis_sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("analysis_sink"))
    tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("tiler"))
    capsfilter.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("rgba"))
    osd.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("osd"))
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("sink"))

    for index, camera in enumerate(cameras):
        source = make("nvurisrcbin", f"stage11_source_{index}")
        q = make("queue", f"stage11_source_q_{index}")
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
        mux_request_pads.append(mux_sink)
        qsrc = q.get_static_pad("src")
        if qsrc is None or qsrc.link(mux_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera.camera_id}: source queue -> mux sink_{index} failed")

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
                f"STAGE11_RTSP camera={cam.camera_id} auth={'yes' if cam.username else 'no'} "
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
                f"STAGE11_LINK camera={cam.camera_id} result={result} caps={caps_text} pid={os.getpid()}",
                flush=True,
            )

        source.connect("pad-added", source_pad_added)

    GstVideo.VideoOverlay.set_window_handle(sink, int(xid))
    try:
        GstVideo.VideoOverlay.handle_events(sink, False)
    except Exception:
        pass
    print(f"STAGE11_BIND event=preplay xid={xid} child_pid={os.getpid()}", flush=True)

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
            f"STAGE11_BIND event=prepare-window-handle xid={xid} child_pid={os.getpid()}",
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
            print(f"STAGE11_ERROR {text}", file=sys.stderr, flush=True)
            _put_latest(status_q, ("ERROR", text))
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("STAGE11_EOS", flush=True)
            _put_latest(status_q, ("EOS", ""))
            loop.quit()
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
            old, new, pending = message.parse_state_changed()
            print(
                f"STAGE11_STATE old={old.value_nick} new={new.value_nick} pending={pending.value_nick} pid={os.getpid()}",
                flush=True,
            )
            if new == Gst.State.PLAYING:
                _put_latest(status_q, ("PLAYING", os.getpid()))

    bus.connect("message", on_bus)

    def stats_tick() -> bool:
        elapsed = max(0.001, time.monotonic() - started)
        sources = " ".join(f"{cid}:{value}" for cid, value in source_counts.items())
        print(
            f"STAGE11_STATS sources=[{sources}] mux={counters['mux']} tee={counters['tee']} "
            f"display_q={counters['display_q']} analysis_q={counters['analysis_q']} "
            f"analysis_sink={counters['analysis_sink']} tiler={counters['tiler']} "
            f"rgba={counters['rgba']} osd={counters['osd']} sink={counters['sink']} "
            f"fps={counters['sink'] / elapsed:.1f} child_pid={os.getpid()}",
            flush=True,
        )
        _put_latest(
            status_q,
            (
                "STATS",
                {
                    "sources": dict(source_counts),
                    "mux": counters["mux"],
                    "tee": counters["tee"],
                    "display_q": counters["display_q"],
                    "analysis_q": counters["analysis_q"],
                    "analysis_sink": counters["analysis_sink"],
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
        f"STAGE11_CHILD_START cameras=6 batch=6 tee=1 display_branch=1 analysis_branch=fakesink "
        f"xid={xid} parent={os.getppid()} child={os.getpid()} detector=0 tracker=0 appsink=0",
        flush=True,
    )
    result = pipeline.set_state(Gst.State.PLAYING)
    if result == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("stage11 pipeline refused PLAYING")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        for pad in mux_request_pads:
            try:
                mux.release_request_pad(pad)
            except Exception:
                pass
        for pad in (tee_display, tee_analysis):
            try:
                tee.release_request_pad(pad)
            except Exception:
                pass
        print(
            f"STAGE11_CHILD_STOP mux={counters['mux']} analysis_sink={counters['analysis_sink']} sink={counters['sink']} child_pid={os.getpid()}",
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


class Stage11Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stage 11 - 6 cameras + post-mux tee")
        self.resize(1600, 900)
        self.host = NativeVideoHost()
        self.setCentralWidget(self.host)
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.status_q = self.ctx.Queue(maxsize=16)
        self.process = None
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(250)
        self.status_timer.timeout.connect(self._poll_status)
        QTimer.singleShot(0, self._start_child)

    def _start_child(self) -> None:
        self.host.winId()
        QApplication.processEvents()
        xid = int(self.host.winId())
        if xid <= 0:
            raise RuntimeError("stage11 parent Qt native XID is invalid")
        print(
            f"STAGE11_PARENT_XID xid={xid} parent_pid={os.getpid()} platform={QApplication.platformName()} display={os.environ.get('DISPLAY', 'unset')}",
            flush=True,
        )
        self.process = self.ctx.Process(
            target=_child_main,
            args=(xid, self.stop_event, self.status_q),
            name="stage11-camera-wall",
            daemon=False,
        )
        self.process.start()
        print(
            f"STAGE11_PARENT_START parent_pid={os.getpid()} child_pid={self.process.pid} cameras=6 tee=1 detector=0 tracker=0",
            flush=True,
        )
        self.status_timer.start()

    def _poll_status(self) -> None:
        while True:
            try:
                state, detail = self.status_q.get_nowait()
            except queue.Empty:
                break
            print(f"STAGE11_PARENT_STATUS state={state} detail={detail}", flush=True)
            if state in {"ERROR", "EOS"}:
                QApplication.quit()
        if self.process is not None and not self.process.is_alive():
            print(f"STAGE11_PARENT_CHILD_EXIT exitcode={self.process.exitcode}", flush=True)
            QApplication.quit()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.status_timer.stop()
        self.stop_event.set()
        process = self.process
        if process is not None:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        super().closeEvent(event)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = Stage11Window()
    window.show()
    rc = app.exec()
    window.close()
    return int(rc)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
