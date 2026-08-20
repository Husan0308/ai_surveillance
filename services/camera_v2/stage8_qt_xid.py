from __future__ import annotations

"""Stage 8 diagnostic: Stage 7 plus exactly one Qt native XID embedding layer.

Graph:
    nvurisrcbin/NVDEC -> queue -> nvstreammux(batch=1)
    -> nvmultistreamtiler(1x1) -> nvvideoconvert
    -> video/x-raw(memory:NVMM),format=RGBA -> nvdsosd
    -> nveglglessink rendered into one native PySide6 QWidget XID

No detector, tracker, multiprocessing controller, 6-camera wall, or production UI
is present. This isolates GstVideoOverlay/Qt XID binding from every later feature.
"""

import os
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
    request_simple = getattr(element, "request_pad_simple", None)
    pad = request_simple(name) if request_simple is not None else None
    if pad is None:
        pad = element.get_request_pad(name)
    if pad is None:
        raise RuntimeError(f"{element.get_name()}: could not request {name}")
    return pad


class NativeVideoHost(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("stage8VideoHost")
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setStyleSheet("background:#000000;")

    def paintEngine(self):  # noqa: N802 - Qt virtual method
        return None


class Stage8Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stage 8 - CAM-01 Qt/XID EGL")
        self.resize(1280, 720)
        self.host = NativeVideoHost()
        self.setCentralWidget(self.host)

        self.pipeline = None
        self.Gst = None
        self.GstVideo = None
        self.mux = None
        self.mux_sink = None
        self.bus = None
        self.sink = None
        self.camera = None
        self.started = time.monotonic()
        self.counters = {
            "source": 0,
            "mux": 0,
            "tiler": 0,
            "rgba": 0,
            "osd": 0,
            "sink": 0,
        }
        self.xid = 0
        self._stopping = False

        self.bus_timer = QTimer(self)
        self.bus_timer.setInterval(50)
        self.bus_timer.timeout.connect(self._poll_bus)
        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(2000)
        self.stats_timer.timeout.connect(self._print_stats)

        QTimer.singleShot(0, self._start_pipeline)

    def _start_pipeline(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        Gst.init(None)
        self.Gst = Gst
        self.GstVideo = GstVideo

        settings = load_settings()
        requested_id = os.environ.get("CAMERA_V2_STAGE8_CAMERA", "CAM-01").strip()
        camera = next((c for c in settings.cameras if c.camera_id == requested_id), None)
        if camera is None:
            known = ", ".join(c.camera_id for c in settings.cameras)
            raise RuntimeError(
                f"unknown CAMERA_V2_STAGE8_CAMERA={requested_id!r}; known={known}"
            )
        self.camera = camera

        latency_ms = int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "250"))
        width = int(os.environ.get("CAMERA_V2_STAGE8_WIDTH", "2560"))
        height = int(os.environ.get("CAMERA_V2_STAGE8_HEIGHT", "1440"))
        gpu_id = int(settings.deepstream.gpu_id)

        pipeline = Gst.Pipeline.new("camera-stage8-qt-xid")
        if pipeline is None:
            raise RuntimeError("could not create stage8 pipeline")

        def make(factory: str, name: str):
            element = Gst.ElementFactory.make(factory, name)
            if element is None:
                raise RuntimeError(f"required GStreamer element unavailable: {factory}")
            return element

        source = make("nvurisrcbin", "stage8_source")
        queue = make("queue", "stage8_queue")
        mux = make("nvstreammux", "stage8_mux")
        tiler = make("nvmultistreamtiler", "stage8_tiler")
        convert = make("nvvideoconvert", "stage8_convert")
        capsfilter = make("capsfilter", "stage8_rgba_caps")
        osd = make("nvdsosd", "stage8_osd")
        sink = make("nveglglessink", "stage8_sink")

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

        for element in (source, queue, mux, tiler, convert, capsfilter, osd, sink):
            pipeline.add(element)

        mux_sink = _request_pad(mux, "sink_0")
        qsrc = queue.get_static_pad("src")
        if qsrc is None:
            raise RuntimeError("stage8 queue src pad missing")
        if qsrc.link(mux_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("stage8 queue -> nvstreammux sink_0 link failed")
        for left, right, label in (
            (mux, tiler, "mux -> tiler"),
            (tiler, convert, "tiler -> nvvideoconvert"),
            (convert, capsfilter, "convert -> RGBA caps"),
            (capsfilter, osd, "RGBA caps -> nvdsosd"),
            (osd, sink, "nvdsosd -> nveglglessink"),
        ):
            if not left.link(right):
                raise RuntimeError(f"stage8 {label} link failed")

        def probe(name):
            def _probe(_pad, _info):
                self.counters[name] += 1
                return Gst.PadProbeReturn.OK

            return _probe

        qsrc.add_probe(Gst.PadProbeType.BUFFER, probe("source"))
        mux.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("mux"))
        tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("tiler"))
        capsfilter.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("rgba"))
        osd.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("osd"))
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
            print(
                f"STAGE8_RTSP camera={camera.camera_id} "
                f"auth={'yes' if camera.username else 'no'} "
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
            print(f"STAGE8_LINK result={result} caps={caps_text}", flush=True)

        source.connect("pad-added", source_pad_added)

        # Force a real native QWidget before touching GstVideoOverlay.
        self.host.winId()
        QApplication.processEvents()
        xid = int(self.host.winId())
        if xid <= 0:
            raise RuntimeError("stage8 Qt native XID is invalid")
        self.xid = xid
        print(
            f"STAGE8_XID xid={xid} platform={QApplication.platformName()} "
            f"display={os.environ.get('DISPLAY', 'unset')}",
            flush=True,
        )

        def bind_overlay(overlay, event: str) -> None:
            GstVideo.VideoOverlay.set_window_handle(overlay, xid)
            try:
                GstVideo.VideoOverlay.handle_events(overlay, False)
            except Exception:
                pass
            print(f"STAGE8_BIND event={event} xid={xid}", flush=True)

        # GStreamer recommends setting a known native handle before PLAYING.
        bind_overlay(sink, "preplay")

        bus = pipeline.get_bus()

        def on_sync_message(_bus, message, _data=None):
            try:
                prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
            except Exception:
                structure = message.get_structure()
                prepare = bool(
                    structure and structure.get_name() == "prepare-window-handle"
                )
            if not prepare:
                return Gst.BusSyncReply.PASS
            try:
                bind_overlay(message.src, "prepare-window-handle")
                return Gst.BusSyncReply.DROP
            except Exception as exc:
                print(
                    f"STAGE8_BIND_ERROR {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return Gst.BusSyncReply.PASS

        bus.set_sync_handler(on_sync_message, None)

        self.pipeline = pipeline
        self.mux = mux
        self.mux_sink = mux_sink
        self.bus = bus
        self.sink = sink
        self.started = time.monotonic()

        print(
            f"STAGE8_START camera={camera.camera_id} graph=nvurisrcbin/NVDEC->queue->"
            f"nvstreammux(batch=1)->nvmultistreamtiler(1x1,{width}x{height})->"
            "nvvideoconvert->NVMM-RGBA->nvdsosd->nveglglessink->Qt-XID "
            "detector=0 tracker=0 cameras=1 controller=0",
            flush=True,
        )

        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self._stop_pipeline()
            raise RuntimeError("stage8 pipeline refused PLAYING")

        self.bus_timer.start()
        self.stats_timer.start()

    def _poll_bus(self) -> None:
        if self.bus is None or self.Gst is None:
            return
        Gst = self.Gst
        while True:
            message = self.bus.timed_pop_filtered(
                0,
                Gst.MessageType.ERROR
                | Gst.MessageType.EOS
                | Gst.MessageType.STATE_CHANGED,
            )
            if message is None:
                break
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                src = message.src.get_name() if message.src is not None else "unknown"
                print(
                    f"STAGE8_ERROR source={src} message={err.message} debug={debug or ''}",
                    file=sys.stderr,
                    flush=True,
                )
                QApplication.quit()
            elif message.type == Gst.MessageType.EOS:
                print("STAGE8_EOS", flush=True)
                QApplication.quit()
            elif message.type == Gst.MessageType.STATE_CHANGED and message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                print(
                    f"STAGE8_STATE old={old.value_nick} new={new.value_nick} "
                    f"pending={pending.value_nick}",
                    flush=True,
                )

    def _print_stats(self) -> None:
        if self.pipeline is None or self.camera is None:
            return
        elapsed = max(0.001, time.monotonic() - self.started)
        print(
            f"STAGE8_STATS camera={self.camera.camera_id} "
            f"source_frames={self.counters['source']} mux_buffers={self.counters['mux']} "
            f"tiler_buffers={self.counters['tiler']} rgba_buffers={self.counters['rgba']} "
            f"osd_buffers={self.counters['osd']} sink_buffers={self.counters['sink']} "
            f"fps={self.counters['sink'] / elapsed:.1f} xid={self.xid} "
            f"platform={QApplication.platformName()}",
            flush=True,
        )

    def _stop_pipeline(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.bus_timer.stop()
        self.stats_timer.stop()
        if self.pipeline is not None and self.Gst is not None:
            try:
                self.pipeline.set_state(self.Gst.State.NULL)
            except Exception:
                pass
        if self.mux is not None and self.mux_sink is not None:
            try:
                self.mux.release_request_pad(self.mux_sink)
            except Exception:
                pass
        print(
            "STAGE8_STOP "
            f"source_frames={self.counters['source']} mux_buffers={self.counters['mux']} "
            f"tiler_buffers={self.counters['tiler']} osd_buffers={self.counters['osd']} "
            f"sink_buffers={self.counters['sink']} xid={self.xid}",
            flush=True,
        )
        self.pipeline = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt virtual method
        self._stop_pipeline()
        super().closeEvent(event)


def main() -> int:
    if os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Sentinel Stage 8 Qt XID")
    window = Stage8Window()
    window.show()

    def stop_handler(_signum, _frame) -> None:
        QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    rc = app.exec()
    window._stop_pipeline()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
