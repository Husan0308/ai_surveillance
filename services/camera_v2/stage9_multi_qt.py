from __future__ import annotations

"""Stage 9 diagnostic: incrementally add cameras to the proven Stage 8 Qt/XID wall.

Use CAMERA_V2_STAGE9_COUNT=2 first, then 3..6 only after each lower count passes.
No detector, tracker, analysis branch, or multiprocessing controller is present.
"""

import math
import os
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


class Stage9Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stage 9 - Incremental multi-camera Qt wall")
        self.resize(1600, 900)
        self.host = NativeVideoHost()
        self.setCentralWidget(self.host)

        self.Gst = None
        self.GstVideo = None
        self.pipeline = None
        self.bus = None
        self.mux = None
        self.sink = None
        self.request_pads = []
        self.cameras = []
        self.xid = 0
        self.started = time.monotonic()
        self.counters = {
            "mux": 0,
            "tiler": 0,
            "rgba": 0,
            "osd": 0,
            "sink": 0,
        }
        self.source_counts: dict[str, int] = {}
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
        count = int(os.environ.get("CAMERA_V2_STAGE9_COUNT", "2"))
        if count < 2 or count > 6:
            raise RuntimeError("CAMERA_V2_STAGE9_COUNT must be 2..6")
        if len(settings.cameras) < count:
            raise RuntimeError(f"requested {count} cameras but only {len(settings.cameras)} are enabled")
        self.cameras = list(settings.cameras[:count])
        self.source_counts = {camera.camera_id: 0 for camera in self.cameras}

        latency_ms = int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "250"))
        gpu_id = int(settings.deepstream.gpu_id)
        mux_width = int(os.environ.get("CAMERA_V2_STAGE9_MUX_WIDTH", "2560"))
        mux_height = int(os.environ.get("CAMERA_V2_STAGE9_MUX_HEIGHT", "1440"))
        columns = 2
        rows = int(math.ceil(count / columns))
        # Match production tile geometry: each camera occupies 800x450.
        wall_width = columns * 800
        wall_height = rows * 450

        pipeline = Gst.Pipeline.new("camera-stage9-multi-qt")
        if pipeline is None:
            raise RuntimeError("could not create stage9 pipeline")

        def make(factory: str, name: str):
            element = Gst.ElementFactory.make(factory, name)
            if element is None:
                raise RuntimeError(f"required GStreamer element unavailable: {factory}")
            return element

        mux = make("nvstreammux", "stage9_mux")
        tiler = make("nvmultistreamtiler", "stage9_tiler")
        convert = make("nvvideoconvert", "stage9_convert")
        capsfilter = make("capsfilter", "stage9_rgba_caps")
        osd = make("nvdsosd", "stage9_osd")
        sink = make("nveglglessink", "stage9_sink")

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
                raise RuntimeError(f"stage9 {label} link failed")

        def make_probe(name: str):
            def _probe(_pad, _info):
                self.counters[name] += 1
                return Gst.PadProbeReturn.OK
            return _probe

        mux.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("mux"))
        tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("tiler"))
        capsfilter.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("rgba"))
        osd.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe("osd"))
        sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, make_probe("sink"))

        for index, camera in enumerate(self.cameras):
            source = make("nvurisrcbin", f"stage9_source_{index}")
            queue = make("queue", f"stage9_queue_{index}")
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
            pipeline.add(source)
            pipeline.add(queue)

            mux_sink = _request_pad(mux, f"sink_{index}")
            self.request_pads.append(mux_sink)
            qsrc = queue.get_static_pad("src")
            if qsrc is None or qsrc.link(mux_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"{camera.camera_id}: queue -> mux sink_{index} failed")

            def source_probe(_pad, _info, cid=camera.camera_id):
                self.source_counts[cid] += 1
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
                    f"STAGE9_RTSP camera={cam.camera_id} auth={'yes' if cam.username else 'no'} "
                    f"transport=tcp latency={latency_ms}ms",
                    flush=True,
                )

            source.connect("deep-element-added", configure_rtsp_child)

            def source_pad_added(_source, pad, q=queue, cam=camera) -> None:
                qsink = q.get_static_pad("sink")
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
                print(f"STAGE9_LINK camera={cam.camera_id} result={result} caps={caps_text}", flush=True)

            source.connect("pad-added", source_pad_added)

        self.host.winId()
        QApplication.processEvents()
        xid = int(self.host.winId())
        if xid <= 0:
            raise RuntimeError("stage9 Qt native XID is invalid")
        self.xid = xid
        print(
            f"STAGE9_XID xid={xid} platform={QApplication.platformName()} "
            f"display={os.environ.get('DISPLAY', 'unset')}",
            flush=True,
        )

        def bind_overlay(overlay, event: str) -> None:
            GstVideo.VideoOverlay.set_window_handle(overlay, xid)
            try:
                GstVideo.VideoOverlay.handle_events(overlay, False)
            except Exception:
                pass
            print(f"STAGE9_BIND event={event} xid={xid}", flush=True)

        bind_overlay(sink, "preplay")
        bus = pipeline.get_bus()

        def sync_handler(_bus, message, _data=None):
            try:
                prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
            except Exception:
                structure = message.get_structure()
                prepare = bool(structure and structure.get_name() == "prepare-window-handle")
            if not prepare:
                return Gst.BusSyncReply.PASS
            bind_overlay(message.src, "prepare-window-handle")
            return Gst.BusSyncReply.DROP

        bus.set_sync_handler(sync_handler, None)

        self.pipeline = pipeline
        self.bus = bus
        self.mux = mux
        self.sink = sink
        self.started = time.monotonic()

        ids = ",".join(camera.camera_id for camera in self.cameras)
        print(
            f"STAGE9_START count={count} cameras={ids} batch={count} layout={columns}x{rows} "
            f"wall={wall_width}x{wall_height} detector=0 tracker=0 controller=0 qt=1 xid=1",
            flush=True,
        )
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self._stop_pipeline()
            raise RuntimeError("stage9 pipeline refused PLAYING")
        self.bus_timer.start()
        self.stats_timer.start()

    def _poll_bus(self) -> None:
        if self.bus is None or self.Gst is None:
            return
        Gst = self.Gst
        while True:
            message = self.bus.timed_pop_filtered(
                0,
                Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.STATE_CHANGED,
            )
            if message is None:
                break
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                src = message.src.get_name() if message.src is not None else "unknown"
                print(f"STAGE9_ERROR source={src} message={err.message} debug={debug or ''}", file=sys.stderr, flush=True)
                QApplication.quit()
            elif message.type == Gst.MessageType.EOS:
                print("STAGE9_EOS", flush=True)
                QApplication.quit()
            elif message.type == Gst.MessageType.STATE_CHANGED and message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                print(f"STAGE9_STATE old={old.value_nick} new={new.value_nick} pending={pending.value_nick}", flush=True)

    def _print_stats(self) -> None:
        if self.pipeline is None:
            return
        elapsed = max(0.001, time.monotonic() - self.started)
        sources = " ".join(f"{cid}:{count}" for cid, count in self.source_counts.items())
        print(
            f"STAGE9_STATS sources=[{sources}] mux_buffers={self.counters['mux']} "
            f"tiler_buffers={self.counters['tiler']} rgba_buffers={self.counters['rgba']} "
            f"osd_buffers={self.counters['osd']} sink_buffers={self.counters['sink']} "
            f"fps={self.counters['sink'] / elapsed:.1f} xid={self.xid}",
            flush=True,
        )

    def _stop_pipeline(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.bus_timer.stop()
        self.stats_timer.stop()
        if self.pipeline is not None and self.Gst is not None:
            self.pipeline.set_state(self.Gst.State.NULL)
        if self.mux is not None:
            for pad in self.request_pads:
                try:
                    self.mux.release_request_pad(pad)
                except Exception:
                    pass
        self.request_pads.clear()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_pipeline()
        super().closeEvent(event)


def main() -> int:
    if os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication.instance() or QApplication([])
    window = Stage9Window()
    window.show()
    rc = app.exec()
    window._stop_pipeline()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
