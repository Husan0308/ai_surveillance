from __future__ import annotations

import ctypes
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path

from services.camera_v11.step1_independent_egl_v4 import X11Wall
from services.ml_service.app.config import CameraConfig, load_settings


class _DetectionCounter:
    def __init__(self, path: str) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise RuntimeError(f"DeepStream YOLO metadata helper missing: {self.path}")
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v11_count_person_detections.argtypes = [ctypes.c_uint64]
        self.lib.camera_v11_count_person_detections.restype = ctypes.c_int

    def count(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v11_count_person_detections(
                ctypes.c_uint64(hash(gst_buffer))
            )
        )


class V11DeepStreamYoloCam01V1:
    """First incremental DeepStream detector milestone: CAM-01 only.

    One RTSP source feeds one batch-1 nvstreammux and one primary Gst-nvinfer.
    The detector is therefore in the same DeepStream/GStreamer pipeline as decode
    and OSD; there is no sidecar TensorRT worker, SHM frame copy, OpenCV detector,
    tracker, ReID, face model or UI service in this milestone.
    """

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import GLib, Gst, GstVideo

        Gst.init(None)
        self.GLib = GLib
        self.Gst = Gst
        self.GstVideo = GstVideo

        settings = load_settings()
        camera_id = os.environ.get("V11_DS_YOLO_CAMERA", "CAM-01").strip()
        matches = [camera for camera in settings.cameras if camera.camera_id == camera_id]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one configured {camera_id}, got {len(matches)}")
        self.camera: CameraConfig = matches[0]
        self.camera_id = self.camera.camera_id
        self.gpu_id = int(os.environ.get("V11_GPU_ID", settings.deepstream.gpu_id))
        self.transport = os.environ.get("V11_RTSP_TRANSPORT", "tcp").strip().lower()
        if self.transport != "tcp":
            raise RuntimeError("CAM-01 DeepStream YOLO v1 intentionally supports TCP only")
        self.latency_ms = max(20, int(os.environ.get("V11_RTSP_LATENCY_MS", "100")))
        self.drop_on_latency = self._env_bool("V11_DROP_ON_LATENCY", True)
        self.extra_surfaces = max(1, min(12, int(os.environ.get("V11_EXTRA_SURFACES", "4"))))
        self.udp_buffer_size = max(
            1_048_576,
            int(os.environ.get("V11_UDP_BUFFER_SIZE", str(max(settings.deepstream.udp_buffer_size, 8 * 1024 * 1024)))),
        )
        self.reconnect_sec = max(2, int(os.environ.get("V11_RECONNECT_SEC", "5")))
        self.width = max(320, int(os.environ.get("V11_DS_YOLO_DISPLAY_WIDTH", "640")))
        self.height = max(180, int(os.environ.get("V11_DS_YOLO_DISPLAY_HEIGHT", "360")))
        self.interval = max(0, int(os.environ.get("V11_DS_YOLO_INTERVAL", "9")))
        self.stats_interval = max(2, int(os.environ.get("V11_STATS_INTERVAL_SEC", "5")))
        self.config_path = Path(os.environ["V11_DS_YOLO_CONFIG"]).expanduser().resolve()
        self.counter = _DetectionCounter(os.environ["V11_DS_YOLO_META_LIB"])

        self.lock = threading.RLock()
        self.decoded = 0
        self.rendered = 0
        self.decoded_last = 0
        self.rendered_last = 0
        self.last_stat = time.monotonic()
        self.infer_buffers = 0
        self.positive_buffers = 0
        self.detections_total = 0
        self.max_objects = 0
        self.counter_errors = 0
        self.errors = 0
        self.warnings = 0
        self.render_gaps_ms: deque[float] = deque(maxlen=4096)
        self.last_render: float | None = None
        self._stopping = False

        self._preflight()
        self.wall = X11Wall(self.width, self.height, self.width, self.height, 1)
        self.loop = GLib.MainLoop()
        self.pipeline = self._build_pipeline()

        print(
            "CAMERA_V11_DS_YOLO_CAM01_ARCH "
            f"camera={self.camera_id} rtsp=1 decode=nvurisrcbin mux=batch1 "
            "detector=gst-nvinfer yolo=yolo26 nms=model-output parser=custom "
            "osd=nvdsosd sidecar_detector=0 shm_frame_copy=0 opencv=0 tracker=0 "
            "reid=0 face=0 ui=0",
            flush=True,
        )
        print(
            "CAMERA_V11_DS_YOLO_CAM01_POLICY "
            f"transport=tcp latency_ms={self.latency_ms} drop_on_latency={int(self.drop_on_latency)} "
            f"display={self.width}x{self.height} interval={self.interval} latest_queue=1 "
            "nvinfer_aspect=maintain+symmetric-padding precision=fp32",
            flush=True,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _preflight(self) -> None:
        if not self.config_path.is_file():
            raise RuntimeError(f"nvinfer config missing: {self.config_path}")
        required = (
            "nvurisrcbin",
            "queue",
            "nvstreammux",
            "nvinfer",
            "nvvideoconvert",
            "capsfilter",
            "nvdsosd",
            "nveglglessink",
            "rtspsrc",
        )
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("missing DeepStream/GStreamer plugins: " + ", ".join(missing))

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"could not create {factory}:{name}")
        return element

    @staticmethod
    def _set_if(element, prop: str, value) -> bool:
        if element.find_property(prop) is None:
            return False
        element.set_property(prop, value)
        return True

    def _require_link(self, src, dst, label: str) -> None:
        if not src.link(dst):
            raise RuntimeError(f"DeepStream YOLO CAM-01 link failed: {label}")

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera: CameraConfig) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)
        self._set_if(element, "protocols", 4)
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", self.drop_on_latency)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "do-rtsp-keep-alive", True)

    def _build_pipeline(self):
        safe = self.camera_id.lower().replace("-", "_")
        pipeline = self.Gst.Pipeline.new(f"v11_ds_yolo_{safe}")
        if pipeline is None:
            raise RuntimeError("could not create DeepStream YOLO CAM-01 pipeline")

        source = self._make("nvurisrcbin", f"source_{safe}")
        queue = self._make("queue", f"latest_{safe}")
        mux = self._make("nvstreammux", f"mux_{safe}")
        infer = self._make("nvinfer", f"pgie_{safe}")
        convert = self._make("nvvideoconvert", f"rgba_{safe}")
        capsfilter = self._make("capsfilter", f"caps_{safe}")
        osd = self._make("nvdsosd", f"osd_{safe}")
        sink = self._make("nveglglessink", f"sink_{safe}")

        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

        source.connect("deep-element-added", self._configure_rtsp_child, self.camera)
        source.connect("pad-added", self._on_source_pad_added)
        source.set_property("uri", self.camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "gpu-id", self.gpu_id)
        self._set_if(source, "latency", self.latency_ms)
        self._set_if(source, "drop-on-latency", self.drop_on_latency)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "select-rtp-protocol", 4)
        self._set_if(source, "rtsp-reconnect-interval", self.reconnect_sec)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)

        self._set_if(mux, "batch-size", 1)
        self._set_if(mux, "live-source", True)
        self._set_if(mux, "width", self.width)
        self._set_if(mux, "height", self.height)
        self._set_if(mux, "enable-padding", False)
        self._set_if(mux, "batched-push-timeout", 40_000)
        self._set_if(mux, "sync-inputs", False)
        self._set_if(mux, "buffer-pool-size", 4)
        self._set_if(mux, "nvbuf-memory-type", 0)
        self._set_if(mux, "gpu-id", self.gpu_id)

        infer.set_property("config-file-path", str(self.config_path))
        if infer.find_property("interval") is not None:
            infer.set_property("interval", self.interval)

        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "nvbuf-memory-type", 0)
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=RGBA,width={self.width},height={self.height}"
            ),
        )
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)

        self._set_if(sink, "sync", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "enable-last-sample", False)
        self._set_if(sink, "force-aspect-ratio", False)

        for element in (source, queue, mux, infer, convert, capsfilter, osd, sink):
            pipeline.add(element)

        queue_src = queue.get_static_pad("src")
        mux_sink = mux.request_pad_simple("sink_0") if hasattr(mux, "request_pad_simple") else None
        if mux_sink is None:
            mux_sink = mux.get_request_pad("sink_0")
        if queue_src is None or mux_sink is None:
            raise RuntimeError("queue/mux pad missing")
        if queue_src.link(mux_sink) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("queue->nvstreammux request-pad link failed")

        self._require_link(mux, infer, "mux->nvinfer")
        self._require_link(infer, convert, "nvinfer->rgba")
        self._require_link(convert, capsfilter, "rgba->caps")
        self._require_link(capsfilter, osd, "caps->osd")
        self._require_link(osd, sink, "osd->egl")

        infer_src = infer.get_static_pad("src")
        sink_pad = sink.get_static_pad("sink")
        if infer_src is None or sink_pad is None:
            raise RuntimeError("nvinfer/sink probe pad missing")
        infer_src.add_probe(self.Gst.PadProbeType.BUFFER, self._infer_probe)
        sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe)

        self.GstVideo.VideoOverlay.set_window_handle(sink, int(self.wall.children[0]))
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.source = source
        self.queue = queue
        self.mux = mux
        self.mux_sink = mux_sink
        self.infer = infer
        self.sink = sink
        return pipeline

    def _on_source_pad_added(self, _source, pad) -> None:
        sink_pad = self.queue.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"source->latest queue link failed: {result}")
        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe)
        caps = pad.get_current_caps()
        print(
            "CAMERA_V11_DS_YOLO_CAM01_LINK "
            f"camera={self.camera_id} status=OK caps={caps.to_string() if caps else 'pending'}",
            flush=True,
        )

    def _source_probe(self, _pad, info):
        if info.get_buffer() is not None:
            with self.lock:
                self.decoded += 1
        return self.Gst.PadProbeReturn.OK

    def _infer_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            count = self.counter.count(buffer)
            if count < 0:
                raise RuntimeError(f"metadata_count={count}")
            with self.lock:
                self.infer_buffers += 1
                self.detections_total += count
                self.max_objects = max(self.max_objects, count)
                if count > 0:
                    self.positive_buffers += 1
        except Exception as exc:
            with self.lock:
                self.counter_errors += 1
                errors = self.counter_errors
            if errors <= 3 or errors % 100 == 0:
                print(
                    f"CAMERA_V11_DS_YOLO_CAM01_META warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )
        return self.Gst.PadProbeReturn.OK

    def _render_probe(self, _pad, info):
        if info.get_buffer() is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.lock:
            self.rendered += 1
            if self.last_render is not None:
                gap = (now - self.last_render) * 1000.0
                if 0.0 <= gap <= 5000.0:
                    self.render_gaps_ms.append(gap)
            self.last_render = now
        return self.Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus, message) -> None:
        if message.type not in {self.Gst.MessageType.ERROR, self.Gst.MessageType.WARNING}:
            return
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            kind = "ERROR"
            with self.lock:
                self.errors += 1
        else:
            err, debug = message.parse_warning()
            kind = "WARNING"
            with self.lock:
                self.warnings += 1
        source = message.src.get_name() if message.src is not None else "unknown"
        print(
            f"CAMERA_V11_DS_YOLO_CAM01_{kind} source={source} message={err} debug={debug}",
            flush=True,
        )
        if kind == "ERROR":
            self.GLib.idle_add(self.stop)

    @staticmethod
    def _pct(values: deque[float], fraction: float) -> float:
        rows = sorted(values)
        if not rows:
            return 0.0
        index = min(len(rows) - 1, int(round((len(rows) - 1) * fraction)))
        return float(rows[index])

    def _print_stats(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        with self.lock:
            elapsed = max(0.001, now - self.last_stat)
            source_fps = (self.decoded - self.decoded_last) / elapsed
            render_fps = (self.rendered - self.rendered_last) / elapsed
            self.decoded_last = self.decoded
            self.rendered_last = self.rendered
            self.last_stat = now
            infer_buffers = self.infer_buffers
            positive = self.positive_buffers
            detections = self.detections_total
            max_objects = self.max_objects
            counter_errors = self.counter_errors
            errors = self.errors
            warnings = self.warnings
            p95 = self._pct(self.render_gaps_ms, 0.95)
        queue_level = int(self.queue.get_property("current-level-buffers"))
        print(
            "CAMERA_V11_DS_YOLO_CAM01 "
            f"camera={self.camera_id} source_fps={source_fps:.2f} render_fps={render_fps:.2f} "
            f"render_gap_p95={p95:.1f}ms queue={queue_level} infer_buffers={infer_buffers} "
            f"positive_buffers={positive} detections_total={detections} max_objects={max_objects} "
            f"meta_errors={counter_errors} errors={errors} warnings={warnings}",
            flush=True,
        )
        return True

    def stop(self) -> bool:
        if self._stopping:
            return False
        self._stopping = True
        try:
            self.loop.quit()
        except Exception:
            pass
        return False

    def run(self) -> int:
        def _handle(_signum, _frame):
            try:
                self.GLib.idle_add(self.stop)
            except Exception:
                self.stop()

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        state_name = result.value_nick if hasattr(result, "value_nick") else str(result)
        print(
            f"CAMERA_V11_DS_YOLO_CAM01_START camera={self.camera_id} state={state_name}",
            flush=True,
        )
        self.GLib.timeout_add_seconds(self.stats_interval, self._print_stats)
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(self.Gst.State.NULL)
            try:
                self.mux.release_request_pad(self.mux_sink)
            except Exception:
                pass
            self.wall.close()
        return 0 if self.errors == 0 else 1


def main() -> int:
    return V11DeepStreamYoloCam01V1().run()


if __name__ == "__main__":
    raise SystemExit(main())
