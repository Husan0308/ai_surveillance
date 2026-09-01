from __future__ import annotations

import os
import signal
import threading
import time
from collections import deque

import numpy as np

from services.camera_v11.step1_independent_egl_v4 import X11Wall
from services.camera_v11.step2_trt86 import CONTENT_H, INPUT_W, Step2TRT86Client
from services.camera_v2.native_bridge import NativeMetaBridge
from services.ml_service.app.config import CameraConfig, load_settings


class V11DeepStreamTRT86Cam01V2:
    """CAM-01 only: one DeepStream RTSP/decode pipeline + isolated TRT8.6 YOLO.

    The camera is opened exactly once. A tee after decode creates two bounded branches:
    display (nvstreammux -> metadata injection -> nvdsosd -> EGL) and detector
    (latest-only/gated nvvideoconvert -> appsink -> existing TRT8.6 worker).

    This deliberately avoids gst-nvinfer/TensorRT 10.x because Pascal-class GPUs are
    outside current TensorRT 10.x hardware support. Detection is still fed from the
    same DeepStream-decoded camera frames; there is no second RTSP session.
    """

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
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
        self.latency_ms = max(20, int(os.environ.get("V11_RTSP_LATENCY_MS", "100")))
        self.extra_surfaces = max(1, min(12, int(os.environ.get("V11_EXTRA_SURFACES", "4"))))
        self.udp_buffer_size = max(
            1_048_576,
            int(os.environ.get("V11_UDP_BUFFER_SIZE", str(max(settings.deepstream.udp_buffer_size, 8 * 1024 * 1024)))),
        )
        self.reconnect_sec = max(2, int(os.environ.get("V11_RECONNECT_SEC", "5")))
        self.width = max(320, int(os.environ.get("V11_DS_YOLO_DISPLAY_WIDTH", "640")))
        self.height = max(180, int(os.environ.get("V11_DS_YOLO_DISPLAY_HEIGHT", "360")))
        self.target_hz = max(0.25, min(4.0, float(os.environ.get("V11_DS_YOLO_HZ", "2.0"))))
        self.period_sec = 1.0 / self.target_hz
        self.conf = min(1.0, max(0.01, float(os.environ.get("V11_DS_YOLO_CONF", "0.18"))))
        self.max_det = max(1, min(100, int(os.environ.get("V11_DS_YOLO_MAX_DET", "20"))))
        self.box_stale_sec = max(
            self.period_sec * 1.25,
            min(2.5, float(os.environ.get("V11_DS_YOLO_BOX_STALE_SEC", "1.10"))),
        )
        self.stats_interval = max(2, int(os.environ.get("V11_STATS_INTERVAL_SEC", "5")))

        self.lock = threading.RLock()
        self._stopping = False
        self.infer_busy = False
        self.next_infer_mono = 0.0
        self.latest_boxes: list[list[float]] = []
        self.latest_result_mono = 0.0

        self.decoded = 0
        self.rendered = 0
        self.decoded_last = 0
        self.rendered_last = 0
        self.infer_completed = 0
        self.infer_completed_last = 0
        self.infer_admitted = 0
        self.infer_gate_drops = 0
        self.positive_buffers = 0
        self.detections_total = 0
        self.max_objects = 0
        self.metadata_added = 0
        self.copy_errors = 0
        self.infer_errors = 0
        self.meta_errors = 0
        self.errors = 0
        self.warnings = 0
        self.last_stat = time.monotonic()
        self.last_render: float | None = None
        self.render_gap_ms: deque[float] = deque(maxlen=4096)
        self.infer_roundtrip_ms: deque[float] = deque(maxlen=1024)

        self._preflight()
        self.detector = Step2TRT86Client()
        self.meta_bridge = NativeMetaBridge()
        self.wall = X11Wall(self.width, self.height, self.width, self.height, 1)
        self.loop = GLib.MainLoop()
        self.pipeline = self._build_pipeline()

        print(
            "CAMERA_V11_DS_YOLO_CAM01_ARCH "
            f"camera={self.camera_id} rtsp_sessions=1 decode=deepstream-nvurisrcbin tee=1 "
            "display=nvstreammux+nvdsosd detector=isolated-trt86-shm-b1 "
            "gst_nvinfer=0 second_rtsp=0 tracker=0 reid=0 face=0 ui=0",
            flush=True,
        )
        print(
            "CAMERA_V11_DS_YOLO_CAM01_POLICY "
            f"transport=tcp latency_ms={self.latency_ms} display={self.width}x{self.height} "
            f"detector={INPUT_W}x384 content_h={CONTENT_H} target_hz={self.target_hz:.2f} "
            f"conf={self.conf:.2f} stale_sec={self.box_stale_sec:.2f} queues=latest1",
            flush=True,
        )

    def _preflight(self) -> None:
        required = (
            "nvurisrcbin",
            "queue",
            "tee",
            "nvstreammux",
            "nvvideoconvert",
            "capsfilter",
            "appsink",
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

    def _latest_queue(self, queue) -> None:
        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera: CameraConfig) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)
        self._set_if(element, "protocols", 4)
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "do-rtsp-keep-alive", True)

    def _build_pipeline(self):
        safe = self.camera_id.lower().replace("-", "_")
        pipeline = self.Gst.Pipeline.new(f"v11_ds_trt86_{safe}")
        if pipeline is None:
            raise RuntimeError("could not create CAM-01 DeepStream/TRT8.6 pipeline")

        source = self._make("nvurisrcbin", f"source_{safe}")
        source_q = self._make("queue", f"source_q_{safe}")
        tee = self._make("tee", f"tee_{safe}")

        display_q = self._make("queue", f"display_q_{safe}")
        mux = self._make("nvstreammux", f"mux_{safe}")
        display_convert = self._make("nvvideoconvert", f"display_rgba_{safe}")
        display_caps = self._make("capsfilter", f"display_caps_{safe}")
        osd = self._make("nvdsosd", f"osd_{safe}")
        sink = self._make("nveglglessink", f"sink_{safe}")

        infer_q = self._make("queue", f"infer_q_{safe}")
        infer_convert = self._make("nvvideoconvert", f"infer_convert_{safe}")
        infer_caps = self._make("capsfilter", f"infer_caps_{safe}")
        appsink = self._make("appsink", f"infer_sink_{safe}")

        for queue in (source_q, display_q, infer_q):
            self._latest_queue(queue)

        source.connect("deep-element-added", self._configure_rtsp_child, self.camera)
        source.connect("pad-added", self._on_source_pad_added)
        source.set_property("uri", self.camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "gpu-id", self.gpu_id)
        self._set_if(source, "latency", self.latency_ms)
        self._set_if(source, "drop-on-latency", True)
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

        self._set_if(display_convert, "gpu-id", self.gpu_id)
        self._set_if(display_convert, "nvbuf-memory-type", 0)
        display_caps.set_property(
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

        self._set_if(infer_convert, "gpu-id", self.gpu_id)
        self._set_if(infer_convert, "compute-hw", 1)
        self._set_if(infer_convert, "interpolation-method", 2)
        infer_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INPUT_W},height={CONTENT_H},pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(appsink, "emit-signals", True)
        self._set_if(appsink, "sync", False)
        self._set_if(appsink, "async", False)
        self._set_if(appsink, "drop", True)
        self._set_if(appsink, "max-buffers", 1)
        self._set_if(appsink, "enable-last-sample", False)
        self._set_if(appsink, "wait-on-eos", False)
        appsink.connect("new-sample", self._on_infer_sample)

        elements = (
            source,
            source_q,
            tee,
            display_q,
            mux,
            display_convert,
            display_caps,
            osd,
            sink,
            infer_q,
            infer_convert,
            infer_caps,
            appsink,
        )
        for element in elements:
            pipeline.add(element)

        if not source_q.link(tee):
            raise RuntimeError("source_q->tee link failed")

        tee_display = tee.request_pad_simple("src_%u") if hasattr(tee, "request_pad_simple") else tee.get_request_pad("src_%u")
        tee_infer = tee.request_pad_simple("src_%u") if hasattr(tee, "request_pad_simple") else tee.get_request_pad("src_%u")
        display_sink_pad = display_q.get_static_pad("sink")
        infer_sink_pad = infer_q.get_static_pad("sink")
        if tee_display is None or tee_infer is None or display_sink_pad is None or infer_sink_pad is None:
            raise RuntimeError("tee request pad missing")
        if tee_display.link(display_sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("tee->display_q link failed")
        if tee_infer.link(infer_sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("tee->infer_q link failed")

        display_q_src = display_q.get_static_pad("src")
        mux_sink = mux.request_pad_simple("sink_0") if hasattr(mux, "request_pad_simple") else mux.get_request_pad("sink_0")
        if display_q_src is None or mux_sink is None:
            raise RuntimeError("display_q/mux pad missing")
        if display_q_src.link(mux_sink) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("display_q->mux link failed")

        for src, dst, label in (
            (mux, display_convert, "mux->display_convert"),
            (display_convert, display_caps, "display_convert->caps"),
            (display_caps, osd, "caps->osd"),
            (osd, sink, "osd->egl"),
            (infer_q, infer_convert, "infer_q->convert"),
            (infer_convert, infer_caps, "infer_convert->caps"),
            (infer_caps, appsink, "infer_caps->appsink"),
        ):
            if not src.link(dst):
                raise RuntimeError(f"link failed: {label}")

        infer_q_src = infer_q.get_static_pad("src")
        mux_src = mux.get_static_pad("src")
        sink_pad = sink.get_static_pad("sink")
        if infer_q_src is None or mux_src is None or sink_pad is None:
            raise RuntimeError("probe pad missing")
        infer_q_src.add_probe(self.Gst.PadProbeType.BUFFER, self._infer_gate_probe)
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._display_meta_probe)
        sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe)

        self.GstVideo.VideoOverlay.set_window_handle(sink, int(self.wall.children[0]))
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.source = source
        self.source_q = source_q
        self.tee = tee
        self.tee_display = tee_display
        self.tee_infer = tee_infer
        self.display_q = display_q
        self.infer_q = infer_q
        self.mux = mux
        self.mux_sink = mux_sink
        self.appsink = appsink
        self.sink = sink
        return pipeline

    def _on_source_pad_added(self, _source, pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            if not caps.get_structure(0).get_name().startswith("video/"):
                return
        sink_pad = self.source_q.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"source->source_q link failed: {result}")
        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe)
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

    def _infer_gate_probe(self, _pad, info):
        if info.get_buffer() is None:
            return self.Gst.PadProbeReturn.DROP
        now = time.monotonic()
        with self.lock:
            if self.infer_busy or now < self.next_infer_mono:
                self.infer_gate_drops += 1
                return self.Gst.PadProbeReturn.DROP
            self.infer_busy = True
            self.next_infer_mono = now + self.period_sec
            self.infer_admitted += 1
        return self.Gst.PadProbeReturn.OK

    def _on_infer_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            with self.lock:
                self.infer_busy = False
                self.infer_errors += 1
            return self.Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer is None:
            with self.lock:
                self.infer_busy = False
                self.infer_errors += 1
            return self.Gst.FlowReturn.OK

        mapped = False
        map_info = None
        try:
            ok, map_info = buffer.map(self.Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("appsink buffer map failed")
            mapped = True
            raw = np.frombuffer(map_info.data, dtype=np.uint8)
            expected = INPUT_W * CONTENT_H * 4
            if raw.size < expected:
                raise RuntimeError(f"appsink buffer too small {raw.size}<{expected}")
            frame = raw[:expected].reshape((CONTENT_H, INPUT_W, 4))
            self.detector.content[:, :, :] = frame[:, :, :3]
            result = self.detector.infer_preloaded(self.conf, self.max_det)
            now = time.monotonic()
            with self.lock:
                self.latest_boxes = [list(row) for row in result.boxes]
                self.latest_result_mono = now
                self.infer_completed += 1
                self.infer_roundtrip_ms.append(float(result.roundtrip_ms))
                count = len(result.boxes)
                self.detections_total += count
                self.max_objects = max(self.max_objects, count)
                if count > 0:
                    self.positive_buffers += 1
        except Exception as exc:
            with self.lock:
                self.infer_errors += 1
                errors = self.infer_errors
            if errors <= 5 or errors % 100 == 0:
                print(
                    f"CAMERA_V11_DS_YOLO_CAM01_INFER warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )
        finally:
            if mapped and map_info is not None:
                buffer.unmap(map_info)
            with self.lock:
                self.infer_busy = False
        return self.Gst.FlowReturn.OK

    def _display_meta_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.lock:
            age = now - self.latest_result_mono if self.latest_result_mono > 0 else 999.0
            boxes = [list(row) for row in self.latest_boxes] if age <= self.box_stale_sec else []
        if not boxes:
            return self.Gst.PadProbeReturn.OK

        sx = self.width / float(INPUT_W)
        sy = self.height / float(CONTENT_H)
        scaled = [
            [row[0] * sx, row[1] * sy, row[2] * sx, row[3] * sy, row[4]]
            for row in boxes
        ]
        try:
            added = self.meta_bridge.add_boxes(buffer, 0, scaled)
            if added < 0:
                raise RuntimeError(f"metadata add returned {added}")
            with self.lock:
                self.metadata_added += int(added)
        except Exception as exc:
            with self.lock:
                self.meta_errors += 1
                errors = self.meta_errors
            if errors <= 5 or errors % 100 == 0:
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
                    self.render_gap_ms.append(gap)
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
        src = message.src.get_name() if message.src is not None else "unknown"
        print(
            f"CAMERA_V11_DS_YOLO_CAM01_{kind} source={src} message={err} debug={debug}",
            flush=True,
        )

    @staticmethod
    def _pct(values, q: float) -> float:
        rows = sorted(values)
        if not rows:
            return 0.0
        idx = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * q))))
        return float(rows[idx])

    def _print_stats(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        with self.lock:
            elapsed = max(0.001, now - self.last_stat)
            source_fps = (self.decoded - self.decoded_last) / elapsed
            render_fps = (self.rendered - self.rendered_last) / elapsed
            infer_hz = (self.infer_completed - self.infer_completed_last) / elapsed
            self.decoded_last = self.decoded
            self.rendered_last = self.rendered
            self.infer_completed_last = self.infer_completed
            self.last_stat = now
            result_age_ms = (
                (now - self.latest_result_mono) * 1000.0 if self.latest_result_mono > 0 else -1.0
            )
            latest_count = len(self.latest_boxes) if result_age_ms >= 0 and result_age_ms <= self.box_stale_sec * 1000.0 else 0
            infer_buffers = self.infer_completed
            positive_buffers = self.positive_buffers
            detections_total = self.detections_total
            max_objects = self.max_objects
            meta_errors = self.meta_errors
            errors = self.errors
            infer_errors = self.infer_errors
            admitted = self.infer_admitted
            gate_drops = self.infer_gate_drops
            metadata_added = self.metadata_added
            roundtrip = list(self.infer_roundtrip_ms)
            render_gaps = list(self.render_gap_ms)
        queue_level = int(self.source_q.get_property("current-level-buffers"))
        print(
            "CAMERA_V11_DS_YOLO_CAM01 "
            f"source_fps={source_fps:.2f} render_fps={render_fps:.2f} infer_hz={infer_hz:.2f} "
            f"queue={queue_level} infer_buffers={infer_buffers} positive_buffers={positive_buffers} "
            f"detections_total={detections_total} max_objects={max_objects} latest_boxes={latest_count} "
            f"result_age_ms={result_age_ms:.0f} infer_p95_ms={self._pct(roundtrip, 0.95):.1f} "
            f"render_gap_p95_ms={self._pct(render_gaps, 0.95):.1f} admitted={admitted} gate_drops={gate_drops} "
            f"metadata_added={metadata_added} infer_errors={infer_errors} meta_errors={meta_errors} errors={errors}",
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
            try:
                self.pipeline.set_state(self.Gst.State.NULL)
            except Exception:
                pass
            for owner, pad in (
                (self.tee, self.tee_display),
                (self.tee, self.tee_infer),
                (self.mux, self.mux_sink),
            ):
                try:
                    owner.release_request_pad(pad)
                except Exception:
                    pass
            try:
                self.detector.close()
            except Exception:
                pass
            try:
                self.wall.close()
            except Exception:
                pass
        return 0


def main() -> int:
    return V11DeepStreamTRT86Cam01V2().run()


if __name__ == "__main__":
    raise SystemExit(main())
