from __future__ import annotations

import math
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from services.camera_v11.step1_independent_egl_v4 import X11Wall
from services.camera_v11.step2_trt86 import CONTENT_H, INPUT_H, INPUT_W, Step2TRT86Client
from services.camera_v2.native_bridge import NativeMetaBridge
from services.ml_service.app.config import CameraConfig, load_settings


MODEL_PAD_TOP = (INPUT_H - CONTENT_H) // 2


@dataclass(frozen=True)
class DetectorSnapshot:
    boxes: tuple[tuple[float, float, float, float, float], ...] = ()
    completed_mono: float = 0.0
    source_pts_ns: int = -1
    sequence: int = 0


@dataclass
class CameraRuntime:
    camera: CameraConfig
    index: int
    decoded: int = 0
    rendered: int = 0
    decoded_last: int = 0
    rendered_last: int = 0
    infer_completed: int = 0
    infer_completed_last: int = 0
    infer_admitted: int = 0
    infer_gate_drops: int = 0
    positive_buffers: int = 0
    detections_total: int = 0
    max_objects: int = 0
    result_clears: int = 0
    stale_expirations: int = 0
    expired_sequence: int = 0
    metadata_added: int = 0
    copy_errors: int = 0
    infer_errors: int = 0
    meta_errors: int = 0
    warnings: int = 0
    next_infer_mono: float = 0.0
    infer_pending: bool = False
    latest_snapshot: DetectorSnapshot = field(default_factory=DetectorSnapshot)
    render_gap_ms: deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    infer_roundtrip_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1024))
    last_render_mono: float | None = None
    pipeline: object | None = None
    source: object | None = None
    source_q: object | None = None
    tee: object | None = None
    display_q: object | None = None
    infer_q: object | None = None
    mux: object | None = None
    mux_sink: object | None = None
    appsink: object | None = None
    sink: object | None = None


def map_detector_boxes_to_display(
    boxes, display_width: int, display_height: int
) -> list[list[float]]:
    """Map padded 672x384 TRT coordinates back to the 672x378 video content."""
    sx = float(display_width) / float(INPUT_W)
    sy = float(display_height) / float(CONTENT_H)
    mapped: list[list[float]] = []
    for row in boxes:
        if len(row) != 5:
            continue
        x1, y1, x2, y2, confidence = (float(value) for value in row)
        left = min(float(display_width), max(0.0, x1 * sx))
        right = min(float(display_width), max(0.0, x2 * sx))
        top = min(float(display_height), max(0.0, (y1 - MODEL_PAD_TOP) * sy))
        bottom = min(float(display_height), max(0.0, (y2 - MODEL_PAD_TOP) * sy))
        if right <= left or bottom <= top:
            continue
        mapped.append([left, top, right, bottom, min(1.0, max(0.0, confidence))])
    return mapped


def _percentile(values: deque[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


class V11DeepStreamTRT86MultiCameraV1:
    """Incremental multi-camera DeepStream display with one shared TRT8.6 detector.

    Each configured camera owns exactly one nvurisrcbin/RTSP session. A tee after
    decode keeps display independent from a gated latest-only detector branch.
    All cameras share one Step2TRT86Client and therefore one TensorRT engine/context.
    Inference is serialized by a dedicated round-robin detector thread.
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
        requested = [
            value.strip()
            for value in os.environ.get("V11_DS_YOLO_CAMERAS", "CAM-01,CAM-02").split(",")
            if value.strip()
        ]
        if not requested:
            raise RuntimeError("V11_DS_YOLO_CAMERAS resolved to zero cameras")
        if len(set(requested)) != len(requested):
            raise RuntimeError(f"duplicate camera IDs requested: {requested}")
        by_id = {camera.camera_id: camera for camera in settings.cameras}
        missing = [cid for cid in requested if cid not in by_id]
        if missing:
            raise RuntimeError(f"configured cameras missing: {missing}")
        self.cameras = [by_id[cid] for cid in requested]
        self.camera_ids = tuple(requested)

        self.gpu_id = int(os.environ.get("V11_GPU_ID", settings.deepstream.gpu_id))
        self.latency_ms = max(20, int(os.environ.get("V11_RTSP_LATENCY_MS", "100")))
        self.extra_surfaces = max(1, min(12, int(os.environ.get("V11_EXTRA_SURFACES", "4"))))
        self.udp_buffer_size = max(
            1_048_576,
            int(
                os.environ.get(
                    "V11_UDP_BUFFER_SIZE",
                    str(max(settings.deepstream.udp_buffer_size, 8 * 1024 * 1024)),
                )
            ),
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
            min(2.5, float(os.environ.get("V11_DS_YOLO_BOX_STALE_SEC", "0.80"))),
        )
        self.stats_interval = max(2, int(os.environ.get("V11_STATS_INTERVAL_SEC", "5")))
        self.detector_enabled = os.environ.get("V11_DS_YOLO_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        self.lock = threading.RLock()
        self._stopping = False
        self.detector_stop = threading.Event()
        self.detector_thread: threading.Thread | None = None
        self.scheduler_index = 0
        self.errors = 0
        self.last_stat = time.monotonic()
        self.states = {
            camera.camera_id: CameraRuntime(camera=camera, index=index)
            for index, camera in enumerate(self.cameras)
        }

        self._preflight()
        self.detector = Step2TRT86Client() if self.detector_enabled else None
        self.meta_bridge = NativeMetaBridge()

        cols = min(3, len(self.cameras))
        rows = int(math.ceil(len(self.cameras) / 3.0))
        self.wall = X11Wall(
            self.width * cols,
            self.height * rows,
            self.width,
            self.height,
            len(self.cameras),
        )
        self.loop = GLib.MainLoop()
        for state in self.states.values():
            self._build_camera(state)

        print(
            "CAMERA_V11_DS_YOLO_MULTI_ARCH "
            f"cameras={len(self.cameras)} camera_ids={','.join(self.camera_ids)} "
            f"rtsp_sources={len(self.cameras)} rtsp_sessions={len(self.cameras)} "
            "rtsp_per_camera=1 decode=deepstream-nvdec source=nvurisrcbin tee_per_camera=1 "
            "display=independent-nvstreammux+nvdsosd detector=shared-trt86-sidecar "
            "detector_workers=1 detector_rtsp=0 detector_queue=latest1-per-camera "
            "detector_thread=dedicated scheduler=round-robin gst_nvinfer=0 second_rtsp=0 "
            "opencv=0 ffmpeg=0 tracker=0 reid=0 face=0 ui=0",
            flush=True,
        )
        print(
            "CAMERA_V11_DS_YOLO_MULTI_POLICY "
            f"transport=tcp latency_ms={self.latency_ms} display={self.width}x{self.height} "
            f"detector={INPUT_W}x{INPUT_H} content_h={CONTENT_H} target_hz={self.target_hz:.2f}/camera "
            f"conf={self.conf:.2f} stale_sec={self.box_stale_sec:.2f} queues=latest1 "
            f"enabled={int(self.detector_enabled)} model_pad_top={MODEL_PAD_TOP}",
            flush=True,
        )

    def _preflight(self) -> None:
        required = (
            "nvurisrcbin",
            "nvv4l2decoder",
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

    def _build_camera(self, state: CameraRuntime) -> None:
        cid = state.camera.camera_id
        safe = cid.lower().replace("-", "_")
        pipeline = self.Gst.Pipeline.new(f"v11_ds_trt86_{safe}")
        if pipeline is None:
            raise RuntimeError(f"could not create {cid} DeepStream/TRT8.6 pipeline")

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

        source.connect("deep-element-added", self._configure_rtsp_child, state.camera)
        source.connect("pad-added", self._on_source_pad_added, cid)
        source.set_property("uri", state.camera.uri)
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
        self._set_if(appsink, "emit-signals", False)
        self._set_if(appsink, "sync", False)
        self._set_if(appsink, "async", False)
        self._set_if(appsink, "drop", True)
        self._set_if(appsink, "max-buffers", 1)
        self._set_if(appsink, "enable-last-sample", False)
        self._set_if(appsink, "wait-on-eos", False)

        for element in (
            source, source_q, tee, display_q, mux, display_convert, display_caps,
            osd, sink, infer_q, infer_convert, infer_caps, appsink,
        ):
            pipeline.add(element)

        if not source_q.link(tee):
            raise RuntimeError(f"{cid}: source_q->tee link failed")
        tee_display = tee.request_pad_simple("src_%u") if hasattr(tee, "request_pad_simple") else tee.get_request_pad("src_%u")
        tee_infer = tee.request_pad_simple("src_%u") if hasattr(tee, "request_pad_simple") else tee.get_request_pad("src_%u")
        display_sink_pad = display_q.get_static_pad("sink")
        infer_sink_pad = infer_q.get_static_pad("sink")
        if tee_display is None or tee_infer is None or display_sink_pad is None or infer_sink_pad is None:
            raise RuntimeError(f"{cid}: tee request pad missing")
        if tee_display.link(display_sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee->display_q link failed")
        if tee_infer.link(infer_sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee->infer_q link failed")

        display_q_src = display_q.get_static_pad("src")
        mux_sink = mux.request_pad_simple("sink_0") if hasattr(mux, "request_pad_simple") else mux.get_request_pad("sink_0")
        if display_q_src is None or mux_sink is None:
            raise RuntimeError(f"{cid}: display_q/mux pad missing")
        if display_q_src.link(mux_sink) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display_q->mux link failed")

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
                raise RuntimeError(f"{cid}: link failed: {label}")

        infer_q_src = infer_q.get_static_pad("src")
        mux_src = mux.get_static_pad("src")
        sink_pad = sink.get_static_pad("sink")
        if infer_q_src is None or mux_src is None or sink_pad is None:
            raise RuntimeError(f"{cid}: probe pad missing")
        infer_q_src.add_probe(self.Gst.PadProbeType.BUFFER, self._infer_gate_probe, cid)
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._display_meta_probe, cid)
        sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe, cid)

        self.GstVideo.VideoOverlay.set_window_handle(sink, int(self.wall.children[state.index]))
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, cid)

        state.pipeline = pipeline
        state.source = source
        state.source_q = source_q
        state.tee = tee
        state.display_q = display_q
        state.infer_q = infer_q
        state.mux = mux
        state.mux_sink = mux_sink
        state.appsink = appsink
        state.sink = sink

    def _on_source_pad_added(self, _source, pad, cid: str) -> None:
        state = self.states[cid]
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            if not caps.get_structure(0).get_name().startswith("video/"):
                return
        sink_pad = state.source_q.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: source->source_q link failed: {result}")
        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        print(
            "CAMERA_V11_DS_YOLO_MULTI_LINK "
            f"camera={cid} status=OK caps={caps.to_string() if caps else 'pending'}",
            flush=True,
        )

    def _source_probe(self, _pad, info, cid: str):
        if info.get_buffer() is not None:
            with self.lock:
                self.states[cid].decoded += 1
        return self.Gst.PadProbeReturn.OK

    def _infer_gate_probe(self, _pad, info, cid: str):
        if info.get_buffer() is None or not self.detector_enabled:
            return self.Gst.PadProbeReturn.DROP
        now = time.monotonic()
        with self.lock:
            state = self.states[cid]
            if state.infer_pending or now < state.next_infer_mono:
                state.infer_gate_drops += 1
                return self.Gst.PadProbeReturn.DROP
            state.infer_pending = True
            state.next_infer_mono = now + self.period_sec
            state.infer_admitted += 1
        return self.Gst.PadProbeReturn.OK

    def _pick_pending_state(self) -> CameraRuntime | None:
        ids = self.camera_ids
        with self.lock:
            for offset in range(len(ids)):
                index = (self.scheduler_index + offset) % len(ids)
                state = self.states[ids[index]]
                if state.infer_pending:
                    self.scheduler_index = (index + 1) % len(ids)
                    return state
        return None

    def _detector_loop(self) -> None:
        print(
            "CAMERA_V11_DS_YOLO_MULTI_DETECTOR_THREAD "
            f"state=START workers=1 cameras={len(self.cameras)} scheduler=round-robin",
            flush=True,
        )
        expected = INPUT_W * CONTENT_H * 4
        try:
            while not self.detector_stop.is_set():
                state = self._pick_pending_state()
                if state is None:
                    self.detector_stop.wait(0.005)
                    continue
                sample = state.appsink.emit("try-pull-sample", 20_000_000)
                if sample is None:
                    self.detector_stop.wait(0.002)
                    continue
                buffer = sample.get_buffer()
                if buffer is None:
                    with self.lock:
                        state.infer_pending = False
                        state.infer_errors += 1
                    continue

                mapped = False
                map_info = None
                try:
                    ok, map_info = buffer.map(self.Gst.MapFlags.READ)
                    if not ok:
                        raise RuntimeError("appsink buffer map failed")
                    mapped = True
                    raw = np.frombuffer(map_info.data, dtype=np.uint8)
                    if raw.size < expected:
                        raise RuntimeError(f"appsink buffer too small {raw.size}<{expected}")
                    frame = raw[:expected].reshape((CONTENT_H, INPUT_W, 4))
                    detector = self.detector
                    if detector is None or detector.content is None:
                        raise RuntimeError("TRT86 detector is not ready")
                    detector.content[:, :, :] = frame[:, :, :3]
                    result = detector.infer_preloaded(self.conf, self.max_det)
                    now = time.monotonic()
                    raw_pts = int(buffer.pts)
                    pts_ns = -1 if raw_pts == int(self.Gst.CLOCK_TIME_NONE) else raw_pts
                    immutable_boxes = tuple(tuple(float(value) for value in row) for row in result.boxes)
                    with self.lock:
                        sequence = state.infer_completed + 1
                        if state.latest_snapshot.boxes and not immutable_boxes:
                            state.result_clears += 1
                        state.latest_snapshot = DetectorSnapshot(
                            boxes=immutable_boxes,
                            completed_mono=now,
                            source_pts_ns=pts_ns,
                            sequence=sequence,
                        )
                        state.infer_completed = sequence
                        state.infer_roundtrip_ms.append(float(result.roundtrip_ms))
                        count = len(immutable_boxes)
                        state.detections_total += count
                        state.max_objects = max(state.max_objects, count)
                        if count > 0:
                            state.positive_buffers += 1
                            positive_index = state.positive_buffers
                        else:
                            positive_index = 0
                    if positive_index and (positive_index <= 3 or positive_index % 20 == 0):
                        display_box = map_detector_boxes_to_display(immutable_boxes[:1], self.width, self.height)
                        first = ",".join(f"{value:.1f}" for value in display_box[0]) if display_box else "none"
                        print(
                            "CAMERA_V11_DS_YOLO_MULTI_DETECTION "
                            f"camera={state.camera.camera_id} sequence={sequence} count={count} first_display_box={first}",
                            flush=True,
                        )
                except Exception as exc:
                    with self.lock:
                        state.infer_errors += 1
                        errors = state.infer_errors
                    if errors <= 5 or errors % 100 == 0:
                        print(
                            "CAMERA_V11_DS_YOLO_MULTI_INFER "
                            f"camera={state.camera.camera_id} warning={type(exc).__name__}:{exc} errors={errors}",
                            flush=True,
                        )
                finally:
                    if mapped and map_info is not None:
                        buffer.unmap(map_info)
                    with self.lock:
                        state.infer_pending = False
        finally:
            with self.lock:
                for state in self.states.values():
                    state.infer_pending = False
            print("CAMERA_V11_DS_YOLO_MULTI_DETECTOR_THREAD state=STOP", flush=True)

    def _display_meta_probe(self, _pad, info, cid: str):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.lock:
            state = self.states[cid]
            snapshot = state.latest_snapshot
            age = now - snapshot.completed_mono if snapshot.completed_mono > 0 else 999.0
            if snapshot.boxes and age > self.box_stale_sec and snapshot.sequence != state.expired_sequence:
                state.expired_sequence = snapshot.sequence
                state.stale_expirations += 1
            boxes = snapshot.boxes if age <= self.box_stale_sec else ()
        if not boxes:
            return self.Gst.PadProbeReturn.OK
        scaled = map_detector_boxes_to_display(boxes, self.width, self.height)
        if not scaled:
            return self.Gst.PadProbeReturn.OK
        try:
            added = self.meta_bridge.add_boxes(buffer, 0, scaled)
            if added < 0:
                raise RuntimeError(f"metadata add returned {added}")
            with self.lock:
                state.metadata_added += int(added)
        except Exception as exc:
            with self.lock:
                state.meta_errors += 1
                errors = state.meta_errors
            if errors <= 5 or errors % 100 == 0:
                print(
                    "CAMERA_V11_DS_YOLO_MULTI_META "
                    f"camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )
        return self.Gst.PadProbeReturn.OK

    def _render_probe(self, _pad, info, cid: str):
        if info.get_buffer() is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.lock:
            state = self.states[cid]
            state.rendered += 1
            if state.last_render_mono is not None:
                gap = (now - state.last_render_mono) * 1000.0
                if 0.0 <= gap <= 5000.0:
                    state.render_gap_ms.append(gap)
            state.last_render_mono = now
        return self.Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus, message, cid: str) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            with self.lock:
                self.errors += 1
            print(
                "CAMERA_V11_DS_YOLO_MULTI_ERROR "
                f"camera={cid} error={error} debug={debug}",
                flush=True,
            )
            self.stop()
        elif message.type == self.Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            with self.lock:
                self.states[cid].warnings += 1
            print(
                "CAMERA_V11_DS_YOLO_MULTI_WARNING "
                f"camera={cid} warning={warning} debug={debug}",
                flush=True,
            )

    def _stats(self) -> bool:
        now = time.monotonic()
        dt = max(1e-6, now - self.last_stat)
        with self.lock:
            rows = []
            for cid in self.camera_ids:
                state = self.states[cid]
                source_fps = (state.decoded - state.decoded_last) / dt
                render_fps = (state.rendered - state.rendered_last) / dt
                infer_hz = (state.infer_completed - state.infer_completed_last) / dt
                snapshot = state.latest_snapshot
                age_ms = (now - snapshot.completed_mono) * 1000.0 if snapshot.completed_mono > 0 else -1.0
                latest_boxes = len(snapshot.boxes) if snapshot.completed_mono > 0 and (now - snapshot.completed_mono) <= self.box_stale_sec else 0
                queue = 1 if state.infer_pending else 0
                rows.append((
                    cid, source_fps, render_fps, infer_hz, queue,
                    state.infer_completed, state.infer_admitted, state.infer_gate_drops,
                    state.positive_buffers, state.detections_total, state.max_objects,
                    latest_boxes, state.result_clears, state.stale_expirations,
                    state.metadata_added, age_ms, _percentile(state.infer_roundtrip_ms, 0.95),
                    _percentile(state.render_gap_ms, 0.95), state.copy_errors,
                    state.infer_errors, state.meta_errors, state.warnings,
                ))
                state.decoded_last = state.decoded
                state.rendered_last = state.rendered
                state.infer_completed_last = state.infer_completed
            pipeline_errors = self.errors
            worker_alive = int(self.detector is not None and self.detector.process is not None and self.detector.process.poll() is None)
            detector_thread_alive = int(self.detector_thread is not None and self.detector_thread.is_alive())
            self.last_stat = now

        for row in rows:
            (
                cid, source_fps, render_fps, infer_hz, queue, infer_completed,
                infer_admitted, infer_gate_drops, positive_buffers, detections_total,
                max_objects, latest_boxes, result_clears, stale_expirations,
                metadata_added, age_ms, infer_p95_ms, render_gap_p95_ms,
                copy_errors, infer_errors, meta_errors, warnings,
            ) = row
            print(
                "CAMERA_V11_DS_YOLO_MULTI "
                f"camera={cid} source_fps={source_fps:.2f} render_fps={render_fps:.2f} "
                f"infer_hz={infer_hz:.2f} queue={queue} infer_count={infer_completed} "
                f"infer_admitted={infer_admitted} detector_drops={infer_gate_drops} "
                f"positive_inferences={positive_buffers} detections_total={detections_total} "
                f"max_objects={max_objects} latest_boxes={latest_boxes} "
                f"result_clears={result_clears} stale_expirations={stale_expirations} "
                f"metadata_added={metadata_added} result_age_ms={age_ms:.1f} "
                f"infer_p95_ms={infer_p95_ms:.1f} render_gap_p95_ms={render_gap_p95_ms:.1f} "
                f"detector_thread_alive={detector_thread_alive} worker_alive={worker_alive} "
                f"copy_errors={copy_errors} infer_errors={infer_errors} meta_errors={meta_errors} "
                f"warnings={warnings} pipeline_errors={pipeline_errors}",
                flush=True,
            )
        return not self._stopping

    def run(self) -> int:
        print(
            "CAMERA_V11_DS_YOLO_MULTI_START "
            f"cameras={','.join(self.camera_ids)} state=async",
            flush=True,
        )
        self.last_stat = time.monotonic()
        self.GLib.timeout_add_seconds(self.stats_interval, self._stats)
        if self.detector_enabled:
            self.detector_thread = threading.Thread(
                target=self._detector_loop,
                name="v11-shared-trt86-detector",
                daemon=False,
            )
            self.detector_thread.start()

        for cid in self.camera_ids:
            state = self.states[cid]
            result = state.pipeline.set_state(self.Gst.State.PLAYING)
            if result == self.Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(f"{cid}: failed to enter PLAYING")
            print(f"CAMERA_V11_DS_YOLO_MULTI_CAMERA camera={cid} state=PLAYING", flush=True)
            time.sleep(0.15)

        def handle_signal(_signum, _frame) -> None:
            self.GLib.idle_add(self.stop)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        try:
            self.loop.run()
            return 0 if self.errors == 0 else 1
        finally:
            self.close()

    def stop(self) -> bool:
        if self._stopping:
            return False
        self._stopping = True
        self.detector_stop.set()
        try:
            self.loop.quit()
        except Exception:
            pass
        return False

    def close(self) -> None:
        self.detector_stop.set()
        if self.detector_thread is not None and self.detector_thread.is_alive():
            self.detector_thread.join(timeout=3.0)
        for state in self.states.values():
            if state.pipeline is not None:
                try:
                    state.pipeline.set_state(self.Gst.State.NULL)
                except Exception:
                    pass
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception as exc:
                print(
                    "CAMERA_V11_DS_YOLO_MULTI_SHUTDOWN "
                    f"warning={type(exc).__name__}:{exc}",
                    flush=True,
                )
        try:
            self.wall.close()
        except Exception:
            pass
        print(
            "CAMERA_V11_DS_YOLO_MULTI_STOP "
            f"cameras={','.join(self.camera_ids)} errors={self.errors}",
            flush=True,
        )


def main() -> int:
    app = V11DeepStreamTRT86MultiCameraV1()
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
