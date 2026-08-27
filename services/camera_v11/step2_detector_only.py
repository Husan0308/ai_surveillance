from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import numpy as np

from services.ml_service.app.config import CameraConfig
from .batch6_trt86 import Batch6TRT86Client
from .step1_cam02_lowlat_v7 import V11Step1Cam02LowLatV7


DETECT_W = 672
DETECT_H = 384
DETECT_CONTENT_H = 378


class FreshBatchMailbox:
    """One latest detector frame per camera; versions prevent backlog reuse."""

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.rows: dict[str, tuple[int, float, np.ndarray]] = {}
        self.versions: dict[str, int] = {}
        self.closed = False

    def put(self, cid: str, captured_at: float, frame: np.ndarray) -> int:
        with self.cv:
            version = self.versions.get(cid, 0) + 1
            self.versions[cid] = version
            self.rows[cid] = (version, captured_at, frame)
            self.cv.notify_all()
            return version

    def wait_new(self, cid: str, old_version: int, timeout: float):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not self.closed:
                row = self.rows.get(cid)
                if row is not None and row[0] > old_version:
                    return row
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.cv.wait(remaining)
        return None

    def version(self, cid: str) -> int:
        with self.cv:
            return int(self.versions.get(cid, 0))

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()


class V11Step2DetectorOnly(V11Step1Cam02LowLatV7):
    """Frozen V11 Step1 V7 display plus one isolated detector branch per camera.

    Per camera:
        nvurisrcbin/NVDEC -> tee
          -> latest1 display queue -> frozen V7 GPU display -> independent EGL sink
          -> latest1 detector queue -> demand gate -> GPU resize 672x378 -> BGRx appsink

    Six demanded fresh frames are letterboxed to 672x384 and sent as one true batch-6
    TensorRT 8.6 enqueue in an isolated subprocess. There is no tracker, OSD, ReID,
    face model, nvstreammux, or nvinfer in Step2. Detector prefetch depth is exactly
    one future batch, so conversion can overlap current TRT without historical frames.
    """

    def __init__(self) -> None:
        self.det_conf = min(1.0, max(0.01, float(os.environ.get("V11_DETECT_CONF", "0.18"))))
        self.det_max_det = max(1, min(100, int(os.environ.get("V11_DETECT_MAX_DET", "20"))))
        self.det_capture_timeout_ms = max(
            80.0, min(500.0, float(os.environ.get("V11_DETECT_CAPTURE_TIMEOUT_MS", "180")))
        )
        self.det_prefetch_batches = 1
        self.det_lock = threading.RLock()
        self.det_stop = threading.Event()
        self.det_thread: threading.Thread | None = None
        self.det_ready = False
        self.det_error = ""
        self.det_batches = 0
        self.det_timeouts = 0
        self.det_total_boxes = 0
        self.det_last_boxes: dict[str, int] = {}
        self.det_result_counts: dict[str, int] = {}
        self.det_capture_counts: dict[str, int] = {}
        self.detector_queues: dict[str, object] = {}
        self.detector_converts: dict[str, object] = {}
        self.detector_sinks: dict[str, object] = {}
        self.tees: dict[str, object] = {}
        self.tee_request_pads: list[tuple[object, object]] = []
        self.capture_requested: dict[str, bool] = {}
        self.capture_origin: dict[str, float] = {}
        self.mailbox = FreshBatchMailbox()
        self.capture_wait_ms: deque[float] = deque(maxlen=2048)
        self.input_skew_ms: deque[float] = deque(maxlen=2048)
        self.shm_copy_ms: deque[float] = deque(maxlen=2048)
        self.prep_ms: deque[float] = deque(maxlen=2048)
        self.trt_ms: deque[float] = deque(maxlen=2048)
        self.roundtrip_ms: deque[float] = deque(maxlen=2048)
        self.result_age_ms: deque[float] = deque(maxlen=2048)
        self.batch_interval_ms: deque[float] = deque(maxlen=2048)
        self.det_conversion_age_ms: dict[str, deque[float]] = {}
        self.det_result_age_ms: dict[str, deque[float]] = {}
        self.det_queue_qmax: dict[str, int] = {}
        self.last_batch_completed = 0.0
        self._detector_started = False
        super().__init__()

        for camera in self.cameras:
            cid = camera.camera_id
            self.det_last_boxes.setdefault(cid, 0)
            self.det_result_counts.setdefault(cid, 0)
            self.det_capture_counts.setdefault(cid, 0)
            self.capture_requested.setdefault(cid, False)
            self.det_conversion_age_ms.setdefault(cid, deque(maxlen=2048))
            self.det_result_age_ms.setdefault(cid, deque(maxlen=2048))
            self.det_queue_qmax.setdefault(cid, 0)

        print(
            "CAMERA_V11_STEP2_ARCH "
            "base=step1-v7-frozen cameras=6 decode_once=1 tee=1 display_independent=1 "
            "detector=trt86-batch6 tracker=0 osd=0 reid=0 face=0 nvinfer=0 "
            "detector_queue=latest1/leaky-downstream detector_prefetch_batches=1",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP2_POLICY "
            f"input={DETECT_W}x{DETECT_CONTENT_H}+3px/3px-pad114 batch=6 "
            f"conf={self.det_conf:.2f} max_det={self.det_max_det} "
            f"capture_timeout_ms={self.det_capture_timeout_ms:.0f} "
            "scheduler=coalesced-fresh-six overlap=one-future-batch",
            flush=True,
        )

    def _preflight(self) -> None:
        super()._preflight()
        required = ("tee", "appsink")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("V11 Step2 missing GStreamer plugins: " + ", ".join(missing))

    def _request_tee_pad(self, tee):
        request_simple = getattr(tee, "request_pad_simple", None)
        pad = request_simple("src_%u") if request_simple else None
        if pad is None:
            pad = tee.get_request_pad("src_%u")
        if pad is None:
            raise RuntimeError(f"{tee.get_name()}: could not allocate src_%u")
        self.tee_request_pads.append((tee, pad))
        return pad

    def _link_tee_to_queue(self, tee, queue, label: str) -> None:
        src = self._request_tee_pad(tee)
        sink = queue.get_static_pad("sink")
        if sink is None or src.link(sink) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"V11 Step2 tee link failed: {label}")

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        safe = cid.lower().replace("-", "_")
        requested_lowlat = int(cid in self.lowlat_cameras)

        pipeline = self.Gst.Pipeline.new(f"v11_step2_{safe}")
        if pipeline is None:
            raise RuntimeError(f"{cid}: could not create Step2 pipeline")

        source = self._make("nvurisrcbin", f"source_{safe}")
        tee = self._make("tee", f"tee_{safe}")
        display_q = self._make("queue", f"latest_{safe}")
        display_convert = self._make("nvvideoconvert", f"scale_{safe}")
        display_caps = self._make("capsfilter", f"caps_{safe}")
        display_sink = self._make("nveglglessink", f"sink_{safe}")
        detector_q = self._make("queue", f"detect_latest_{safe}")
        detector_convert = self._make("nvvideoconvert", f"detect_scale_{safe}")
        detector_caps = self._make("capsfilter", f"detect_caps_{safe}")
        detector_sink = self._make("appsink", f"detect_sink_{safe}")

        self._configure_latest_queue(display_q)
        self._configure_latest_queue(detector_q)

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.connect("pad-added", self._on_source_pad_added_to_tee, tee, cid)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "gpu-id", self.gpu_id)
        self._set_if(source, "latency", self.latency_ms)
        self._set_if(source, "drop-on-latency", self.drop_on_latency)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "select-rtp-protocol", 4)  # Frozen V7 TCP baseline.
        self._set_if(source, "rtsp-reconnect-interval", self.reconnect_sec)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)

        self._set_if(display_convert, "gpu-id", self.gpu_id)
        self._set_if(display_convert, "nvbuf-memory-type", 0)
        self._set_if(display_convert, "interpolation-method", self.interpolation)
        display_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.tile_width},height={self.tile_height}"
            ),
        )
        self._set_if(display_sink, "sync", False)
        self._set_if(display_sink, "qos", False)
        self._set_if(display_sink, "async", False)
        self._set_if(display_sink, "enable-last-sample", False)
        self._set_if(display_sink, "force-aspect-ratio", False)

        self._set_if(detector_convert, "gpu-id", self.gpu_id)
        self._set_if(detector_convert, "nvbuf-memory-type", 0)
        self._set_if(detector_convert, "compute-hw", 1)
        self._set_if(detector_convert, "interpolation-method", 2)
        detector_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={DETECT_W},height={DETECT_CONTENT_H},"
                "pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(detector_sink, "emit-signals", True)
        self._set_if(detector_sink, "sync", False)
        self._set_if(detector_sink, "async", False)
        self._set_if(detector_sink, "drop", True)
        self._set_if(detector_sink, "max-buffers", 1)
        self._set_if(detector_sink, "enable-last-sample", False)
        self._set_if(detector_sink, "wait-on-eos", False)

        for element in (
            source,
            tee,
            display_q,
            display_convert,
            display_caps,
            display_sink,
            detector_q,
            detector_convert,
            detector_caps,
            detector_sink,
        ):
            pipeline.add(element)

        self._link_tee_to_queue(tee, display_q, f"{cid}:tee->display")
        self._link_tee_to_queue(tee, detector_q, f"{cid}:tee->detector")
        self._require_link(display_q, display_convert, f"{cid}:display_q->convert")
        self._require_link(display_convert, display_caps, f"{cid}:display_convert->caps")
        self._require_link(display_caps, display_sink, f"{cid}:display_caps->sink")
        self._require_link(detector_q, detector_convert, f"{cid}:detector_q->convert")
        self._require_link(detector_convert, detector_caps, f"{cid}:detector_convert->caps")
        self._require_link(detector_caps, detector_sink, f"{cid}:detector_caps->appsink")

        display_sink_pad = display_sink.get_static_pad("sink")
        if display_sink_pad is None:
            raise RuntimeError(f"{cid}: display sink pad missing")
        display_sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe, cid)
        detector_q_src = detector_q.get_static_pad("src")
        if detector_q_src is None:
            raise RuntimeError(f"{cid}: detector queue src pad missing")
        detector_q_src.add_probe(self.Gst.PadProbeType.BUFFER, self._detector_gate_probe, cid)
        detector_sink.connect("new-sample", self._on_detector_sample, cid)

        try:
            self.GstVideo.VideoOverlay.set_window_handle(
                display_sink, int(self.wall.children[index])
            )
            overlay_ok = 1
        except Exception as exc:
            raise RuntimeError(f"{cid}: GstVideoOverlay window binding failed: {exc}") from exc

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, cid)

        self.pipelines[cid] = pipeline
        self.sources[cid] = source
        self.queues[cid] = display_q
        self.converters[cid] = display_convert
        self.capsfilters[cid] = display_caps
        self.sinks[cid] = display_sink
        self.tees[cid] = tee
        self.detector_queues[cid] = detector_q
        self.detector_converts[cid] = detector_convert
        self.detector_sinks[cid] = detector_sink
        self.capture_requested[cid] = False
        self.det_capture_counts[cid] = 0
        self.det_result_counts[cid] = 0
        self.det_last_boxes[cid] = 0
        self.det_conversion_age_ms[cid] = deque(maxlen=2048)
        self.det_result_age_ms[cid] = deque(maxlen=2048)
        self.det_queue_qmax[cid] = 0

        # Preserve frozen Step1 checker compatibility while adding Step2 metadata.
        print(
            "CAMERA_V11_STEP1V7_WINDOW "
            f"camera={cid} transport=tcp low_latency={requested_lowlat} "
            f"xid={self.wall.children[index]} overlay={overlay_ok} "
            f"tile={self.tile_width}x{self.tile_height}",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP2_WINDOW "
            f"camera={cid} display_queue=latest1 detector_queue=latest1 "
            f"detector_gate=demand detector_input={DETECT_W}x{DETECT_CONTENT_H}",
            flush=True,
        )

    def _on_source_pad_added_to_tee(self, _source, pad, tee, cid: str) -> None:
        caps = pad.get_current_caps()
        if caps is None or caps.get_size() == 0:
            try:
                caps = pad.query_caps(None)
            except Exception:
                caps = None
        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            try:
                media = str(caps.get_structure(0).get_name())
            except Exception:
                media = ""
            if media and not media.startswith("video/"):
                return

        sink = tee.get_static_pad("sink")
        if sink is None or sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            with self.lock:
                self.stats[cid].errors += 1
            name = result.value_nick if hasattr(result, "value_nick") else str(result)
            print(f"CAMERA_V11_STEP2_LINK camera={cid} status=error result={name}", flush=True)
            return
        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        with self.lock:
            self.stats[cid].source_linked = True
        print(
            f"CAMERA_V11_STEP2_LINK camera={cid} status=OK pad={pad.get_name()} "
            f"caps={caps.to_string() if caps else 'pending'}",
            flush=True,
        )

    def _detector_gate_probe(self, _pad, info, cid: str):
        with self.det_lock:
            if not self.capture_requested.get(cid, False):
                return self.Gst.PadProbeReturn.DROP
            self.capture_requested[cid] = False

        buffer = info.get_buffer()
        now = time.monotonic()
        origin = now
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts = int(buffer.pts)
            with self.lock:
                matched = self._match_pts_arrival(self.stats[cid], pts)
            if matched is not None:
                origin = matched
        with self.det_lock:
            self.capture_origin[cid] = origin
        return self.Gst.PadProbeReturn.OK

    def _on_detector_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK
        try:
            structure = sample.get_caps().get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            if width != DETECT_W or height != DETECT_CONTENT_H:
                raise RuntimeError(
                    f"{cid}: detector geometry={width}x{height}, "
                    f"expected={DETECT_W}x{DETECT_CONTENT_H}"
                )
            buffer = sample.get_buffer()
            ok, mapped = buffer.map(self.Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError(f"{cid}: detector BGRx map failed")
            try:
                tight_stride = width * 4
                mapped_size = int(getattr(mapped, "size", len(mapped.data)))
                if mapped_size < tight_stride * height:
                    raise RuntimeError(
                        f"{cid}: mapped detector bytes={mapped_size} < {tight_stride * height}"
                    )
                row_stride = mapped_size // height if mapped_size % height == 0 else tight_stride
                if row_stride < tight_stride:
                    raise RuntimeError(f"{cid}: detector stride={row_stride} < {tight_stride}")
                raw = np.frombuffer(mapped.data, dtype=np.uint8, count=row_stride * height)
                bgrx = raw.reshape((height, row_stride))[:, :tight_stride].reshape(
                    (height, width, 4)
                )
                frame = np.full((DETECT_H, DETECT_W, 3), 114, dtype=np.uint8)
                frame[3:381, :, :] = bgrx[..., :3]
            finally:
                buffer.unmap(mapped)

            now = time.monotonic()
            with self.det_lock:
                captured_at = self.capture_origin.pop(cid, now)
                self.det_capture_counts[cid] = self.det_capture_counts.get(cid, 0) + 1
                age_ms = max(0.0, (now - captured_at) * 1000.0)
                self.det_conversion_age_ms[cid].append(age_ms)
            self.mailbox.put(cid, captured_at, frame)
        except Exception as exc:
            with self.det_lock:
                self.det_error = f"capture:{cid}:{type(exc).__name__}:{exc}"
            print(
                f"CAMERA_V11_STEP2_CAPTURE_WARNING camera={cid} "
                f"error={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        return self.Gst.FlowReturn.OK

    def _request_all(self, ids: list[str]) -> float:
        with self.det_lock:
            for cid in ids:
                self.capture_requested[cid] = True
        return time.monotonic()

    @staticmethod
    def _safe_error(value: str) -> str:
        if not value:
            return "none"
        return value.replace(" ", "_").replace(";", ",")[:180]

    def _detector_scheduler(self) -> None:
        client: Batch6TRT86Client | None = None
        try:
            client = Batch6TRT86Client()
            warm = np.full((DETECT_H, DETECT_W, 3), 114, dtype=np.uint8)
            ids = [camera.camera_id for camera in self.cameras]
            for _ in range(3):
                client.infer(ids, [warm] * 6, conf=self.det_conf, max_det=self.det_max_det)
            with self.det_lock:
                self.det_ready = True
            print(
                "CAMERA_V11_STEP2_DETECT_READY "
                f"engine={client.engine} worker={client.worker.name} batch=6 "
                "tensorrt=8.6.1 warmup=3 prefetch=1",
                flush=True,
            )

            while not self.det_stop.is_set():
                with self.lock:
                    source_ready = all(self.stats[cid].decoded > 0 for cid in ids)
                if source_ready:
                    break
                self.det_stop.wait(0.02)
            if self.det_stop.is_set():
                return

            versions = {cid: self.mailbox.version(cid) for cid in ids}
            request_started = self._request_all(ids)

            while not self.det_stop.is_set():
                deadline = request_started + self.det_capture_timeout_ms / 1000.0
                frames: list[np.ndarray] = []
                captured: list[float] = []
                new_versions: dict[str, int] = {}
                complete = True
                for cid in ids:
                    remaining = max(0.0, deadline - time.monotonic())
                    row = self.mailbox.wait_new(cid, versions[cid], remaining)
                    if row is None:
                        complete = False
                        break
                    version, captured_at, frame = row
                    new_versions[cid] = int(version)
                    frames.append(frame)
                    captured.append(float(captured_at))

                if not complete or len(frames) != 6:
                    with self.det_lock:
                        self.det_timeouts += 1
                    for cid in ids:
                        versions[cid] = self.mailbox.version(cid)
                    request_started = self._request_all(ids)
                    continue

                versions.update(new_versions)
                full_at = time.monotonic()
                capture_wait = max(0.0, (full_at - request_started) * 1000.0)
                skew = max(0.0, (max(captured) - min(captured)) * 1000.0)

                # Bounded one-batch prefetch: request exactly the next future frame
                # from each camera before current TRT starts. No historical queue exists.
                next_request_started = self._request_all(ids)
                result = client.infer(
                    ids,
                    frames,
                    conf=self.det_conf,
                    max_det=self.det_max_det,
                )
                completed = time.monotonic()
                max_age = max((completed - ts) * 1000.0 for ts in captured)

                with self.det_lock:
                    if self.last_batch_completed > 0.0:
                        self.batch_interval_ms.append(
                            (completed - self.last_batch_completed) * 1000.0
                        )
                    self.last_batch_completed = completed
                    self.det_batches += 1
                    self.capture_wait_ms.append(capture_wait)
                    self.input_skew_ms.append(skew)
                    self.shm_copy_ms.append(result.shm_copy_ms)
                    self.prep_ms.append(result.prep_ms)
                    self.trt_ms.append(result.trt_ms)
                    self.roundtrip_ms.append(result.roundtrip_ms)
                    self.result_age_ms.append(max_age)
                    total_boxes = 0
                    for cid, ts in zip(ids, captured):
                        count = len(result.boxes.get(cid, []))
                        self.det_last_boxes[cid] = count
                        self.det_result_counts[cid] += 1
                        self.det_result_age_ms[cid].append(
                            max(0.0, (completed - ts) * 1000.0)
                        )
                        total_boxes += count
                    self.det_total_boxes += total_boxes
                    self.det_error = ""
                    batch_n = self.det_batches

                if batch_n <= 5 or batch_n % 20 == 0:
                    print(
                        "CAMERA_V11_STEP2_BATCH "
                        f"n={batch_n} capture_wait={capture_wait:.1f}ms skew={skew:.1f}ms "
                        f"shm={result.shm_copy_ms:.1f}ms prep={result.prep_ms:.1f}ms "
                        f"trt={result.trt_ms:.1f}ms roundtrip={result.roundtrip_ms:.1f}ms "
                        f"result_age={max_age:.1f}ms boxes={total_boxes}",
                        flush=True,
                    )
                request_started = next_request_started
        except BaseException as exc:
            with self.det_lock:
                self.det_error = f"{type(exc).__name__}:{exc}"
            print(
                f"CAMERA_V11_STEP2_DETECT_FATAL error={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            if client is not None:
                client.close()
            with self.det_lock:
                self.det_ready = False

    def _start_detector(self) -> None:
        if self._detector_started:
            return
        self._detector_started = True
        self.det_thread = threading.Thread(
            target=self._detector_scheduler,
            name="camera-v11-step2-batch6-detector",
            daemon=True,
        )
        self.det_thread.start()

    def _start_pipelines(self) -> None:
        super()._start_pipelines()
        self._start_detector()

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        if not keep:
            return False

        rows = []
        with self.det_lock:
            for camera in self.cameras:
                cid = camera.camera_id
                q = int(self.detector_queues[cid].get_property("current-level-buffers"))
                self.det_queue_qmax[cid] = max(self.det_queue_qmax.get(cid, 0), q)
                rows.append(
                    (
                        cid,
                        self.det_capture_counts.get(cid, 0),
                        self.det_result_counts.get(cid, 0),
                        list(self.det_conversion_age_ms[cid]),
                        list(self.det_result_age_ms[cid]),
                        q,
                        self.det_queue_qmax[cid],
                        self.det_last_boxes.get(cid, 0),
                    )
                )

            intervals = list(self.batch_interval_ms)
            actual_hz = 0.0
            if intervals:
                avg = sum(intervals) / len(intervals)
                actual_hz = 1000.0 / avg if avg > 0.0 else 0.0
            ready = int(self.det_ready)
            batches = self.det_batches
            timeouts = self.det_timeouts
            total_boxes = self.det_total_boxes
            error = self._safe_error(self.det_error)
            capture_wait = list(self.capture_wait_ms)
            skew = list(self.input_skew_ms)
            shm = list(self.shm_copy_ms)
            prep = list(self.prep_ms)
            trt = list(self.trt_ms)
            roundtrip = list(self.roundtrip_ms)
            result_age = list(self.result_age_ms)

        for cid, captures, results, conversion, ages, q, qmax, last_boxes in rows:
            print(
                "CAMERA_V11_STEP2_DETECT_CAMERA "
                f"camera={cid} hz={actual_hz:.2f} captures={captures} results={results} "
                f"conversion_p95={self._pct(conversion, 0.95):.1f}ms "
                f"result_age_p95={self._pct(ages, 0.95):.1f}ms "
                f"q={q} qmax={qmax} last_boxes={last_boxes}",
                flush=True,
            )

        print(
            "CAMERA_V11_STEP2_DETECT_STATS "
            f"ready={ready} batch_actual={actual_hz:.2f}Hz "
            f"capture_wait_p95={self._pct(capture_wait, 0.95):.1f}ms "
            f"input_skew_p95={self._pct(skew, 0.95):.1f}ms "
            f"shm_copy_p95={self._pct(shm, 0.95):.1f}ms "
            f"prep_p95={self._pct(prep, 0.95):.1f}ms "
            f"trt_p50={self._pct(trt, 0.50):.1f}ms trt_p95={self._pct(trt, 0.95):.1f}ms "
            f"roundtrip_p95={self._pct(roundtrip, 0.95):.1f}ms "
            f"result_age_p95={self._pct(result_age, 0.95):.1f}ms "
            f"batches={batches} timeouts={timeouts} boxes={total_boxes} error={error}",
            flush=True,
        )
        return True

    def stop(self) -> bool:
        self.det_stop.set()
        self.mailbox.close()
        return super().stop()

    def run(self) -> int:
        try:
            return super().run()
        finally:
            self.det_stop.set()
            self.mailbox.close()
            thread = self.det_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=12.0)
            for owner, pad in self.tee_request_pads:
                try:
                    owner.release_request_pad(pad)
                except Exception:
                    pass
            self.tee_request_pads.clear()


def main() -> int:
    return V11Step2DetectorOnly().run()


if __name__ == "__main__":
    raise SystemExit(main())
