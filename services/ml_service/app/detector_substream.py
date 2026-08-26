from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W, TRT86DetectorClient
from services.shared.camera_config import CameraConfig, load_settings


@dataclass
class CaptureSlot:
    seq: int = 0
    captured_ns: int = 0
    frame: np.ndarray | None = None


@dataclass
class SourceStat:
    frames: int = 0
    last_frames: int = 0
    last_at: float = field(default_factory=time.monotonic)
    intervals_ms: deque[float] = field(default_factory=lambda: deque(maxlen=240))
    last_pts_ns: int | None = None
    caps: str = "pending"


def _percentile(values: deque[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    index = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * q))))
    return float(rows[index])


def substream_uri(main_uri: str) -> str:
    parts = urlsplit(main_uri)
    path = parts.path
    marker = "/Streaming/Channels/"
    if marker not in path:
        marker = "/Streaming/channels/"
    if marker not in path:
        raise ValueError(f"unsupported Hikvision RTSP path: {path}")
    prefix, channel = path.rsplit("/", 1)
    if len(channel) < 2 or channel[-2:] != "01":
        raise ValueError(f"expected Hikvision main-stream channel ending in 01, got {channel}")
    return urlunsplit((parts.scheme, parts.netloc, f"{prefix}/{channel[:-2]}02", parts.query, parts.fragment))


class DetectorSubstreamService:
    """Detector-only ML process consuming Hikvision substreams directly.

    Camera Service owns only the high-quality main streams. This process owns the
    low-resolution substreams, performs one JIT resize/color conversion only when
    TensorRT is ready, then runs one isolated TRT8.6 inference. No camera-service
    SHM, tracker, API, UI, ReID or identity code is on this path.
    """

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self.Gst = Gst
        settings = load_settings()
        self.cameras = list(settings.cameras)
        self.gpu_id = int(os.environ.get("ML_SUBSTREAM_GPU_ID", str(settings.gpu_id)))
        self.rtsp_transport = os.environ.get("ML_SUBSTREAM_RTSP_TRANSPORT", "tcp").strip().lower()
        self.rtsp_latency_ms = max(40, int(os.environ.get("ML_SUBSTREAM_RTSP_LATENCY_MS", "80")))
        self.extra_surfaces = max(2, min(12, int(os.environ.get("ML_SUBSTREAM_EXTRA_SURFACES", "4"))))
        self.target_hz = max(0.1, min(10.0, float(os.environ.get("ML_DETECTOR_TARGET_HZ", "2.0"))))
        self.target_period = 1.0 / self.target_hz
        self.conf = min(1.0, max(0.01, float(os.environ.get("ML_DETECTOR_CONF", "0.18"))))
        self.max_det = max(1, min(100, int(os.environ.get("ML_DETECTOR_MAX_DET", "20"))))
        self.capture_timeout_ms = max(100.0, float(os.environ.get("ML_SUBSTREAM_CAPTURE_TIMEOUT_MS", "300")))
        self.startup_stagger_sec = max(0.05, float(os.environ.get("ML_SUBSTREAM_STARTUP_STAGGER_SEC", "0.35")))

        required = ["nvurisrcbin", "queue", "nvvideoconvert", "capsfilter", "appsink"]
        missing = [name for name in required if Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("ML substream runtime missing plugins: " + ", ".join(missing))

        self.pipeline = Gst.Pipeline.new("ml-detector-substreams")
        if self.pipeline is None:
            raise RuntimeError("could not create ML substream pipeline")

        self.sources: dict[str, object] = {}
        self.input_queues: dict[str, object] = {}
        self.capture_requested = {camera.camera_id: False for camera in self.cameras}
        self.capture_slots = {camera.camera_id: CaptureSlot() for camera in self.cameras}
        self.capture_lock = threading.Lock()
        self.capture_condition = threading.Condition(self.capture_lock)
        self.source_stats = {camera.camera_id: SourceStat() for camera in self.cameras}
        self.processed = {camera.camera_id: 0 for camera in self.cameras}
        self.processed_last = {camera.camera_id: 0 for camera in self.cameras}
        self.capture_timeouts = {camera.camera_id: 0 for camera in self.cameras}
        self.box_counts = {camera.camera_id: 0 for camera in self.cameras}
        self.next_due = {camera.camera_id: 0.0 for camera in self.cameras}
        self.capture_wait_ms: deque[float] = deque(maxlen=240)
        self.input_age_ms: deque[float] = deque(maxlen=240)
        self.infer_ms: deque[float] = deque(maxlen=240)
        self.result_age_ms: deque[float] = deque(maxlen=240)
        self.stats_at = time.monotonic()
        self.stop_requested = False
        self.detector: TRT86DetectorClient | None = None

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        self.bus = self.pipeline.get_bus()
        print(
            "ML_SUBSTREAM_ARCH "
            f"cameras={len(self.cameras)} main_stream_dependency=0 camera_shm=0 "
            "source=Hikvision-substream nvdec=1 tracker=0 api=0 ui=0 "
            "policy=one-JIT-convert-then-one-TRT sequential_gpu_compute=1",
            flush=True,
        )

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

    @staticmethod
    def _link(src, dst, label: str) -> None:
        if not src.link(dst):
            raise RuntimeError(f"failed to link {label}")

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
        if self.rtsp_transport == "tcp":
            self._set_if(element, "protocols", 4)
            self._set_if(element, "tcp-timestamp", True)
        elif self.rtsp_transport == "udp":
            self._set_if(element, "protocols", 1)
        self._set_if(element, "latency", self.rtsp_latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "buffer-mode", 3)
        self._set_if(element, "do-rtsp-keep-alive", True)

    def _add_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        source = self._make("nvurisrcbin", f"ml_sub_source_{index}")
        input_q = self._make("queue", f"ml_sub_input_q_{index}")
        convert = self._make("nvvideoconvert", f"ml_sub_convert_{index}")
        caps = self._make("capsfilter", f"ml_sub_caps_{index}")
        output_q = self._make("queue", f"ml_sub_output_q_{index}")
        sink = self._make("appsink", f"ml_sub_sink_{index}")

        self._latest_queue(input_q)
        self._latest_queue(output_q)
        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "compute-hw", 1)
        self._set_if(convert, "interpolation-method", 2)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INPUT_W},height={CONTENT_H},pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(sink, "emit-signals", True)
        self._set_if(sink, "sync", False)
        self._set_if(sink, "max-buffers", 1)
        self._set_if(sink, "drop", True)
        self._set_if(sink, "wait-on-eos", False)
        self._set_if(sink, "enable-last-sample", False)
        sink.connect("new-sample", self._on_sample, cid)

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", substream_uri(camera.uri))
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 4 if self.rtsp_transport == "tcp" else 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", 3)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)
        self._set_if(source, "gpu-id", self.gpu_id)

        for element in (source, input_q, convert, caps, output_q, sink):
            self.pipeline.add(element)
        self._link(input_q, convert, f"{cid}:input_q->convert")
        self._link(convert, caps, f"{cid}:convert->caps")
        self._link(caps, output_q, f"{cid}:caps->output_q")
        self._link(output_q, sink, f"{cid}:output_q->appsink")

        input_q.get_static_pad("sink").add_probe(
            self.Gst.PadProbeType.BUFFER, self._source_probe, cid
        )
        input_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._capture_gate_probe, cid
        )
        source.connect("pad-added", self._source_pad_added, input_q, cid)
        self.sources[cid] = source
        self.input_queues[cid] = input_q

    def _source_pad_added(self, _source, pad, queue, cid: str) -> None:
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
        sink = queue.get_static_pad("sink")
        if sink is None or sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"ML_SUBSTREAM_ERROR {cid} source-link result={result}", file=sys.stderr, flush=True)

    def _source_probe(self, pad, info, cid: str):
        stat = self.source_stats[cid]
        stat.frames += 1
        if stat.caps == "pending":
            caps = pad.get_current_caps()
            if caps is not None:
                stat.caps = caps.to_string()
                print(f"ML_SUBSTREAM_SOURCE {cid} negotiated={stat.caps}", flush=True)
        buffer = info.get_buffer()
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts = int(buffer.pts)
            prev = stat.last_pts_ns
            stat.last_pts_ns = pts
            if prev is not None and pts > prev:
                delta = (pts - prev) / 1_000_000.0
                if 0.0 < delta < 1000.0:
                    stat.intervals_ms.append(delta)
        return self.Gst.PadProbeReturn.OK

    def _capture_gate_probe(self, _pad, _info, cid: str):
        with self.capture_lock:
            if not self.capture_requested[cid]:
                return self.Gst.PadProbeReturn.DROP
            self.capture_requested[cid] = False
        return self.Gst.PadProbeReturn.OK

    def _on_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK
        try:
            structure = sample.get_caps().get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            if width != INPUT_W or height != CONTENT_H:
                raise RuntimeError(f"{cid}: capture={width}x{height}, expected={INPUT_W}x{CONTENT_H}")
            buffer = sample.get_buffer()
            ok, mapped = buffer.map(self.Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError(f"{cid}: BGRx map failed")
            try:
                tight = width * 4
                size = int(getattr(mapped, "size", len(mapped.data)))
                row_stride = size // height if size % height == 0 else tight
                if row_stride < tight or size < row_stride * height:
                    raise RuntimeError(f"{cid}: invalid BGRx stride={row_stride} size={size}")
                raw = np.frombuffer(mapped.data, dtype=np.uint8, count=row_stride * height)
                rows = raw.reshape((height, row_stride))
                bgrx = rows[:, :tight].reshape((height, width, 4))
                frame = np.ascontiguousarray(bgrx[..., :3])
            finally:
                buffer.unmap(mapped)
            captured_ns = time.monotonic_ns()
            with self.capture_condition:
                slot = self.capture_slots[cid]
                slot.seq += 1
                slot.captured_ns = captured_ns
                slot.frame = frame
                self.capture_condition.notify_all()
        except Exception as exc:
            print(f"ML_SUBSTREAM_ERROR {cid} sample={type(exc).__name__}:{exc}", file=sys.stderr, flush=True)
        return self.Gst.FlowReturn.OK

    def _capture_fresh(self, cid: str):
        with self.capture_condition:
            baseline = self.capture_slots[cid].seq
            self.capture_requested[cid] = True
            started = time.monotonic()
            deadline = started + self.capture_timeout_ms / 1000.0
            while not self.stop_requested:
                slot = self.capture_slots[cid]
                if slot.seq > baseline and slot.frame is not None:
                    waited = (time.monotonic() - started) * 1000.0
                    return slot.seq, slot.captured_ns, slot.frame.copy(), waited
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.capture_requested[cid] = False
                    self.capture_timeouts[cid] += 1
                    return baseline, 0, None, (time.monotonic() - started) * 1000.0
                self.capture_condition.wait(timeout=min(remaining, 0.02))
        return baseline, 0, None, 0.0

    def _start_sources(self) -> None:
        for source in self.sources.values():
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)
        self.pipeline.set_state(self.Gst.State.PLAYING)
        for camera in self.cameras:
            source = self.sources[camera.camera_id]
            source.set_locked_state(False)
            source.sync_state_with_parent()
            time.sleep(self.startup_stagger_sec)

    def _poll_bus(self) -> None:
        while True:
            message = self.bus.pop_filtered(
                self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS
            )
            if message is None:
                return
            if message.type == self.Gst.MessageType.WARNING:
                err, debug = message.parse_warning()
                print(f"ML_SUBSTREAM_WARNING {err} debug={debug or '-'}", file=sys.stderr, flush=True)
            elif message.type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"ML_SUBSTREAM_ERROR {err} debug={debug or '-'}", file=sys.stderr, flush=True)
            elif message.type == self.Gst.MessageType.EOS:
                print("ML_SUBSTREAM_WARNING eos=1", file=sys.stderr, flush=True)

    def _print_stats(self) -> None:
        now = time.monotonic()
        elapsed = max(0.001, now - self.stats_at)
        source_parts = []
        detect_parts = []
        for camera in self.cameras:
            cid = camera.camera_id
            stat = self.source_stats[cid]
            fps = (stat.frames - stat.last_frames) / max(0.001, now - stat.last_at)
            stat.last_frames = stat.frames
            stat.last_at = now
            p50 = _percentile(stat.intervals_ms, 0.50)
            p95 = _percentile(stat.intervals_ms, 0.95)
            cadence = "?" if not stat.intervals_ms else f"{p50:.0f}/{p95:.0f}ms"
            source_parts.append(f"{cid}:{fps:.1f}fps pts={cadence}")
            total = self.processed[cid]
            previous = self.processed_last[cid]
            detect_parts.append(f"{cid}:{(total - previous) / elapsed:.2f}Hz")
            self.processed_last[cid] = total
        self.stats_at = now
        print("ML_SUBSTREAM_SOURCE_STATS " + " | ".join(source_parts), flush=True)
        print(
            "ML_DETECTOR_STATS "
            f"actual=[{' '.join(detect_parts)}] "
            f"capture_wait_p95={_percentile(self.capture_wait_ms, 0.95):.1f}ms "
            f"infer_avg={sum(self.infer_ms) / len(self.infer_ms) if self.infer_ms else 0.0:.1f}ms "
            f"infer_p95={_percentile(self.infer_ms, 0.95):.1f}ms "
            f"input_age_p95={_percentile(self.input_age_ms, 0.95):.1f}ms "
            f"result_age_p95={_percentile(self.result_age_ms, 0.95):.1f}ms "
            f"capture_timeouts={sum(self.capture_timeouts.values())} boxes={sum(self.box_counts.values())}",
            flush=True,
        )

    def run(self) -> int:
        print(
            "ML_DETECTOR_PROFILE "
            f"source=Hikvision-substream-direct cameras={len(self.cameras)} "
            f"capture={INPUT_W}x{CONTENT_H} target={self.target_hz:.2f}Hz/cam "
            f"rtsp={self.rtsp_latency_ms}ms extra_surfaces={self.extra_surfaces} "
            f"conf={self.conf:.2f} max_det={self.max_det}",
            flush=True,
        )
        print(
            "ML_DETECTOR_BOUNDARY camera_service=independent main_stream=0 camera_shm=0 "
            "tracker=0 api=0 ui=0 substream_nvdec=1 JIT_convert=1",
            flush=True,
        )
        self._start_sources()
        self.detector = TRT86DetectorClient()

        while not self.stop_requested:
            self._poll_bus()
            did_work = False
            for camera in self.cameras:
                cid = camera.camera_id
                if self.stop_requested:
                    break
                now = time.monotonic()
                if now < self.next_due[cid]:
                    continue
                did_work = True
                request_started = now
                seq, captured_ns, frame, capture_wait = self._capture_fresh(cid)
                self.capture_wait_ms.append(capture_wait)
                self.next_due[cid] = request_started + self.target_period
                if frame is None:
                    continue

                input_age = max(0.0, (time.monotonic_ns() - captured_ns) / 1_000_000.0)
                self.input_age_ms.append(input_age)
                result = self.detector.infer(frame, self.conf, self.max_det)
                result_age = max(0.0, (time.monotonic_ns() - captured_ns) / 1_000_000.0)
                self.processed[cid] += 1
                self.box_counts[cid] += len(result.boxes)
                self.infer_ms.append(result.roundtrip_ms)
                self.result_age_ms.append(result_age)
                n = sum(self.processed.values())
                if n <= 3 or n % 20 == 0:
                    best = max((row[4] for row in result.boxes), default=0.0)
                    print(
                        "ML_DETECTOR_TRT "
                        f"n={n} camera={cid} frame_seq={seq} capture_wait={capture_wait:.1f}ms "
                        f"input_age={input_age:.1f}ms roundtrip={result.roundtrip_ms:.1f}ms "
                        f"prep={result.prep_ms:.1f}ms trt={result.trt_ms:.1f}ms "
                        f"result_age={result_age:.1f}ms boxes={len(result.boxes)} best={best:.3f}",
                        flush=True,
                    )
                if time.monotonic() - self.stats_at >= 5.0:
                    self._print_stats()

            if not did_work:
                if time.monotonic() - self.stats_at >= 5.0:
                    self._print_stats()
                time.sleep(0.002)
        return 0

    def close(self) -> None:
        self.stop_requested = True
        with self.capture_condition:
            self.capture_condition.notify_all()
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        self.pipeline.set_state(self.Gst.State.NULL)


def main() -> int:
    service = DetectorSubstreamService()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
