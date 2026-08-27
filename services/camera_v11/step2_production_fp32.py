from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from services.shared.camera_config import CameraConfig, load_settings
from .step2_trt86 import CONTENT_H, INPUT_W, Step2TRT86Client


@dataclass
class CameraStat:
    rtsp_frames: int = 0
    decoded_frames: int = 0
    accepted: int = 0
    processed: int = 0
    gate_drops: int = 0
    timeouts: int = 0
    stale_drops: int = 0
    queue_max: int = 0
    rtsp_last: int = 0
    decoded_last: int = 0
    processed_last: int = 0
    last_report: float = field(default_factory=time.monotonic)
    last_pts_ns: int | None = None
    pts_delta_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1024))


def _pct(values: deque[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[index])


def _jitter_p95(values: deque[float]) -> float:
    if not values:
        return 0.0
    median = _pct(values, 0.50)
    deviations = deque((abs(value - median) for value in values), maxlen=len(values))
    return _pct(deviations, 0.95)


def _substream_uri(main_uri: str) -> str:
    parts = urlsplit(main_uri)
    path = parts.path
    marker = "/Streaming/Channels/"
    if marker not in path:
        marker = "/Streaming/channels/"
    if marker not in path:
        raise ValueError(f"unsupported detector substream path: {path}")
    prefix, channel = path.rsplit("/", 1)
    if len(channel) < 2 or channel[-2:] != "01":
        raise ValueError(f"expected main stream channel ending in 01, got {channel}")
    return urlunsplit(
        (parts.scheme, parts.netloc, f"{prefix}/{channel[:-2]}02", parts.query, parts.fragment)
    )


class V11Step2ProductionFP32:
    """Display-independent latest-only detector ingest.

    Frozen Step1 is deliberately not imported or subclassed. This process opens
    camera substreams, admits one requested decoded frame through a pre-convert gate,
    pulls the sample on the scheduler thread, and immediately consumes or drops it.
    There is no Python frame queue and no GStreamer queue deeper than one buffer.
    """

    MODES = {"extraction", "preprocessing", "synthetic-trt", "full"}

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self.Gst = Gst
        settings = load_settings()
        self.cameras = list(settings.cameras)
        if len(self.cameras) != 6:
            raise RuntimeError(f"Step2 requires exactly six cameras, got {len(self.cameras)}")

        self.mode = os.environ.get("V11_STEP2_MODE", "full").strip().lower()
        if self.mode not in self.MODES:
            raise RuntimeError(f"V11_STEP2_MODE must be one of {sorted(self.MODES)}, got {self.mode}")
        self.target_hz = max(0.1, min(4.0, float(os.environ.get("V11_STEP2_HZ", "2.0"))))
        self.global_period = 1.0 / (self.target_hz * len(self.cameras))
        self.conf = min(1.0, max(0.01, float(os.environ.get("V11_STEP2_CONF", "0.18"))))
        self.max_det = max(1, min(100, int(os.environ.get("V11_STEP2_MAX_DET", "20"))))
        self.capture_timeout_ms = max(
            50.0, min(500.0, float(os.environ.get("V11_STEP2_CAPTURE_TIMEOUT_MS", "180")))
        )
        self.max_input_age_ms = max(
            50.0, min(500.0, float(os.environ.get("V11_STEP2_MAX_INPUT_AGE_MS", "220")))
        )
        self.rtsp_latency_ms = max(40, int(os.environ.get("V11_STEP2_RTSP_LATENCY_MS", "80")))
        self.extra_surfaces = max(2, min(8, int(os.environ.get("V11_STEP2_EXTRA_SURFACES", "3"))))
        self.startup_stagger = max(
            0.05, min(1.0, float(os.environ.get("V11_STEP2_STARTUP_STAGGER_SEC", "0.25")))
        )
        self.gpu_id = int(os.environ.get("V11_STEP2_GPU_ID", str(settings.gpu_id)))

        self.lock = threading.Lock()
        self.stop_requested = False
        self.requested = {camera.camera_id: False for camera in self.cameras}
        self.accepted_seq = {camera.camera_id: 0 for camera in self.cameras}
        self.accepted_ns = {camera.camera_id: 0 for camera in self.cameras}
        self.accepted_pts_ns = {camera.camera_id: -1 for camera in self.cameras}
        self.stats = {camera.camera_id: CameraStat() for camera in self.cameras}
        self.decoder_probe_ids: set[int] = set()
        self.sources: dict[str, object] = {}
        self.input_queues: dict[str, object] = {}
        self.output_queues: dict[str, object] = {}
        self.sinks: dict[str, object] = {}
        self.detector: Step2TRT86Client | None = None
        self.preprocess_buffer = np.empty((1, 3, 384, INPUT_W), dtype=np.float32)

        self.stage_values = {
            name: deque(maxlen=2048)
            for name in (
                "schedule_wait",
                "capture_wait",
                "nvmm_resize_bgrx",
                "map_copy",
                "preprocess",
                "h2d",
                "inference",
                "d2h",
                "postprocess",
                "cuda_sync_wait",
                "ipc_roundtrip",
                "cpu_ipc_block",
                "result_age",
            )
        }
        self.report_at = time.monotonic()

        if self.mode != "synthetic-trt":
            required = ("nvurisrcbin", "nvv4l2decoder", "queue", "nvvideoconvert", "capsfilter", "appsink")
            missing = [name for name in required if Gst.ElementFactory.find(name) is None]
            if missing:
                raise RuntimeError("Step2 detector ingest missing plugins: " + ", ".join(missing))
            self.pipeline = Gst.Pipeline.new("camera_v11_step2_detector_ingest")
            if self.pipeline is None:
                raise RuntimeError("could not create Step2 detector ingest pipeline")
            for index, camera in enumerate(self.cameras):
                self._build_camera(index, camera)
            self.bus = self.pipeline.get_bus()
        else:
            self.pipeline = None
            self.bus = None

        print(
            "CAMERA_V11_STEP2_PRODUCTION_ARCH "
            f"mode={self.mode} display_process=independent display_topology_changed=0 "
            "ingest=separate-substream-process queue_max=1 app_pending=0 batch=1 "
            "scheduler=round-robin-latest tracker=0 reid=0 face=0 osd=0 jpeg=0 base64=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP2_PRODUCTION_POLICY "
            f"precision=fp32 conf={self.conf:.2f} target={self.target_hz:.2f}Hz/camera "
            f"capture_timeout={self.capture_timeout_ms:.0f}ms max_age={self.max_input_age_ms:.0f}ms "
            "busy_policy=drop-old/no-debt stream=nonblocking-low-priority",
            flush=True,
        )

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"could not create {factory}:{name}")
        return element

    @staticmethod
    def _set_if(element, name: str, value) -> bool:
        if element.find_property(name) is None:
            return False
        element.set_property(name, value)
        return True

    @staticmethod
    def _link(source, destination, label: str) -> None:
        if not source.link(destination):
            raise RuntimeError(f"failed to link {label}")

    def _latest_queue(self, queue) -> None:
        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

    def _configure_deep_element(self, _bin, _sub_bin, element, camera: CameraConfig) -> None:
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name == "rtspsrc":
            if camera.username:
                self._set_if(element, "user-id", camera.username)
                self._set_if(element, "user-pw", camera.password)
            self._set_if(element, "protocols", 4)
            self._set_if(element, "tcp-timestamp", False)
            self._set_if(element, "latency", self.rtsp_latency_ms)
            self._set_if(element, "drop-on-latency", True)
            self._set_if(element, "buffer-mode", 3)
            return
        if factory_name != "nvv4l2decoder":
            return
        identity = id(element)
        if identity in self.decoder_probe_ids:
            return
        self.decoder_probe_ids.add(identity)
        sink_pad = element.get_static_pad("sink")
        src_pad = element.get_static_pad("src")
        if sink_pad is not None:
            sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._rtsp_frame_probe, camera.camera_id)
        if src_pad is not None:
            src_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._decoded_probe, camera.camera_id)

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        source = self._make("nvurisrcbin", f"step2_source_{index}")
        input_q = self._make("queue", f"step2_input_q_{index}")
        convert = self._make("nvvideoconvert", f"step2_convert_{index}")
        caps = self._make("capsfilter", f"step2_caps_{index}")
        output_q = self._make("queue", f"step2_output_q_{index}")
        sink = self._make("appsink", f"step2_sink_{index}")
        self._latest_queue(input_q)
        self._latest_queue(output_q)

        source.connect("deep-element-added", self._configure_deep_element, camera)
        source.connect("pad-added", self._source_pad_added, input_q, cid)
        source.set_property("uri", _substream_uri(camera.uri))
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "gpu-id", self.gpu_id)
        self._set_if(source, "select-rtp-protocol", 4)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)

        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "compute-hw", 1)
        self._set_if(convert, "interpolation-method", 2)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INPUT_W},height={CONTENT_H},pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(sink, "emit-signals", False)
        self._set_if(sink, "sync", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "drop", True)
        self._set_if(sink, "max-buffers", 1)
        self._set_if(sink, "enable-last-sample", False)
        self._set_if(sink, "wait-on-eos", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "processing-deadline", 0)

        for element in (source, input_q, convert, caps, output_q, sink):
            self.pipeline.add(element)
        self._link(input_q, convert, f"{cid}:input->convert")
        self._link(convert, caps, f"{cid}:convert->caps")
        self._link(caps, output_q, f"{cid}:caps->output")
        self._link(output_q, sink, f"{cid}:output->appsink")

        input_sink = input_q.get_static_pad("sink")
        input_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._capture_gate_probe, cid)
        self.sources[cid] = source
        self.input_queues[cid] = input_q
        self.output_queues[cid] = output_q
        self.sinks[cid] = sink

    def _source_pad_added(self, _source, pad, queue, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            if not caps.get_structure(0).get_name().startswith("video/"):
                return
        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"CAMERA_V11_STEP2_ERROR camera={cid} source_link={result}", file=sys.stderr, flush=True)

    def _rtsp_frame_probe(self, _pad, _info, cid: str):
        with self.lock:
            self.stats[cid].rtsp_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _decoded_probe(self, _pad, info, cid: str):
        buffer = info.get_buffer()
        with self.lock:
            stat = self.stats[cid]
            stat.decoded_frames += 1
            if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
                pts = int(buffer.pts)
                if stat.last_pts_ns is not None and pts > stat.last_pts_ns:
                    delta = (pts - stat.last_pts_ns) / 1_000_000.0
                    if 0.0 < delta < 2000.0:
                        stat.pts_delta_ms.append(delta)
                stat.last_pts_ns = pts
        return self.Gst.PadProbeReturn.OK

    def _capture_gate_probe(self, _pad, info, cid: str):
        with self.lock:
            stat = self.stats[cid]
            if not self.requested[cid]:
                stat.gate_drops += 1
                return self.Gst.PadProbeReturn.DROP
            self.requested[cid] = False
            stat.accepted += 1
            self.accepted_seq[cid] += 1
            self.accepted_ns[cid] = time.monotonic_ns()
            buffer = info.get_buffer()
            self.accepted_pts_ns[cid] = (
                int(buffer.pts)
                if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE
                else -1
            )
        return self.Gst.PadProbeReturn.OK

    def _drain_sink(self, cid: str) -> int:
        drained = 0
        sink = self.sinks[cid]
        while sink.emit("try-pull-sample", 0) is not None:
            drained += 1
        return drained

    def _capture_into_preallocated(self, cid: str):
        self._drain_sink(cid)
        with self.lock:
            baseline = self.accepted_seq[cid]
            self.requested[cid] = True
        started = time.perf_counter()
        sample = self.sinks[cid].emit(
            "try-pull-sample", int(self.capture_timeout_ms * 1_000_000.0)
        )
        capture_wait_ms = (time.perf_counter() - started) * 1000.0
        if sample is None:
            with self.lock:
                self.requested[cid] = False
                self.stats[cid].timeouts += 1
            return None, capture_wait_ms, 0.0, 0.0

        sample_ready_ns = time.monotonic_ns()
        with self.lock:
            sequence = self.accepted_seq[cid]
            accepted_ns = self.accepted_ns[cid]
        if sequence <= baseline or accepted_ns <= 0:
            return None, capture_wait_ms, 0.0, 0.0
        conversion_ms = max(0.0, (sample_ready_ns - accepted_ns) / 1_000_000.0)

        copy_started = time.perf_counter()
        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        if (width, height) != (INPUT_W, CONTENT_H):
            raise RuntimeError(f"{cid}: detector sample is {width}x{height}")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError(f"{cid}: detector sample map failed")
        try:
            tight = width * 4
            size = int(getattr(mapped, "size", len(mapped.data)))
            stride = size // height if size % height == 0 else tight
            if stride < tight or size < stride * height:
                raise RuntimeError(f"{cid}: invalid BGRx stride={stride} size={size}")
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=stride * height)
            bgrx = raw.reshape(height, stride)[:, :tight].reshape(height, width, 4)
            target = self.detector.content if self.detector is not None else self._local_content
            np.copyto(target, bgrx[:, :, :3], casting="no")
        finally:
            buffer.unmap(mapped)
        copy_ms = (time.perf_counter() - copy_started) * 1000.0
        return accepted_ns, capture_wait_ms, conversion_ms, copy_ms

    @property
    def _local_content(self) -> np.ndarray:
        if not hasattr(self, "_local_frame"):
            self._local_frame = np.full((384, INPUT_W, 3), 114, dtype=np.uint8)
        return self._local_frame[3:381]

    def _local_preprocess(self) -> float:
        started = time.perf_counter()
        frame = self._local_frame
        scale = 1.0 / 255.0
        np.multiply(frame[:, :, 2], scale, out=self.preprocess_buffer[0, 0], casting="unsafe")
        np.multiply(frame[:, :, 1], scale, out=self.preprocess_buffer[0, 1], casting="unsafe")
        np.multiply(frame[:, :, 0], scale, out=self.preprocess_buffer[0, 2], casting="unsafe")
        return (time.perf_counter() - started) * 1000.0

    def _start_ingest(self) -> None:
        for source in self.sources.values():
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)
        self.pipeline.set_state(self.Gst.State.PLAYING)
        for camera in self.cameras:
            source = self.sources[camera.camera_id]
            source.set_locked_state(False)
            source.sync_state_with_parent()
            time.sleep(self.startup_stagger)

    def _poll_bus(self) -> None:
        if self.bus is None:
            return
        while True:
            message = self.bus.pop_filtered(
                self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS
            )
            if message is None:
                return
            if message.type == self.Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                print(f"CAMERA_V11_STEP2_ERROR error={error} debug={debug or '-'}", file=sys.stderr, flush=True)
            elif message.type == self.Gst.MessageType.WARNING:
                error, debug = message.parse_warning()
                print(f"CAMERA_V11_STEP2_WARNING warning={error} debug={debug or '-'}", file=sys.stderr, flush=True)

    def _append_worker_stages(self, stages: dict[str, float]) -> None:
        mapping = {
            "preprocess_ms": "preprocess",
            "h2d_ms": "h2d",
            "inference_ms": "inference",
            "d2h_ms": "d2h",
            "postprocess_ms": "postprocess",
            "sync_wait_ms": "cuda_sync_wait",
        }
        for source, destination in mapping.items():
            self.stage_values[destination].append(float(stages.get(source, 0.0)))

    def _print_stats(self) -> None:
        now = time.monotonic()
        camera_rows = []
        queue_rows = []
        with self.lock:
            for camera in self.cameras:
                cid = camera.camera_id
                stat = self.stats[cid]
                elapsed = max(0.001, now - stat.last_report)
                rtsp_fps = (stat.rtsp_frames - stat.rtsp_last) / elapsed
                decoded_fps = (stat.decoded_frames - stat.decoded_last) / elapsed
                detector_hz = (stat.processed - stat.processed_last) / elapsed
                stat.rtsp_last = stat.rtsp_frames
                stat.decoded_last = stat.decoded_frames
                stat.processed_last = stat.processed
                stat.last_report = now
                in_depth = int(self.input_queues[cid].get_property("current-level-buffers")) if cid in self.input_queues else 0
                out_depth = int(self.output_queues[cid].get_property("current-level-buffers")) if cid in self.output_queues else 0
                stat.queue_max = max(stat.queue_max, in_depth, out_depth)
                camera_rows.append(
                    f"{cid}:rtsp={rtsp_fps:.1f}/decode={decoded_fps:.1f}/detect={detector_hz:.2f}Hz/"
                    f"pts={_pct(stat.pts_delta_ms, 0.50):.1f}/{_pct(stat.pts_delta_ms, 0.95):.1f}ms/"
                    f"jitter95={_jitter_p95(stat.pts_delta_ms):.1f}ms/timeouts={stat.timeouts}/stale={stat.stale_drops}"
                )
                queue_rows.append(
                    f"{cid}:in={in_depth},out={out_depth},app=0,max={stat.queue_max},gate_drop={stat.gate_drops}"
                )
        print("CAMERA_V11_STEP2_SOURCE " + " | ".join(camera_rows), flush=True)
        print(
            "CAMERA_V11_STEP2_BACKLOG "
            + " | ".join(queue_rows)
            + " configured_max=1 app_pending_max=0 old_frame_retry=0",
            flush=True,
        )
        fields = []
        for name, values in self.stage_values.items():
            fields.append(f"{name}_p50={_pct(values, 0.50):.2f}ms")
            fields.append(f"{name}_p95={_pct(values, 0.95):.2f}ms")
        print("CAMERA_V11_STEP2_PROFILE " + " ".join(fields), flush=True)
        self.report_at = now

    def _warmup(self) -> None:
        if self.detector is None or self.detector.content is None:
            return
        self.detector.content.fill(114)
        for _ in range(10):
            self.detector.infer_preloaded(self.conf, self.max_det)
        print("CAMERA_V11_STEP2_WARMUP iterations=10 status=OK", flush=True)

    def run(self) -> int:
        needs_trt = self.mode in {"synthetic-trt", "full"}
        if needs_trt:
            self.detector = Step2TRT86Client()
            self._warmup()
        if self.mode != "synthetic-trt":
            self._start_ingest()
        if self.mode in {"extraction", "preprocessing"}:
            self._local_content.fill(114)

        ids = [camera.camera_id for camera in self.cameras]
        camera_index = 0
        next_slot = time.monotonic()
        while not self.stop_requested:
            self._poll_bus()
            now = time.monotonic()
            if now < next_slot:
                time.sleep(min(0.005, next_slot - now))
                continue
            cycle_started = time.monotonic()
            self.stage_values["schedule_wait"].append(max(0.0, (cycle_started - next_slot) * 1000.0))
            next_slot = cycle_started + self.global_period  # no accumulated schedule debt
            cid = ids[camera_index]
            camera_index = (camera_index + 1) % len(ids)

            if self.mode == "synthetic-trt":
                accepted_ns = time.monotonic_ns()
                capture_wait = conversion_ms = copy_ms = 0.0
            else:
                accepted_ns, capture_wait, conversion_ms, copy_ms = self._capture_into_preallocated(cid)
                self.stage_values["capture_wait"].append(capture_wait)
                self.stage_values["nvmm_resize_bgrx"].append(conversion_ms)
                self.stage_values["map_copy"].append(copy_ms)
                if accepted_ns is None:
                    if time.monotonic() - self.report_at >= 5.0:
                        self._print_stats()
                    continue
                age_ms = max(0.0, (time.monotonic_ns() - accepted_ns) / 1_000_000.0)
                if age_ms > self.max_input_age_ms:
                    with self.lock:
                        self.stats[cid].stale_drops += 1
                    continue

            if self.mode == "preprocessing":
                self.stage_values["preprocess"].append(self._local_preprocess())
            elif needs_trt:
                started = time.perf_counter()
                result = self.detector.infer_preloaded(self.conf, self.max_det)
                roundtrip = (time.perf_counter() - started) * 1000.0
                self.stage_values["ipc_roundtrip"].append(roundtrip)
                worker_total = float(result.stages.get("total_ms", 0.0))
                self.stage_values["cpu_ipc_block"].append(max(0.0, roundtrip - worker_total))
                self._append_worker_stages(result.stages)

            result_age = max(0.0, (time.monotonic_ns() - accepted_ns) / 1_000_000.0)
            self.stage_values["result_age"].append(result_age)
            with self.lock:
                self.stats[cid].processed += 1
            if time.monotonic() - self.report_at >= 5.0:
                self._print_stats()
        return 0

    def close(self) -> None:
        self.stop_requested = True
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)


def main() -> int:
    service = V11Step2ProductionFP32()

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
