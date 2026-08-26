from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue as pyqueue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from services.ml_service.app.config import CameraConfig, load_settings

from .native_bridge import NativeMetaBridge
from .yolo_trt86_shm_bridge import yolo_trt86_shm_worker


ROOT = Path(__file__).resolve().parents[2]
RESTART_EXIT_CODE = 75

DETECT_W = 672
DETECT_H = 384
DETECT_CONTENT_H = 378


@dataclass
class SourceStats:
    frames: int = 0
    last_frames: int = 0
    last_stat_time: float = field(default_factory=time.monotonic)
    last_pts_ns: int | None = None
    intervals_ms: deque[float] = field(default_factory=lambda: deque(maxlen=240))
    caps: str = "pending"


class FreshFrameMailbox:
    """Single latest detector frame per camera; never a backlog."""

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.rows: dict[str, tuple[int, float, np.ndarray]] = {}
        self.versions: dict[str, int] = {}
        self.closed = False

    def put(self, cid: str, captured: float, frame: np.ndarray) -> None:
        with self.cv:
            version = self.versions.get(cid, 0) + 1
            self.versions[cid] = version
            self.rows[cid] = (version, captured, frame)
            self.cv.notify_all()

    def wait_new(self, cid: str, old_version: int, timeout: float):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not self.closed:
                row = self.rows.get(cid)
                if row is not None and row[0] > old_version:
                    return row
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cv.wait(remaining)
        return None

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()


class CleanCameraRuntime:
    """One decode, three isolated branches, no inheritance stack.

    Per camera:
        nvurisrcbin/NVDEC -> tee
          -> display queue  -> display mux  -> tiler -> OSD -> EGL
          -> tracker queue  -> 10 Hz gate   -> analytics mux -> NvDCF -> fakesink
          -> detector queue -> JIT gate     -> 672x378 BGRx appsink -> TRT8.6 sidecar

    The key production invariant is that the visible wall never passes through
    NvDCF or TensorRT. Tracker work can slow or restart without forcing camera
    presentation to the same cadence. Decoder output is shared once by a tee.
    """

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self.Gst = Gst
        self.GLib = GLib

        self.settings = load_settings()
        self.cameras = list(self.settings.cameras)
        if not 1 <= len(self.cameras) <= 16:
            raise RuntimeError(f"Camera V2 requires 1..16 cameras, got {len(self.cameras)}")

        ds = self.settings.deepstream
        self.gpu_id = int(os.environ.get("CAMERA_V2_GPU_ID", ds.gpu_id))
        self.rtsp_latency_ms = max(
            40,
            int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "80")),
        )
        self.udp_buffer_size = max(
            1_048_576,
            int(os.environ.get("CAMERA_V2_UDP_BUFFER_SIZE", str(8 * 1024 * 1024))),
        )
        self.extra_surfaces = max(
            2,
            min(12, int(os.environ.get("CAMERA_V2_EXTRA_SURFACES", "4"))),
        )
        self.source_fps = max(
            1,
            int(os.environ.get("CAMERA_V2_SOURCE_FPS", "20")),
        )
        self.display_width = max(
            640,
            int(os.environ.get("CAMERA_V2_DISPLAY_WIDTH", "1280")),
        )
        self.display_height = max(
            360,
            int(os.environ.get("CAMERA_V2_DISPLAY_HEIGHT", "720")),
        )
        self.wall_width = max(
            960,
            int(os.environ.get("CAMERA_V2_WALL_WIDTH", "1920")),
        )
        self.wall_height = max(
            360,
            int(os.environ.get("CAMERA_V2_WALL_HEIGHT", "720")),
        )
        self.track_width = DETECT_W
        self.track_height = DETECT_H
        self.track_fps = max(
            2.0,
            min(
                float(self.source_fps),
                float(os.environ.get("CAMERA_V2_TRACK_FPS", "10")),
            ),
        )
        self.detect_hz = max(
            0.1,
            min(2.0, float(os.environ.get("CAMERA_V2_DETECT_HZ", "0.40"))),
        )
        self.max_result_age_ms = max(
            200.0,
            float(os.environ.get("CAMERA_V2_MAX_DETECT_RESULT_AGE_MS", "350")),
        )
        self.display_track_max_age_ms = max(
            150.0,
            float(os.environ.get("CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS", "450")),
        )
        self.detect_enabled = self._env_bool("CAMERA_V2_DETECT_ENABLED", True)
        self.analytics_enabled = self._env_bool("CAMERA_V2_ANALYTICS_ENABLED", True)
        self.startup_stagger_s = max(
            0.10,
            min(
                3.0,
                float(os.environ.get("CAMERA_V2_STARTUP_STAGGER_SEC", "0.50")),
            ),
        )
        self.stall_s = max(
            8.0,
            float(os.environ.get("CAMERA_V2_SOURCE_STALL_SEC", "12")),
        )

        self.stats = {camera.camera_id: SourceStats() for camera in self.cameras}
        self.camera_index = {
            camera.camera_id: index for index, camera in enumerate(self.cameras)
        }
        self.index_camera = {index: cid for cid, index in self.camera_index.items()}

        self.sources: dict[str, object] = {}
        self.tees: dict[str, object] = {}
        self.display_queues: dict[str, object] = {}
        self.tracker_queues: dict[str, object] = {}
        self.detector_queues: dict[str, object] = {}
        self._request_pads: list[tuple[object, object]] = []
        self._warning_last: dict[str, float] = {}

        self.track_gate_lock = threading.Lock()
        self.track_last_pts_ns: dict[str, int] = {}
        self.track_last_mono: dict[str, float] = {}
        self.track_buffers_passed = 0

        self.capture_lock = threading.Lock()
        self.capture_requested: dict[str, bool] = {
            camera.camera_id: False for camera in self.cameras
        }
        self.mailbox = FreshFrameMailbox()
        self.capture_gate_logged: set[str] = set()
        self.capture_layout_logged: set[str] = set()

        self.pending_lock = threading.RLock()
        self.pending_seq = 0
        self.pending: dict[
            str,
            tuple[int, float, list[tuple[float, float, float, float, float]]],
        ] = {}
        self.injected_seq = {camera.camera_id: 0 for camera in self.cameras}
        self.stale_results = 0
        self.detector_frames_applied = 0

        self.track_cache_lock = threading.RLock()
        self.track_cache: dict[
            int,
            tuple[
                float,
                list[tuple[int, float, float, float, float, float]],
            ],
        ] = {}
        self.tracked_now = 0
        self.tracker_batches = 0
        self.tracker_batches_last = 0
        self.tracker_stat_time = time.monotonic()

        self.det_lock = threading.RLock()
        self.det_ready = False
        self.det_calls = 0
        self.det_inputs = 0
        self.det_batch_ms = 0.0
        self.det_result_age_ms = 0.0
        self.det_counts = {camera.camera_id: 0 for camera in self.cameras}
        self.detector_times = {
            camera.camera_id: deque(maxlen=100) for camera in self.cameras
        }
        self.det_error = ""
        self.capture_timeouts = 0
        self.det_stop = threading.Event()
        self.det_process = None
        self.det_thread = None
        self.job_q = None
        self.result_q = None

        self._stopping = False
        self._restart_requested = False
        self._restart_reason = ""
        self._source_started_at: dict[str, float] = {}
        self._source_last_frames: dict[str, int] = {}
        self._source_last_progress: dict[str, float] = {}

        self._preflight()
        self.bridge = NativeMetaBridge()
        self.tracker_lib, self.tracker_config = self._prepare_tracker_files()

        self.pipeline = Gst.Pipeline.new("camera-v2-clean-pascal")
        if self.pipeline is None:
            raise RuntimeError("Could not create GStreamer pipeline")

        self.display_mux = self._make("nvstreammux", "display_mux")
        self.tracker_mux = self._make("nvstreammux", "tracker_mux")
        self.tiler = self._make("nvmultistreamtiler", "display_tiler")
        self.wall_queue = self._make("queue", "display_wall_queue")
        self.wall_convert = self._make("nvvideoconvert", "display_wall_convert")
        self.wall_caps = self._make("capsfilter", "display_rgba_caps")
        self.osd = self._make("nvdsosd", "display_osd")
        self.sink = self._make("nveglglessink", "display_sink")
        self.tracker = self._make("nvtracker", "analytics_nvdcf")
        self.tracker_sink = self._make("fakesink", "analytics_sink")

        self._configure_display_mux()
        self._configure_tracker_mux()
        self._configure_display_path()
        self._configure_tracker_path()

        for element in (
            self.display_mux,
            self.tracker_mux,
            self.tiler,
            self.wall_queue,
            self.wall_convert,
            self.wall_caps,
            self.osd,
            self.sink,
            self.tracker,
            self.tracker_sink,
        ):
            self.pipeline.add(element)

        self._link_display_path()
        self._link_tracker_path()

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        self.display_mux.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._display_overlay_probe,
        )
        self.tracker_mux.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._inject_detector_probe,
        )
        self.tracker.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._tracker_probe,
        )

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()

        self._audit_graph()
        print(
            "CAMERA_CLEAN_ARCH "
            f"sources={len(self.cameras)} decode_once=1 "
            f"display={self.display_width}x{self.display_height}@source "
            f"wall={self.wall_width}x{self.wall_height} "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"detector=TRT8.6/{DETECT_W}x{DETECT_H}@{self.detect_hz:.2f}Hz/cam "
            "display_bypasses_tracker=1 display_bypasses_detector=1",
            flush=True,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _preflight(self) -> None:
        required = (
            "nvurisrcbin",
            "tee",
            "queue",
            "nvstreammux",
            "nvmultistreamtiler",
            "nvvideoconvert",
            "appsink",
            "nvtracker",
            "nvdsosd",
            "nveglglessink",
            "fakesink",
        )
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("Missing DeepStream/GStreamer plugins: " + ", ".join(missing))

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Could not create {factory}:{name}")
        return element

    @staticmethod
    def _set_if(element, prop: str, value) -> bool:
        if element.find_property(prop) is None:
            return False
        element.set_property(prop, value)
        return True

    @staticmethod
    def _require_link(src, dst, label: str) -> None:
        if not src.link(dst):
            raise RuntimeError(f"Failed to link {label}")

    def _transport(self) -> str:
        configured = str(
            getattr(self.settings.deepstream, "rtsp_transport", "auto") or "auto"
        )
        value = os.environ.get("CAMERA_V2_RTSP_TRANSPORT", configured).strip().lower()
        if value not in {"auto", "tcp", "udp"}:
            raise RuntimeError("CAMERA_V2_RTSP_TRANSPORT must be auto, tcp, or udp")
        return value

    def _configure_display_mux(self) -> None:
        mux = self.display_mux
        self._set_if(mux, "batch-size", len(self.cameras))
        self._set_if(mux, "live-source", True)
        self._set_if(mux, "width", self.display_width)
        self._set_if(mux, "height", self.display_height)
        self._set_if(mux, "enable-padding", False)
        self._set_if(mux, "batched-push-timeout", round(1_000_000 / self.source_fps))
        self._set_if(mux, "sync-inputs", False)
        self._set_if(mux, "max-latency", 0)
        self._set_if(mux, "buffer-pool-size", max(8, len(self.cameras) + 2))
        self._set_if(mux, "nvbuf-memory-type", 2)
        self._set_if(mux, "gpu-id", self.gpu_id)
        self._set_if(mux, "compute-hw", 1)
        # Cubic is the quality/performance compromise for presentation scaling.
        self._set_if(mux, "interpolation-method", 2)

    def _configure_tracker_mux(self) -> None:
        mux = self.tracker_mux
        self._set_if(mux, "batch-size", len(self.cameras))
        self._set_if(mux, "live-source", True)
        self._set_if(mux, "width", self.track_width)
        self._set_if(mux, "height", self.track_height)
        self._set_if(mux, "enable-padding", False)
        self._set_if(mux, "batched-push-timeout", round(1_000_000 / self.track_fps))
        self._set_if(mux, "sync-inputs", False)
        self._set_if(mux, "max-latency", 0)
        self._set_if(mux, "buffer-pool-size", max(8, len(self.cameras) + 2))
        self._set_if(mux, "nvbuf-memory-type", 2)
        self._set_if(mux, "gpu-id", self.gpu_id)
        self._set_if(mux, "compute-hw", 1)
        # Analytics is intentionally cheaper than presentation.
        self._set_if(mux, "interpolation-method", 1)

    def _configure_display_path(self) -> None:
        rows = max(1, math.ceil(len(self.cameras) / 3))
        columns = min(3, len(self.cameras))
        self._set_if(self.tiler, "rows", rows)
        self._set_if(self.tiler, "columns", columns)
        self._set_if(self.tiler, "width", self.wall_width)
        self._set_if(self.tiler, "height", self.wall_height)
        self._set_if(self.tiler, "gpu-id", self.gpu_id)
        self._set_if(self.tiler, "nvbuf-memory-type", 2)
        self._set_if(self.tiler, "compute-hw", 1)
        self._set_if(self.tiler, "interpolation-method", 2)

        self._set_latest_queue(self.wall_queue)
        self._set_if(self.wall_convert, "gpu-id", self.gpu_id)
        self.wall_caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(self.osd, "process-mode", 1)
        self._set_if(self.osd, "display-bbox", True)
        self._set_if(self.osd, "display-text", True)
        self._set_if(self.osd, "display-mask", False)
        self._set_if(self.osd, "gpu-id", self.gpu_id)

        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "async", False)
        self._set_if(self.sink, "enable-last-sample", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "render-delay", 0)
        self._set_if(self.sink, "throttle-time", 0)
        self._set_if(self.sink, "force-aspect-ratio", True)
        self._set_if(self.sink, "gpu-id", self.gpu_id)

    def _configure_tracker_path(self) -> None:
        self._set_if(self.tracker, "tracker-width", self.track_width)
        self._set_if(self.tracker, "tracker-height", self.track_height)
        self.tracker.set_property("ll-lib-file", str(self.tracker_lib))
        self.tracker.set_property("ll-config-file", str(self.tracker_config))
        self._set_if(self.tracker, "gpu-id", self.gpu_id)
        self._set_if(self.tracker, "compute-hw", 1)
        self._set_if(self.tracker, "enable-batch-process", True)
        self._set_if(self.tracker, "display-tracking-id", False)
        self._set_if(self.tracker, "tracking-id-reset-mode", 1)
        self._set_if(self.tracker, "user-meta-pool-size", 64)
        self._set_if(self.tracker_sink, "sync", False)
        self._set_if(self.tracker_sink, "async", False)
        self._set_if(self.tracker_sink, "qos", False)

    def _link_display_path(self) -> None:
        self._require_link(self.display_mux, self.tiler, "display mux -> tiler")
        self._require_link(self.tiler, self.wall_queue, "tiler -> display queue")
        self._require_link(self.wall_queue, self.wall_convert, "display queue -> convert")
        self._require_link(self.wall_convert, self.wall_caps, "convert -> RGBA caps")
        self._require_link(self.wall_caps, self.osd, "RGBA -> OSD")
        self._require_link(self.osd, self.sink, "OSD -> EGL")

    def _link_tracker_path(self) -> None:
        self._require_link(self.tracker_mux, self.tracker, "analytics mux -> NvDCF")
        self._require_link(self.tracker, self.tracker_sink, "NvDCF -> fakesink")

    def _set_latest_queue(self, queue) -> None:
        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

    def _request_mux_pad(self, mux, index: int):
        name = f"sink_{index}"
        request_simple = getattr(mux, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            pad = mux.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"{mux.get_name()} could not allocate {name}")
        self._request_pads.append((mux, pad))
        return pad

    def _request_tee_pad(self, tee):
        request_simple = getattr(tee, "request_pad_simple", None)
        pad = request_simple("src_%u") if request_simple else None
        if pad is None:
            pad = tee.get_request_pad("src_%u")
        if pad is None:
            raise RuntimeError(f"{tee.get_name()} could not allocate src_%u")
        self._request_pads.append((tee, pad))
        return pad

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera: CameraConfig) -> None:
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return

        transport = self._transport()
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)
        if transport == "tcp":
            self._set_if(element, "protocols", 4)
            # Receive-time timestamps prevent long-running TCP clock drift. The
            # bounded jitterbuffer below still owns burst absorption.
            self._set_if(element, "tcp-timestamp", True)
        elif transport == "udp":
            self._set_if(element, "protocols", 1)

        self._set_if(element, "latency", self.rtsp_latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "buffer-mode", 3)
        self._set_if(element, "do-rtsp-keep-alive", True)
        print(
            f"CAMERA_CLEAN_RTSP {camera.camera_id} "
            f"transport={transport} latency={self.rtsp_latency_ms}ms "
            f"auth={'yes' if camera.username else 'no'}",
            flush=True,
        )

    def _add_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        source = self._make("nvurisrcbin", f"source_{index}")
        tee = self._make("tee", f"tee_{index}")
        display_q = self._make("queue", f"display_q_{index}")
        tracker_q = self._make("queue", f"tracker_q_{index}")
        detector_q = self._make("queue", f"detector_q_{index}")
        detector_convert = self._make("nvvideoconvert", f"detector_convert_{index}")
        detector_caps = self._make("capsfilter", f"detector_caps_{index}")
        detector_sink = self._make("appsink", f"detector_sink_{index}")

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", camera.uri)
        transport = self._transport()
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", 3)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)
        self._set_if(source, "gpu-id", self.gpu_id)

        for q in (display_q, tracker_q, detector_q):
            self._set_latest_queue(q)

        self._set_if(detector_convert, "gpu-id", self.gpu_id)
        self._set_if(detector_convert, "compute-hw", 1)
        self._set_if(detector_convert, "interpolation-method", 2)
        detector_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={DETECT_W},height={DETECT_CONTENT_H},"
                "pixel-aspect-ratio=1/1"
            ),
        )
        detector_sink.set_property("emit-signals", True)
        detector_sink.set_property("sync", False)
        detector_sink.set_property("async", False)
        detector_sink.set_property("drop", True)
        detector_sink.set_property("max-buffers", 1)
        self._set_if(detector_sink, "wait-on-eos", False)
        self._set_if(detector_sink, "enable-last-sample", False)

        for element in (
            source,
            tee,
            display_q,
            tracker_q,
            detector_q,
            detector_convert,
            detector_caps,
            detector_sink,
        ):
            self.pipeline.add(element)

        display_tee = self._request_tee_pad(tee)
        tracker_tee = self._request_tee_pad(tee)
        detector_tee = self._request_tee_pad(tee)
        if display_tee.link(display_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> display queue failed")
        if tracker_tee.link(tracker_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> tracker queue failed")
        if detector_tee.link(detector_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> detector queue failed")

        display_mux_pad = self._request_mux_pad(self.display_mux, index)
        if display_q.get_static_pad("src").link(display_mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display queue -> display mux failed")

        tracker_mux_pad = self._request_mux_pad(self.tracker_mux, index)
        if tracker_q.get_static_pad("src").link(tracker_mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tracker queue -> analytics mux failed")

        if not detector_q.link(detector_convert):
            raise RuntimeError(f"{cid}: detector queue -> converter failed")
        if not detector_convert.link(detector_caps):
            raise RuntimeError(f"{cid}: detector converter -> caps failed")
        if not detector_caps.link(detector_sink):
            raise RuntimeError(f"{cid}: detector caps -> appsink failed")

        tee.get_static_pad("sink").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._source_probe,
            cid,
        )
        tracker_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._tracker_rate_probe,
            cid,
        )
        detector_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._detector_gate_probe,
            cid,
        )
        detector_sink.connect("new-sample", self._on_detector_sample, cid)
        source.connect("pad-added", self._source_pad_added, tee, cid)

        self.sources[cid] = source
        self.tees[cid] = tee
        self.display_queues[cid] = display_q
        self.tracker_queues[cid] = tracker_q
        self.detector_queues[cid] = detector_q

    def _source_pad_added(self, _source, pad, tee, cid: str) -> None:
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
            print(
                f"CAMERA_CLEAN {cid} source->tee failed result={result}",
                file=sys.stderr,
                flush=True,
            )
            return
        print(f"CAMERA_CLEAN {cid} source->tee linked", flush=True)

    def _source_probe(self, pad, info, cid: str):
        stat = self.stats[cid]
        stat.frames += 1
        buffer = info.get_buffer()
        if stat.caps == "pending":
            caps = pad.get_current_caps()
            if caps is not None:
                stat.caps = caps.to_string()
                print(f"CAMERA_CLEAN_SOURCE {cid} negotiated={stat.caps}", flush=True)
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts_ns = int(buffer.pts)
            previous = stat.last_pts_ns
            stat.last_pts_ns = pts_ns
            if previous is not None and pts_ns > previous:
                interval = (pts_ns - previous) / 1_000_000.0
                if 0.0 < interval < 1000.0:
                    stat.intervals_ms.append(interval)
        return self.Gst.PadProbeReturn.OK

    def _tracker_rate_probe(self, _pad, info, cid: str):
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.DROP
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.DROP

        period_ns = int(round(1_000_000_000.0 / self.track_fps))
        accept = False
        with self.track_gate_lock:
            if buffer.pts != self.Gst.CLOCK_TIME_NONE:
                pts = int(buffer.pts)
                previous = self.track_last_pts_ns.get(cid)
                if previous is None or pts - previous >= int(period_ns * 0.90):
                    self.track_last_pts_ns[cid] = pts
                    accept = True
            else:
                now = time.monotonic()
                previous = self.track_last_mono.get(cid)
                if previous is None or now - previous >= 0.90 / self.track_fps:
                    self.track_last_mono[cid] = now
                    accept = True
            if accept:
                self.track_buffers_passed += 1
        return self.Gst.PadProbeReturn.OK if accept else self.Gst.PadProbeReturn.DROP

    def _detector_gate_probe(self, _pad, _info, cid: str):
        if not self.detect_enabled:
            return self.Gst.PadProbeReturn.DROP
        with self.capture_lock:
            if not self.capture_requested.get(cid, False):
                return self.Gst.PadProbeReturn.DROP
            self.capture_requested[cid] = False
        if cid not in self.capture_gate_logged:
            self.capture_gate_logged.add(cid)
            print(f"CAMERA_CLEAN_DETECT_GATE {cid} first_buffer=1", flush=True)
        return self.Gst.PadProbeReturn.OK

    def _on_detector_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            with self.capture_lock:
                self.capture_requested[cid] = True
            return self.Gst.FlowReturn.OK
        try:
            structure = sample.get_caps().get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            if width != DETECT_W or height != DETECT_CONTENT_H:
                raise RuntimeError(
                    f"{cid}: detector capture={width}x{height}, "
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
                        f"{cid}: BGRx too small {mapped_size} < {tight_stride * height}"
                    )
                row_stride = mapped_size // height if mapped_size % height == 0 else tight_stride
                if row_stride < tight_stride:
                    raise RuntimeError(
                        f"{cid}: invalid stride={row_stride}, tight={tight_stride}"
                    )
                raw = np.frombuffer(
                    mapped.data,
                    dtype=np.uint8,
                    count=row_stride * height,
                )
                bgrx = raw.reshape((height, row_stride))[:, :tight_stride].reshape(
                    (height, width, 4)
                )
                frame = np.full((DETECT_H, DETECT_W, 3), 114, dtype=np.uint8)
                frame[3:381, :, :] = bgrx[..., :3]
            finally:
                buffer.unmap(mapped)

            captured = time.monotonic()
            self.mailbox.put(cid, captured, frame)
            if cid not in self.capture_layout_logged:
                self.capture_layout_logged.add(cid)
                print(
                    f"CAMERA_CLEAN_DETECT_LAYOUT {cid} capture={width}x{height} "
                    f"stride={row_stride} tensor={DETECT_W}x{DETECT_H} "
                    "letterbox=3+378+3 pad114",
                    flush=True,
                )
        except Exception as exc:
            with self.capture_lock:
                self.capture_requested[cid] = True
            print(
                f"CAMERA_CLEAN_DETECT_CAPTURE {cid} "
                f"warning={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        return self.Gst.FlowReturn.OK

    def _request_capture(self, cid: str) -> None:
        with self.capture_lock:
            self.capture_requested[cid] = True

    def _clear_capture(self, cid: str) -> None:
        with self.capture_lock:
            self.capture_requested[cid] = False

    @staticmethod
    def _area(box) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _intersection(a, b) -> float:
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0.0,
            min(a[3], b[3]) - max(a[1], b[1]),
        )

    def _map_detector_rows(self, rows):
        mapped: list[tuple[tuple[float, float, float, float], float]] = []
        y_scale = self.track_height / float(DETECT_CONTENT_H)
        for coords, conf in rows:
            x1, y1, x2, y2 = [float(v) for v in coords]
            x1 = max(0.0, min(float(self.track_width - 1), x1))
            x2 = max(0.0, min(float(self.track_width - 1), x2))
            y1 = max(0.0, min(float(self.track_height - 1), (y1 - 3.0) * y_scale))
            y2 = max(0.0, min(float(self.track_height - 1), (y2 - 3.0) * y_scale))
            if x2 <= x1 or y2 <= y1:
                continue
            mapped.append(((x1, y1, x2, y2), float(conf)))

        mapped.sort(key=lambda item: item[1], reverse=True)
        kept: list[tuple[tuple[float, float, float, float], float]] = []
        for box, conf in mapped:
            duplicate = False
            area = max(1.0, self._area(box))
            for other, _ in kept:
                inter = self._intersection(box, other)
                union = area + self._area(other) - inter
                iou = inter / union if union > 0 else 0.0
                containment = inter / max(1.0, min(area, self._area(other)))
                if iou >= 0.82 or containment >= 0.94:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((box, conf))
        return [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2), conf in kept]

    def _publish_detector(self, cid: str, captured: float, boxes) -> None:
        with self.pending_lock:
            self.pending_seq += 1
            self.pending[cid] = (self.pending_seq, float(captured), list(boxes))

    def _inject_detector_probe(self, _pad, info):
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.pending_lock:
            pending = dict(self.pending)

        applied = 0
        stale = 0
        max_age = 0.0
        for cid, source_id in self.camera_index.items():
            row = pending.get(cid)
            if row is None:
                continue
            seq, captured, boxes = row
            if seq <= self.injected_seq.get(cid, 0):
                continue
            age_ms = max(0.0, (now - captured) * 1000.0)
            if age_ms > self.max_result_age_ms:
                self.injected_seq[cid] = seq
                stale += 1
                continue
            result = self.bridge.apply_detector_result(buffer, source_id, boxes)
            if result == -2:
                continue
            if result < 0:
                continue
            self.injected_seq[cid] = seq
            applied += 1
            max_age = max(max_age, age_ms)

        if applied or stale:
            with self.det_lock:
                self.detector_frames_applied += applied
                self.stale_results += stale
                self.det_result_age_ms = max_age
        return self.Gst.PadProbeReturn.OK

    def _tracker_probe(self, _pad, info):
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            rows = self.bridge.copy_tracks(buffer, max_rows=256)
            now = time.monotonic()
            grouped: dict[int, list[tuple[int, float, float, float, float, float]]] = {}
            sx = self.display_width / float(self.track_width)
            sy = self.display_height / float(self.track_height)
            for row in rows:
                source_id = int(row["source_id"])
                left = float(row["left"]) * sx
                top = float(row["top"]) * sy
                right = (float(row["left"]) + float(row["width"])) * sx
                bottom = (float(row["top"]) + float(row["height"])) * sy
                conf = float(row["tracker_confidence"])
                if conf < 0.0:
                    conf = float(row["confidence"])
                grouped.setdefault(source_id, []).append(
                    (
                        int(row["object_id"]),
                        left,
                        top,
                        right,
                        bottom,
                        conf,
                    )
                )
            with self.track_cache_lock:
                for source_id, tracks in grouped.items():
                    self.track_cache[source_id] = (now, tracks)
                self.tracked_now = sum(len(tracks) for tracks in grouped.values())
                self.tracker_batches += 1
        except Exception as exc:
            print(
                f"CAMERA_CLEAN_TRACK warning={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        return self.Gst.PadProbeReturn.OK

    def _display_overlay_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.track_cache_lock:
            cache = dict(self.track_cache)
        for source_id in self.index_camera:
            row = cache.get(source_id)
            if row is None:
                continue
            updated, tracks = row
            age_ms = (now - updated) * 1000.0
            if age_ms > self.display_track_max_age_ms:
                continue
            self.bridge.add_tracked_boxes(buffer, source_id, tracks)
        self.bridge.apply_local_track_style(buffer)
        return self.Gst.PadProbeReturn.OK

    def _start_detector(self) -> None:
        if not self.detect_enabled:
            print("CAMERA_CLEAN_DETECT enabled=0", flush=True)
            return
        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.det_process = ctx.Process(
            target=yolo_trt86_shm_worker,
            args=(self.job_q, self.result_q),
            name="camera-v2-trt86-bridge",
        )
        self.det_process.start()
        self.det_thread = threading.Thread(
            target=self._detector_scheduler,
            name="camera-v2-detector-scheduler",
            daemon=True,
        )
        self.det_thread.start()

    def _detector_scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "TRT86 startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "TRT86 startup failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_CLEAN_DETECT_READY "
            f"model={ready.get('model')} backend={ready.get('backend')} "
            f"target={self.detect_hz:.2f}Hz/cam capture=jit-latest-no-prefetch",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        versions = {cid: 0 for cid in ids}
        index = 0
        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            cid = ids[index % len(ids)]
            index += 1

            # Do not create detector timeout noise before a staggered source has
            # produced its first decoded frame.
            if self.stats[cid].frames <= 0:
                self.det_stop.wait(0.03)
                continue

            self._request_capture(cid)
            row = self.mailbox.wait_new(cid, versions[cid], timeout=0.8)
            if row is None:
                self._clear_capture(cid)
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.025)
                continue

            version, captured, frame = row
            versions[cid] = version
            self._clear_capture(cid)
            try:
                self.job_q.put(
                    {"cameras": [cid], "frames": [frame], "captured": [captured]},
                    timeout=0.3,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "TRT86 result timeout"
                self.det_stop.wait(0.05)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "TRT86 fatal")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "TRT86 batch error")
                self.det_stop.wait(0.10)
                continue
            if result.get("type") != "result":
                continue

            completed = time.monotonic()
            rows = result.get("boxes", {}).get(cid, [])
            boxes = self._map_detector_rows(rows)
            self._publish_detector(cid, captured, boxes)
            age_ms = max(0.0, (completed - captured) * 1000.0)
            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += 1
                self.det_batch_ms = batch_ms
                self.det_counts[cid] = len(boxes)
                self.det_result_age_ms = age_ms
                self.detector_times[cid].append(completed)
                self.det_error = ""

            desired_interval = 1.0 / max(0.1, self.detect_hz * len(ids))
            elapsed = time.monotonic() - cycle_started
            self.det_stop.wait(max(0.005, desired_interval - elapsed))

    @staticmethod
    def _deepstream_roots() -> list[Path]:
        roots = [Path("/opt/nvidia/deepstream/deepstream")]
        roots.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
        output: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key not in seen:
                seen.add(key)
                output.append(root)
        return output

    @staticmethod
    def _replace_yaml_key(lines: list[str], key: str, value: str, required: bool = True) -> bool:
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped.startswith(key + ":"):
                continue
            indent = line[: len(line) - len(stripped)]
            comment = ""
            if "#" in stripped:
                comment = "  #" + stripped.split("#", 1)[1]
            lines[index] = f"{indent}{key}: {value}{comment}"
            return True
        if required:
            raise RuntimeError(f"NvDCF config missing key: {key}")
        return False

    @staticmethod
    def _insert_target_management_key(lines: list[str], key: str, value: str) -> None:
        start = None
        end = len(lines)
        for i, line in enumerate(lines):
            if line and not line[0].isspace() and line.split("#", 1)[0].strip() == "TargetManagement:":
                start = i
                break
        if start is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend(["TargetManagement:", f"  {key}: {value}"])
            return
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line and not line[0].isspace() and line.split("#", 1)[0].strip().endswith(":"):
                end = i
                break
        for i in range(start + 1, end):
            stripped = lines[i].lstrip()
            if stripped.startswith(key + ":"):
                indent = lines[i][: len(lines[i]) - len(stripped)] or "  "
                lines[i] = f"{indent}{key}: {value}"
                return
        lines.insert(end, f"  {key}: {value}")

    def _prepare_tracker_files(self) -> tuple[Path, Path]:
        lib = next(
            (
                root / "lib/libnvds_nvmultiobjecttracker.so"
                for root in self._deepstream_roots()
                if (root / "lib/libnvds_nvmultiobjecttracker.so").exists()
            ),
            None,
        )
        stock = next(
            (
                root / "samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml"
                for root in self._deepstream_roots()
                if (root / "samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml").exists()
            ),
            None,
        )
        if lib is None or stock is None:
            raise RuntimeError("NvDCF library/max_perf config not found")

        runtime_dir = ROOT / ".runtime" / "camera_v2"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        generated = runtime_dir / "config_tracker_clean.yml"
        lines = stock.read_text(encoding="utf-8").splitlines()
        self._replace_yaml_key(lines, "enableBboxUnClipping", "0")
        self._replace_yaml_key(lines, "minIouDiff4NewTarget", "0.45")
        self._replace_yaml_key(lines, "minTrackerConfidence", "0.05")
        self._replace_yaml_key(lines, "probationAge", "0")
        # Tracker branch is 10 Hz by default: 70 frames ~= seven seconds.
        shadow_frames = max(20, int(round(self.track_fps * 7.0)))
        self._replace_yaml_key(lines, "maxShadowTrackingAge", str(shadow_frames))
        self._replace_yaml_key(lines, "earlyTerminationAge", "3")
        self._replace_yaml_key(lines, "minIou4TargetDuplicate", "0.94", required=False)
        self._replace_yaml_key(lines, "targetDuplicateRunInterval", "5", required=False)
        self._insert_target_management_key(lines, "outputShadowTracks", "1")
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_CLEAN_NVDCF "
            f"profile=max_perf tracker={self.track_width}x{self.track_height} "
            f"input_hz={self.track_fps:.1f} shadow_frames={shadow_frames} "
            "outputShadowTracks=1",
            flush=True,
        )
        return lib, generated

    @staticmethod
    def _percentile(values: deque[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
        return float(ordered[index])

    def _detector_actual_hz(self, cid: str, now: float) -> float:
        times = self.detector_times[cid]
        while times and now - times[0] > 20.0:
            times.popleft()
        if len(times) < 2:
            return 0.0
        span = max(0.001, times[-1] - times[0])
        return (len(times) - 1) / span

    def _sink_stats(self) -> tuple[int | None, int | None]:
        if self.sink.find_property("stats") is None:
            return None, None
        try:
            stats = self.sink.get_property("stats")
            rendered = int(stats.get_value("rendered")) if stats.has_field("rendered") else None
            dropped = int(stats.get_value("dropped")) if stats.has_field("dropped") else None
            return rendered, dropped
        except Exception:
            return None, None

    def _print_stats(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        parts = []
        for camera in self.cameras:
            cid = camera.camera_id
            stat = self.stats[cid]
            elapsed = max(0.001, now - stat.last_stat_time)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_stat_time = now
            p50 = self._percentile(stat.intervals_ms, 0.50)
            p95 = self._percentile(stat.intervals_ms, 0.95)
            cadence = "?" if p50 is None else f"{p50:.0f}/{p95:.0f}ms"
            dq = int(self.display_queues[cid].get_property("current-level-buffers"))
            tq = int(self.tracker_queues[cid].get_property("current-level-buffers"))
            parts.append(f"{cid}:{fps:.1f}fps pts={cadence} q={dq}/{tq}")

        tracker_elapsed = max(0.001, now - self.tracker_stat_time)
        tracker_fps = (self.tracker_batches - self.tracker_batches_last) / tracker_elapsed
        self.tracker_batches_last = self.tracker_batches
        self.tracker_stat_time = now
        rendered, dropped = self._sink_stats()
        with self.det_lock:
            actual = " ".join(
                f"{cid}:{self._detector_actual_hz(cid, now):.2f}"
                for cid in self.camera_index
            )
            det = (
                f"ready={int(self.det_ready)} calls={self.det_calls} "
                f"batch={self.det_batch_ms:.1f}ms age={self.det_result_age_ms:.0f}ms "
                f"actual=[{actual}] timeouts={self.capture_timeouts} "
                f"stale={self.stale_results} error={self.det_error or '-'}"
            )
        print(
            "CAMERA_CLEAN_STATS "
            + " | ".join(parts)
            + f" || display_rendered={rendered} dropped={dropped} "
            + f"tracker_batches={self.tracker_batches} tracker_rate={tracker_fps:.1f}Hz "
            + f"tracked_now={self.tracked_now} detector={det}",
            flush=True,
        )
        return True

    def _audit_graph(self) -> None:
        # Static graph contracts: presentation and analytics must not be serial.
        display_peer = self.display_mux.get_static_pad("src").get_peer()
        tracker_peer = self.tracker_mux.get_static_pad("src").get_peer()
        display_name = (
            display_peer.get_parent_element().get_name() if display_peer is not None else None
        )
        tracker_name = (
            tracker_peer.get_parent_element().get_name() if tracker_peer is not None else None
        )
        if display_name != self.tiler.get_name():
            raise RuntimeError(
                f"CAMERA_CLEAN_AUDIT display mux peer={display_name}, expected={self.tiler.get_name()}"
            )
        if tracker_name != self.tracker.get_name():
            raise RuntimeError(
                f"CAMERA_CLEAN_AUDIT tracker mux peer={tracker_name}, expected={self.tracker.get_name()}"
            )
        if self.pipeline.get_by_name("native_yolo26_pgie") is not None:
            raise RuntimeError("CAMERA_CLEAN_AUDIT Gst-nvinfer must be absent on Pascal")
        for index, camera in enumerate(self.cameras):
            tee = self.pipeline.get_by_name(f"tee_{index}")
            detector_convert = self.pipeline.get_by_name(f"detector_convert_{index}")
            detector_sink = self.pipeline.get_by_name(f"detector_sink_{index}")
            if tee is None or detector_convert is None or detector_sink is None:
                raise RuntimeError(f"CAMERA_CLEAN_AUDIT {camera.camera_id} branch missing")
        print(
            "CAMERA_CLEAN_AUDIT status=OK "
            "order=decode-once->tee->{display,tracker,detector} "
            "display=display_mux->tiler->OSD->EGL "
            "analytics=tracker_mux->NvDCF->fakesink "
            "detector=JIT-gate->TRT86-SHM",
            flush=True,
        )

    def _prepare_staggered_sources(self) -> None:
        ordered = [camera.camera_id for camera in self.cameras]
        for cid in ordered:
            source = self.sources[cid]
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)
        print(
            f"CAMERA_CLEAN_STAGGER order={ordered} interval={self.startup_stagger_s:.2f}s",
            flush=True,
        )
        for index, cid in enumerate(ordered):
            delay_ms = max(1, int(round(index * self.startup_stagger_s * 1000.0)))

            def _start(camera_id=cid, ordinal=index):
                if self._stopping:
                    return False
                source = self.sources[camera_id]
                source.set_locked_state(False)
                sync = bool(source.sync_state_with_parent())
                now = time.monotonic()
                self._source_started_at[camera_id] = now
                self._source_last_progress[camera_id] = now
                self._source_last_frames[camera_id] = self.stats[camera_id].frames
                print(
                    f"CAMERA_CLEAN_SOURCE_START cid={camera_id} index={ordinal} sync={int(sync)}",
                    flush=True,
                )
                return False

            self.GLib.timeout_add(delay_ms, _start)

    def _source_watchdog(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        for cid, started in list(self._source_started_at.items()):
            frames = self.stats[cid].frames
            if frames != self._source_last_frames.get(cid, -1):
                self._source_last_frames[cid] = frames
                self._source_last_progress[cid] = now
                continue
            if now - started < self.stall_s:
                continue
            stalled = now - self._source_last_progress.get(cid, started)
            if stalled < self.stall_s:
                continue
            self._restart_requested = True
            self._restart_reason = f"{cid} no decoded frames for {stalled:.1f}s"
            print(
                f"CAMERA_CLEAN_RESTART reason={self._restart_reason} exit={RESTART_EXIT_CODE}",
                file=sys.stderr,
                flush=True,
            )
            self.stop()
            return False
        return True

    def _on_bus_message(self, _bus, message) -> None:
        msg_type = message.type
        if msg_type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(
                f"CAMERA_CLEAN_GST ERROR {err} debug={debug}",
                file=sys.stderr,
                flush=True,
            )
            self._restart_requested = True
            self._restart_reason = f"gstreamer:{err}"
            self.stop()
        elif msg_type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            key = str(err)
            now = time.monotonic()
            if now - self._warning_last.get(key, 0.0) >= 5.0:
                self._warning_last[key] = now
                print(
                    f"CAMERA_CLEAN_GST WARNING {err} debug={debug}",
                    file=sys.stderr,
                    flush=True,
                )
        elif msg_type == self.Gst.MessageType.EOS:
            self.stop()

    def _install_signals(self) -> None:
        def _handle(_signum, _frame):
            try:
                self.GLib.idle_add(self.stop)
            except Exception:
                self.stop()

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def stop(self) -> bool:
        if self._stopping:
            return False
        self._stopping = True
        self.det_stop.set()
        self.mailbox.close()
        with self.capture_lock:
            for cid in self.capture_requested:
                self.capture_requested[cid] = False
        try:
            if self.job_q is not None:
                self.job_q.put_nowait(None)
        except Exception:
            pass
        try:
            self.loop.quit()
        except Exception:
            pass
        return False

    def _shutdown_detector(self) -> None:
        if self.det_thread is not None:
            self.det_thread.join(timeout=3.0)
        if self.det_process is not None:
            self.det_process.join(timeout=3.0)
            if self.det_process.is_alive():
                self.det_process.terminate()
                self.det_process.join(timeout=2.0)

    def run(self) -> int:
        self._install_signals()
        self._prepare_staggered_sources()
        self._start_detector()
        state = self.pipeline.set_state(self.Gst.State.PLAYING)
        if state == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Camera clean pipeline failed to enter PLAYING")
        self.GLib.timeout_add_seconds(5, self._print_stats)
        self.GLib.timeout_add_seconds(1, self._source_watchdog)
        try:
            self.loop.run()
        finally:
            self.det_stop.set()
            self.mailbox.close()
            self._shutdown_detector()
            self.pipeline.set_state(self.Gst.State.NULL)
            for owner, pad in reversed(self._request_pads):
                try:
                    owner.release_request_pad(pad)
                except Exception:
                    pass
        return RESTART_EXIT_CODE if self._restart_requested else 0


def main() -> int:
    return CleanCameraRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
