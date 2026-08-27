from __future__ import annotations

import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from services.ml_service.app.config import CameraConfig, load_settings
from .pts_bridge import NativePtsBridge


@dataclass
class CameraStats:
    decoded: int = 0
    decoded_last: int = 0
    stat_mono: float = field(default_factory=time.monotonic)
    last_arrival_mono: float | None = None
    last_pts_ns: int | None = None
    wall_dt_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    pts_dt_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))

    # Source-buffer provenance. We intentionally measure freshness before the
    # tiler destroys per-source identity by compositing six inputs into one
    # output frame.
    pts_arrival: dict[int, float] = field(default_factory=dict)
    pts_order: deque[int] = field(default_factory=deque)
    mux_age_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    mux_gap_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    last_mux_update_mono: float | None = None
    displayed_source_arrival_mono: float | None = None
    display_age_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    mux_match_miss: int = 0

    input_qmax: int = 0
    errors: int = 0
    warnings: int = 0
    source_linked: bool = False


class V11Step1DisplayV2:
    """Clean V11 Step1 V2 display baseline.

    This is a new implementation; it does not modify the original Step1.

    Invariants:
      * Step0 ingest policy remains TCP/60 ms, NVDEC, latest-only.
      * No detector, tracker, OSD, ReID, face model, JPEG, OpenCV or CPU copy.
      * One GPU scaling step at nvstreammux directly to the final tile size.
      * GPU Lanczos scaling is used for a slightly sharper 640x360 tile.
      * nvmultistreamtiler is treated as a compositor/cache, not as a reliable
        per-camera metadata stage. NVIDIA composites the batch to one frame.
      * Per-camera display freshness is measured by carrying source PTS arrival
        timestamps through nvstreammux, then measuring the age of the last frame
        cached by the tiler at every render output.
    """

    PTS_HISTORY = 768
    PTS_MATCH_TOLERANCE_NS = 5_000_000

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self.GLib = GLib
        self.Gst = Gst
        self.settings = load_settings()
        self.cameras = list(self.settings.cameras)
        if len(self.cameras) != 6:
            raise RuntimeError(f"V11 Step1 V2 requires exactly 6 cameras, got {len(self.cameras)}")

        ds = self.settings.deepstream
        self.gpu_id = int(os.environ.get("V11_GPU_ID", ds.gpu_id))
        self.transport = os.environ.get("V11_RTSP_TRANSPORT", "tcp").strip().lower()
        if self.transport not in {"tcp", "udp", "auto"}:
            raise RuntimeError("V11_RTSP_TRANSPORT must be tcp, udp, or auto")

        # Frozen Step0 ingest policy. Do not tune camera-specific latency here.
        self.latency_ms = max(20, int(os.environ.get("V11_RTSP_LATENCY_MS", "60")))
        self.drop_on_latency = self._env_bool("V11_DROP_ON_LATENCY", True)
        self.extra_surfaces = max(1, min(12, int(os.environ.get("V11_EXTRA_SURFACES", "4"))))
        self.udp_buffer_size = max(
            1_048_576,
            int(os.environ.get("V11_UDP_BUFFER_SIZE", str(max(ds.udp_buffer_size, 8 * 1024 * 1024)))),
        )
        self.reconnect_sec = max(2, int(os.environ.get("V11_RECONNECT_SEC", "5")))
        self.startup_stagger = max(
            0.1,
            min(2.0, float(os.environ.get("V11_STARTUP_STAGGER_SEC", "0.40"))),
        )
        self.stats_interval = max(2, int(os.environ.get("V11_STATS_INTERVAL_SEC", "5")))

        # The wall is 1920x720 with a 3x2 grid, therefore one visible tile is
        # exactly 640x360. Keeping the mux at that exact size avoids a second
        # resize before display. Lanczos improves the one unavoidable downscale.
        self.tile_width = 640
        self.tile_height = 360
        self.wall_width = 1920
        self.wall_height = 720
        self.batch_timeout_us = max(20_000, min(80_000, int(os.environ.get("V11_BATCH_TIMEOUT_US", "40000"))))
        self.interpolation = max(0, min(6, int(os.environ.get("V11_MUX_INTERPOLATION", "4"))))

        self.lock = threading.RLock()
        self.stats = {camera.camera_id: CameraStats() for camera in self.cameras}
        self.camera_index = {camera.camera_id: index for index, camera in enumerate(self.cameras)}
        self.index_camera = {index: cid for cid, index in self.camera_index.items()}
        self.sources: dict[str, object] = {}
        self.input_queues: dict[str, object] = {}
        self.request_pads: list[tuple[object, object]] = []
        self._stopping = False

        self.mux_batches = 0
        self.mux_batches_last = 0
        self.render_frames = 0
        self.render_frames_last = 0
        self.display_stat_mono = time.monotonic()
        self.batch_sizes: deque[int] = deque(maxlen=8192)
        self.full_batches = 0
        self.batch_qmax = 0
        self.render_qmax = 0

        self._preflight()
        self.pts_bridge = NativePtsBridge()
        self.pipeline = Gst.Pipeline.new("camera-v11-step1-display-v2")
        if self.pipeline is None:
            raise RuntimeError("Could not create V11 Step1 V2 pipeline")

        self.mux = self._make("nvstreammux", "v11v2_display_mux")
        self.batch_q = self._make("queue", "v11v2_batch_latest")
        self.tiler = self._make("nvmultistreamtiler", "v11v2_tiler")
        self.render_q = self._make("queue", "v11v2_render_latest")
        self.sink = self._make("nveglglessink", "v11v2_display_sink")

        self._configure_mux()
        self._configure_latest_queue(self.batch_q)
        self._configure_latest_queue(self.render_q)

        self._set_if(self.tiler, "rows", 2)
        self._set_if(self.tiler, "columns", 3)
        self._set_if(self.tiler, "width", self.wall_width)
        self._set_if(self.tiler, "height", self.wall_height)
        self._set_if(self.tiler, "gpu-id", self.gpu_id)
        self._set_if(self.tiler, "nvbuf-memory-type", 0)
        self._set_if(self.tiler, "show-source", -1)
        # Some DeepStream builds expose these on the tiler and some do not.
        # Setting only when present keeps this runtime portable across 7.1 builds.
        self._set_if(self.tiler, "compute-hw", 1)
        self._set_if(self.tiler, "interpolation-method", self.interpolation)

        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "async", False)

        for element in (self.mux, self.batch_q, self.tiler, self.render_q, self.sink):
            self.pipeline.add(element)
        self._require_link(self.mux, self.batch_q, "mux->batch_latest")
        self._require_link(self.batch_q, self.tiler, "batch_latest->tiler")
        self._require_link(self.tiler, self.render_q, "tiler->render_latest")
        self._require_link(self.render_q, self.sink, "render_latest->egl")

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        mux_src = self.mux.get_static_pad("src")
        render_src = self.render_q.get_static_pad("src")
        if mux_src is None or render_src is None:
            raise RuntimeError("V11 Step1 V2 probe pad missing")
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._mux_probe)
        render_src.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()

        main_streams = int(all(self._looks_like_main_stream(camera.uri) for camera in self.cameras))
        print(
            "CAMERA_V11_STEP1V2_ARCH cameras=6 clean_new_file=1 step0_ingest=frozen "
            "display_only=1 tracker=0 detector=0 osd=0 jpeg=0 cpu_copy=0 "
            "input_queue=latest1 batch_queue=latest1 render_queue=latest1",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V2_POLICY "
            f"transport={self.transport} latency_ms={self.latency_ms} drop_on_latency={int(self.drop_on_latency)} "
            f"extra_surfaces={self.extra_surfaces} mux={self.tile_width}x{self.tile_height} "
            f"batch_size=6 timeout_us={self.batch_timeout_us} live_source=1 sync_inputs=0 pool=4 "
            f"tiler=3x2/{self.wall_width}x{self.wall_height} sink_sync=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V2_QUALITY "
            f"mux_interpolation={self.interpolation} gpu_scaling=1 single_resize=1 jpeg=0 "
            f"main_streams={main_streams} visible_tile={self.tile_width}x{self.tile_height}",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V2_MEASURE "
            "source_pts_to_arrival_map=1 mux_exact_frame_age=1 tiler_cache_display_age=1 "
            "post_tiler_source_meta_required=0",
            flush=True,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _looks_like_main_stream(uri: str) -> bool:
        tail = uri.rstrip("/").split("/")[-1]
        return tail.isdigit() and tail.endswith("01")

    def _preflight(self) -> None:
        required = ("nvurisrcbin", "queue", "nvstreammux", "nvmultistreamtiler", "nveglglessink", "rtspsrc")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("V11 Step1 V2 missing plugins: " + ", ".join(missing))

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
            raise RuntimeError(f"V11 Step1 V2 link failed: {label}")

    def _configure_latest_queue(self, queue) -> None:
        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

    def _configure_mux(self) -> None:
        self._set_if(self.mux, "batch-size", len(self.cameras))
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", self.tile_width)
        self._set_if(self.mux, "height", self.tile_height)
        self._set_if(self.mux, "enable-padding", False)
        self._set_if(self.mux, "batched-push-timeout", self.batch_timeout_us)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "max-latency", 0)
        self._set_if(self.mux, "buffer-pool-size", 4)
        self._set_if(self.mux, "nvbuf-memory-type", 0)
        self._set_if(self.mux, "gpu-id", self.gpu_id)
        self._set_if(self.mux, "compute-hw", 1)
        self._set_if(self.mux, "interpolation-method", self.interpolation)

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera: CameraConfig) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)
        if self.transport == "tcp":
            self._set_if(element, "protocols", 4)
        elif self.transport == "udp":
            self._set_if(element, "protocols", 1)
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", self.drop_on_latency)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "do-rtsp-keep-alive", True)

    def _request_mux_pad(self, index: int):
        name = f"sink_{index}"
        pad = self.mux.request_pad_simple(name) if hasattr(self.mux, "request_pad_simple") else None
        if pad is None:
            pad = self.mux.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"Could not request nvstreammux pad {name}")
        self.request_pads.append((self.mux, pad))
        return pad

    def _add_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        safe = cid.lower().replace("-", "_")
        source = self._make("nvurisrcbin", f"v11v2_source_{safe}")
        queue = self._make("queue", f"v11v2_input_latest_{safe}")
        self._configure_latest_queue(queue)

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.connect("pad-added", self._on_source_pad_added, cid)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "gpu-id", self.gpu_id)
        self._set_if(source, "latency", self.latency_ms)
        self._set_if(source, "drop-on-latency", self.drop_on_latency)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "select-rtp-protocol", 4 if self.transport == "tcp" else 0)
        self._set_if(source, "rtsp-reconnect-interval", self.reconnect_sec)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)

        source.set_locked_state(True)
        self.pipeline.add(source)
        self.pipeline.add(queue)

        mux_pad = self._request_mux_pad(index)
        qsrc = queue.get_static_pad("src")
        if qsrc is None or qsrc.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: input queue -> mux link failed")

        self.sources[cid] = source
        self.input_queues[cid] = queue

    def _on_source_pad_added(self, _source, pad, cid: str) -> None:
        queue = self.input_queues.get(cid)
        if queue is None:
            return
        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            with self.lock:
                self.stats[cid].errors += 1
            print(f"CAMERA_V11_STEP1V2_LINK camera={cid} status=error result={result}", flush=True)
            return
        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        with self.lock:
            self.stats[cid].source_linked = True
        caps = pad.get_current_caps()
        print(
            f"CAMERA_V11_STEP1V2_LINK camera={cid} status=OK pad={pad.get_name()} "
            f"caps={caps.to_string() if caps else 'pending'}",
            flush=True,
        )

    def _remember_pts(self, st: CameraStats, pts: int, arrival: float) -> None:
        if pts not in st.pts_arrival:
            st.pts_order.append(pts)
        st.pts_arrival[pts] = arrival
        while len(st.pts_order) > self.PTS_HISTORY:
            old = st.pts_order.popleft()
            st.pts_arrival.pop(old, None)

    def _lookup_arrival(self, st: CameraStats, pts: int) -> float | None:
        arrival = st.pts_arrival.get(pts)
        if arrival is not None:
            return arrival
        # A few elements can round/adjust PTS by a tiny amount. Search only the
        # recent tail and accept a match within 5 ms; never guess across frames.
        recent = list(st.pts_order)[-48:]
        if not recent:
            return None
        nearest = min(recent, key=lambda value: abs(value - pts))
        if abs(nearest - pts) <= self.PTS_MATCH_TOLERANCE_NS:
            return st.pts_arrival.get(nearest)
        return None

    def _source_probe(self, _pad, info, cid: str):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        pts = int(buffer.pts) if buffer.pts != self.Gst.CLOCK_TIME_NONE else None
        with self.lock:
            st = self.stats[cid]
            st.decoded += 1
            if st.last_arrival_mono is not None:
                dt = (now - st.last_arrival_mono) * 1000.0
                if 0.0 <= dt <= 5000.0:
                    st.wall_dt_ms.append(dt)
            st.last_arrival_mono = now
            if pts is not None and st.last_pts_ns is not None and pts >= st.last_pts_ns:
                dt = (pts - st.last_pts_ns) / 1_000_000.0
                if 0.0 <= dt <= 5000.0:
                    st.pts_dt_ms.append(dt)
            if pts is not None:
                st.last_pts_ns = pts
                self._remember_pts(st, pts, now)
            try:
                q = int(self.input_queues[cid].get_property("current-level-buffers"))
                st.input_qmax = max(st.input_qmax, q)
            except Exception:
                pass
        return self.Gst.PadProbeReturn.OK

    def _mux_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        rows = self.pts_bridge.copy_frame_pts(buffer, 32)
        with self.lock:
            self.mux_batches += 1
            self.batch_sizes.append(len(rows))
            if len(rows) >= len(self.cameras):
                self.full_batches += 1

            for row in rows:
                cid = self.index_camera.get(int(row["source_id"]))
                if cid is None:
                    continue
                st = self.stats[cid]
                pts = int(row["buf_pts"])
                arrival = self._lookup_arrival(st, pts)
                if arrival is None:
                    st.mux_match_miss += 1
                    continue
                age_ms = (now - arrival) * 1000.0
                if 0.0 <= age_ms <= 5000.0:
                    st.mux_age_ms.append(age_ms)
                if st.last_mux_update_mono is not None:
                    gap_ms = (now - st.last_mux_update_mono) * 1000.0
                    if 0.0 <= gap_ms <= 5000.0:
                        st.mux_gap_ms.append(gap_ms)
                st.last_mux_update_mono = now
                # This is the exact source frame that the tiler receives for the
                # camera. Until the next update, the tiler cache displays it.
                st.displayed_source_arrival_mono = arrival

            try:
                bq = int(self.batch_q.get_property("current-level-buffers"))
                self.batch_qmax = max(self.batch_qmax, bq)
            except Exception:
                pass
        return self.Gst.PadProbeReturn.OK

    def _render_probe(self, _pad, info):
        if info.get_buffer() is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.lock:
            self.render_frames += 1
            for st in self.stats.values():
                arrival = st.displayed_source_arrival_mono
                if arrival is None:
                    continue
                age_ms = (now - arrival) * 1000.0
                if 0.0 <= age_ms <= 5000.0:
                    st.display_age_ms.append(age_ms)
            try:
                rq = int(self.render_q.get_property("current-level-buffers"))
                self.render_qmax = max(self.render_qmax, rq)
            except Exception:
                pass
        return self.Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus, message) -> None:
        if message.type not in {self.Gst.MessageType.ERROR, self.Gst.MessageType.WARNING}:
            return
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            kind = "ERROR"
        else:
            err, debug = message.parse_warning()
            kind = "WARNING"
        src_name = message.src.get_name() if message.src is not None else "unknown"
        cid = next((camera_id for camera_id in self.stats if camera_id.lower().replace("-", "_") in src_name), None)
        if cid:
            with self.lock:
                if kind == "ERROR":
                    self.stats[cid].errors += 1
                else:
                    self.stats[cid].warnings += 1
        print(f"CAMERA_V11_STEP1V2_{kind} source={src_name} message={err} debug={debug}", flush=True)

    @staticmethod
    def _pct(values, fraction: float) -> float:
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
            camera_rows = []
            for camera in self.cameras:
                cid = camera.camera_id
                st = self.stats[cid]
                elapsed = max(0.001, now - st.stat_mono)
                fps = (st.decoded - st.decoded_last) / elapsed
                st.decoded_last = st.decoded
                st.stat_mono = now
                q = int(self.input_queues[cid].get_property("current-level-buffers"))
                st.input_qmax = max(st.input_qmax, q)
                camera_rows.append(
                    (
                        cid,
                        fps,
                        list(st.wall_dt_ms),
                        list(st.pts_dt_ms),
                        list(st.mux_age_ms),
                        list(st.display_age_ms),
                        list(st.mux_gap_ms),
                        len(st.mux_age_ms),
                        len(st.display_age_ms),
                        st.mux_match_miss,
                        q,
                        st.input_qmax,
                        st.errors,
                        st.warnings,
                    )
                )

            display_elapsed = max(0.001, now - self.display_stat_mono)
            batch_fps = (self.mux_batches - self.mux_batches_last) / display_elapsed
            render_fps = (self.render_frames - self.render_frames_last) / display_elapsed
            self.mux_batches_last = self.mux_batches
            self.render_frames_last = self.render_frames
            self.display_stat_mono = now
            bq = int(self.batch_q.get_property("current-level-buffers"))
            rq = int(self.render_q.get_property("current-level-buffers"))
            self.batch_qmax = max(self.batch_qmax, bq)
            self.render_qmax = max(self.render_qmax, rq)
            sizes = list(self.batch_sizes)
            full_pct = 100.0 * self.full_batches / max(1, self.mux_batches)

        for row in camera_rows:
            (
                cid,
                fps,
                wall,
                pts,
                mux_age,
                display_age,
                mux_gap,
                mux_samples,
                display_samples,
                misses,
                q,
                qmax,
                errors,
                warnings,
            ) = row
            print(
                "CAMERA_V11_STEP1V2_CAMERA "
                f"camera={cid} source_fps={fps:.2f} wall_p95={self._pct(wall, 0.95):.0f}ms "
                f"pts_p95={self._pct(pts, 0.95):.0f}ms mux_age_p95={self._pct(mux_age, 0.95):.0f}ms "
                f"display_age_p95={self._pct(display_age, 0.95):.0f}ms "
                f"mux_gap_p95={self._pct(mux_gap, 0.95):.0f}ms "
                f"mux_samples={mux_samples} display_samples={display_samples} match_miss={misses} "
                f"input_q={q} input_qmax={qmax} errors={errors} warnings={warnings}",
                flush=True,
            )

        print(
            "CAMERA_V11_STEP1V2_DISPLAY "
            f"batch_fps={batch_fps:.2f} render_fps={render_fps:.2f} "
            f"batch_size_p50={self._pct(sizes, 0.50):.0f} batch_size_p95={self._pct(sizes, 0.95):.0f} "
            f"full_pct={full_pct:.1f} batch_q={bq} batch_qmax={self.batch_qmax} "
            f"render_q={rq} render_qmax={self.render_qmax} batches={self.mux_batches}",
            flush=True,
        )
        return True

    def _start_sources(self) -> None:
        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            delay_ms = max(1, int(round(index * self.startup_stagger * 1000.0)))

            def _start(camera_id=cid):
                if self._stopping:
                    return False
                source = self.sources[camera_id]
                source.set_locked_state(False)
                source.sync_state_with_parent()
                print(f"CAMERA_V11_STEP1V2_START camera={camera_id}", flush=True)
                return False

            self.GLib.timeout_add(delay_ms, _start)

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
        state_name = result.value_nick if hasattr(result, "value_nick") else result
        print(f"CAMERA_V11_STEP1V2_PIPELINE state={state_name}", flush=True)
        self._start_sources()
        self.GLib.timeout_add_seconds(self.stats_interval, self._print_stats)
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(self.Gst.State.NULL)
            for mux, pad in self.request_pads:
                try:
                    mux.release_request_pad(pad)
                except Exception:
                    pass
        return 0


def main() -> int:
    return V11Step1DisplayV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
