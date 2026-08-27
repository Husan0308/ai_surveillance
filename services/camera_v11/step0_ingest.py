from __future__ import annotations

import os
import signal
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from services.ml_service.app.config import CameraConfig, load_settings


@dataclass
class CameraStats:
    decoded: int = 0
    delivered: int = 0
    decoded_last: int = 0
    delivered_last: int = 0
    stat_mono: float = field(default_factory=time.monotonic)
    last_arrival_mono: float | None = None
    last_pts_ns: int | None = None
    wall_dt_ms: deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    pts_dt_ms: deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    max_queue: int = 0
    errors: int = 0
    warnings: int = 0
    caps_logged: bool = False
    source_linked: bool = False


class V11Step0Ingest:
    """Clean RTSP/NVDEC baseline: one independent pipeline per camera.

    There is deliberately no mux, tracker, detector, tiler, OSD or display.
    A bad camera must not create backpressure in any other camera. Each source
    decodes into a one-buffer downstream-leaky queue and a non-synchronizing
    fakesink. Step 0 measures only source/decode health and RTP jitter.

    Important: nvurisrcbin is decodebin-backed and its output pad is created
    dynamically. We therefore link its pad when it appears instead of calling
    Gst.Element.link() during graph construction.
    """

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self.GLib = GLib
        self.Gst = Gst
        self.settings = load_settings()
        self.cameras = list(self.settings.cameras)
        if not self.cameras:
            raise RuntimeError("V11 Step0 requires at least one camera")

        ds = self.settings.deepstream
        self.gpu_id = int(os.environ.get("V11_GPU_ID", ds.gpu_id))
        self.transport = os.environ.get("V11_RTSP_TRANSPORT", "tcp").strip().lower()
        if self.transport not in {"tcp", "udp", "auto"}:
            raise RuntimeError("V11_RTSP_TRANSPORT must be tcp, udp, or auto")

        # Keep the proven V10.9 TCP/60 ms baseline. No per-camera tuning in Step0.
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

        self.lock = threading.RLock()
        self.stats = {camera.camera_id: CameraStats() for camera in self.cameras}
        self.pipelines: dict[str, object] = {}
        self.sources: dict[str, object] = {}
        self.queues: dict[str, object] = {}
        self.sinks: dict[str, object] = {}
        self.jitterbuffers: dict[str, list[object]] = defaultdict(list)
        self._stopping = False
        self.loop = GLib.MainLoop()

        self._preflight()
        for camera in self.cameras:
            self._build_camera(camera)

        print(
            "CAMERA_V11_STEP0_ARCH "
            f"cameras={len(self.cameras)} independent_pipelines=1 decode_only=1 "
            "mux=0 tracker=0 detector=0 display=0 queue=latest1/leaky-downstream "
            "nvurisrcbin_link=dynamic-pad",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP0_POLICY "
            f"transport={self.transport} latency_ms={self.latency_ms} "
            f"drop_on_latency={int(self.drop_on_latency)} extra_surfaces={self.extra_surfaces} "
            f"cudadec_memtype=0 reconnect_sec={self.reconnect_sec} "
            "tcp_timestamp=default-not-forced",
            flush=True,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _preflight(self) -> None:
        required = ("nvurisrcbin", "queue", "fakesink", "rtspsrc")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("V11 Step0 missing GStreamer/DeepStream plugins: " + ", ".join(missing))

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

        tcp_timestamp_available = int(element.find_property("tcp-timestamp") is not None)
        print(
            "CAMERA_V11_STEP0_RTSP "
            f"camera={camera.camera_id} latency_ms={self.latency_ms} "
            f"drop_on_latency={int(self.drop_on_latency)} transport={self.transport} "
            f"tcp_timestamp_property={tcp_timestamp_available} tcp_timestamp_forced=0",
            flush=True,
        )
        try:
            element.connect("new-manager", self._on_new_manager, camera.camera_id)
        except Exception as exc:
            print(
                f"CAMERA_V11_STEP0_RTP_HOOK camera={camera.camera_id} manager_hook=0 "
                f"error={type(exc).__name__}",
                flush=True,
            )

    def _on_new_manager(self, _rtspsrc, manager, cid: str) -> None:
        try:
            manager.connect("new-jitterbuffer", self._on_new_jitterbuffer, cid)
            hooked = 1
        except Exception:
            hooked = 0
        print(
            f"CAMERA_V11_STEP0_MANAGER camera={cid} jitterbuffer_hook={hooked}",
            flush=True,
        )

    def _on_new_jitterbuffer(self, _manager, jitterbuffer, session: int, ssrc: int, cid: str) -> None:
        with self.lock:
            rows = self.jitterbuffers[cid]
            if jitterbuffer not in rows:
                rows.append(jitterbuffer)
            index = rows.index(jitterbuffer)
        print(
            f"CAMERA_V11_STEP0_JB camera={cid} index={index} session={int(session)} ssrc={int(ssrc)}",
            flush=True,
        )

    def _link_source_pad(self, pad, cid: str) -> bool:
        queue = self.queues.get(cid)
        if queue is None:
            return False
        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None:
            with self.lock:
                self.stats[cid].errors += 1
            print(f"CAMERA_V11_STEP0_LINK camera={cid} status=error reason=queue-sink-pad-missing", flush=True)
            return False

        if sink_pad.is_linked():
            return True

        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            with self.lock:
                self.stats[cid].errors += 1
            result_name = result.value_nick if hasattr(result, "value_nick") else str(result)
            print(
                f"CAMERA_V11_STEP0_LINK camera={cid} status=error result={result_name} pad={pad.get_name()}",
                flush=True,
            )
            return False

        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._decode_probe, cid)
        with self.lock:
            self.stats[cid].source_linked = True
        caps = pad.get_current_caps()
        print(
            "CAMERA_V11_STEP0_LINK "
            f"camera={cid} status=OK mode=dynamic-pad pad={pad.get_name()} "
            f"caps={caps.to_string() if caps is not None else 'pending'}",
            flush=True,
        )
        return True

    def _on_source_pad_added(self, _source, pad, cid: str) -> None:
        self._link_source_pad(pad, cid)

    def _build_camera(self, camera: CameraConfig) -> None:
        cid = camera.camera_id
        safe = cid.lower().replace("-", "_")
        pipeline = self.Gst.Pipeline.new(f"v11_step0_{safe}")
        source = self._make("nvurisrcbin", f"source_{safe}")
        queue = self._make("queue", f"queue_{safe}")
        sink = self._make("fakesink", f"sink_{safe}")

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
        # Keep decoder low-latency-mode at default. We have not proven that all
        # NVR channels are IPPP-only; B-frame streams can require reorder delay.

        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

        self._set_if(sink, "sync", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "enable-last-sample", False)

        for element in (source, queue, sink):
            pipeline.add(element)

        # queue/fakesink have static pads and can be linked immediately. The
        # nvurisrcbin output pad is decodebin-backed and is linked in pad-added.
        if not queue.link(sink):
            raise RuntimeError(f"{cid}: queue -> fakesink link failed")

        qsrc = queue.get_static_pad("src")
        if qsrc is None:
            raise RuntimeError(f"{cid}: queue src pad missing")
        qsrc.add_probe(self.Gst.PadProbeType.BUFFER, self._delivery_probe, cid)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, cid)

        self.pipelines[cid] = pipeline
        self.sources[cid] = source
        self.queues[cid] = queue
        self.sinks[cid] = sink

        # Some builds may expose the src pad immediately. Handle that too,
        # while keeping pad-added as the canonical path for delayed pads.
        src_pad = source.get_static_pad("src")
        if src_pad is not None:
            self._link_source_pad(src_pad, cid)

    def _decode_probe(self, pad, info, cid: str):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        pts = int(buffer.pts) if buffer.pts != self.Gst.CLOCK_TIME_NONE else None
        with self.lock:
            st = self.stats[cid]
            st.decoded += 1
            if st.last_arrival_mono is not None:
                dt_ms = (now - st.last_arrival_mono) * 1000.0
                if 0.0 <= dt_ms <= 5000.0:
                    st.wall_dt_ms.append(dt_ms)
            st.last_arrival_mono = now
            if pts is not None and st.last_pts_ns is not None and pts >= st.last_pts_ns:
                dt_ms = (pts - st.last_pts_ns) / 1_000_000.0
                if 0.0 <= dt_ms <= 5000.0:
                    st.pts_dt_ms.append(dt_ms)
            if pts is not None:
                st.last_pts_ns = pts
            if not st.caps_logged:
                caps = pad.get_current_caps()
                print(
                    f"CAMERA_V11_STEP0_CAPS camera={cid} "
                    f"caps={caps.to_string() if caps is not None else 'unknown'}",
                    flush=True,
                )
                st.caps_logged = True
        return self.Gst.PadProbeReturn.OK

    def _delivery_probe(self, _pad, _info, cid: str):
        with self.lock:
            self.stats[cid].delivered += 1
        return self.Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus, message, cid: str) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            with self.lock:
                self.stats[cid].errors += 1
            print(
                f"CAMERA_V11_STEP0_ERROR camera={cid} error={err} debug={debug}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            with self.lock:
                self.stats[cid].warnings += 1
            print(
                f"CAMERA_V11_STEP0_WARNING camera={cid} warning={err} debug={debug}",
                flush=True,
            )

    @staticmethod
    def _pct(values: list[float] | deque[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return float(ordered[index])

    @staticmethod
    def _stat_int(stats, key: str) -> int:
        try:
            value = stats.get_value(key)
            return int(value) if value is not None else 0
        except Exception:
            return 0

    def _primary_rtp(self, cid: str) -> tuple[int, int, int, int, float]:
        best = (0, 0, 0, 0, 0.0)
        best_pushed = -1
        with self.lock:
            rows = list(self.jitterbuffers.get(cid, ()))
        for jb in rows:
            try:
                stats = jb.get_property("stats")
                pushed = self._stat_int(stats, "num-pushed")
                lost = self._stat_int(stats, "num-lost")
                late = self._stat_int(stats, "num-late")
                dup = self._stat_int(stats, "num-duplicates")
                jitter_ms = self._stat_int(stats, "avg-jitter") / 1_000_000.0
                if pushed > best_pushed:
                    best = (pushed, lost, late, dup, jitter_ms)
                    best_pushed = pushed
            except Exception:
                continue
        return best

    def _print_stats(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        with self.lock:
            snapshot = []
            for camera in self.cameras:
                cid = camera.camera_id
                st = self.stats[cid]
                elapsed = max(0.001, now - st.stat_mono)
                decode_fps = (st.decoded - st.decoded_last) / elapsed
                sink_fps = (st.delivered - st.delivered_last) / elapsed
                st.decoded_last = st.decoded
                st.delivered_last = st.delivered
                st.stat_mono = now
                qlevel = int(self.queues[cid].get_property("current-level-buffers"))
                st.max_queue = max(st.max_queue, qlevel)
                snapshot.append(
                    (
                        cid,
                        decode_fps,
                        sink_fps,
                        list(st.wall_dt_ms),
                        list(st.pts_dt_ms),
                        qlevel,
                        st.max_queue,
                        st.errors,
                        st.warnings,
                        st.decoded,
                        st.source_linked,
                    )
                )

        for cid, decode_fps, sink_fps, wall, pts, qlevel, qmax, errors, warnings, decoded, linked in snapshot:
            pushed, lost, late, dup, jitter_ms = self._primary_rtp(cid)
            # Keep the checker-compatible STATS line stable; LINK is emitted as
            # its own event and link failures increment errors.
            print(
                "CAMERA_V11_STEP0_STATS "
                f"camera={cid} decoded_total={decoded} decode_fps={decode_fps:.2f} sink_fps={sink_fps:.2f} "
                f"wall_p50={self._pct(wall, 0.50):.0f}ms wall_p95={self._pct(wall, 0.95):.0f}ms "
                f"wall_p99={self._pct(wall, 0.99):.0f}ms pts_p50={self._pct(pts, 0.50):.0f}ms "
                f"pts_p95={self._pct(pts, 0.95):.0f}ms q={qlevel} qmax={qmax} "
                f"rtp_pushed={pushed} rtp_lost={lost} rtp_late={late} rtp_dup={dup} "
                f"rtp_jitter_ms={jitter_ms:.3f} errors={errors} warnings={warnings}",
                flush=True,
            )
            if not linked:
                print(f"CAMERA_V11_STEP0_LINK_WAIT camera={cid} linked=0", flush=True)
        return True

    def _start_all(self) -> None:
        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            delay_ms = max(1, int(round(index * self.startup_stagger * 1000.0)))

            def _start(camera_id=cid):
                if self._stopping:
                    return False
                result = self.pipelines[camera_id].set_state(self.Gst.State.PLAYING)
                print(
                    f"CAMERA_V11_STEP0_START camera={camera_id} "
                    f"state={result.value_nick if hasattr(result, 'value_nick') else result}",
                    flush=True,
                )
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
        self._start_all()
        self.GLib.timeout_add_seconds(self.stats_interval, self._print_stats)
        try:
            self.loop.run()
        finally:
            for pipeline in self.pipelines.values():
                pipeline.set_state(self.Gst.State.NULL)
        return 0


def main() -> int:
    return V11Step0Ingest().run()


if __name__ == "__main__":
    raise SystemExit(main())
