from __future__ import annotations

import ctypes
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from services.ml_service.app.config import CameraConfig, load_settings


@dataclass
class CameraStats:
    decoded: int = 0
    rendered: int = 0
    decoded_last: int = 0
    rendered_last: int = 0
    stat_mono: float = field(default_factory=time.monotonic)
    last_arrival_mono: float | None = None
    last_pts_ns: int | None = None
    last_render_mono: float | None = None
    wall_dt_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    pts_dt_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    render_gap_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    display_age_ms: deque[float] = field(default_factory=lambda: deque(maxlen=8192))
    pts_history: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=1024))
    display_samples: int = 0
    pts_match_miss: int = 0
    input_qmax: int = 0
    errors: int = 0
    warnings: int = 0
    source_linked: bool = False


class X11Wall:
    """Minimal X11 parent + six child windows, using libX11 directly."""

    def __init__(self, width: int, height: int, tile_w: int, tile_h: int, count: int) -> None:
        self.lib = ctypes.CDLL("libX11.so.6")
        self.lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.lib.XOpenDisplay.restype = ctypes.c_void_p
        self.lib.XDefaultScreen.argtypes = [ctypes.c_void_p]
        self.lib.XDefaultScreen.restype = ctypes.c_int
        self.lib.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.XRootWindow.restype = ctypes.c_ulong
        self.lib.XBlackPixel.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.XBlackPixel.restype = ctypes.c_ulong
        self.lib.XCreateSimpleWindow.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        self.lib.XCreateSimpleWindow.restype = ctypes.c_ulong
        self.lib.XStoreName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_char_p]
        self.lib.XMapWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.lib.XMapRaised.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.lib.XFlush.argtypes = [ctypes.c_void_p]
        self.lib.XDestroyWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.lib.XCloseDisplay.argtypes = [ctypes.c_void_p]

        display_name = os.environ.get("DISPLAY", "").encode() or None
        self.display = self.lib.XOpenDisplay(display_name)
        if not self.display:
            raise RuntimeError(f"V11 Step1 V4 could not open X11 DISPLAY={os.environ.get('DISPLAY')!r}")

        screen = self.lib.XDefaultScreen(self.display)
        root = self.lib.XRootWindow(self.display, screen)
        black = self.lib.XBlackPixel(self.display, screen)

        self.parent = self.lib.XCreateSimpleWindow(
            self.display, root, 0, 0, width, height, 0, black, black
        )
        if not self.parent:
            raise RuntimeError("V11 Step1 V4 could not create X11 wall window")
        self.lib.XStoreName(self.display, self.parent, b"Camera V11 Step1 V4 - Independent EGL")
        self.lib.XMapRaised(self.display, self.parent)

        self.children: list[int] = []
        for index in range(count):
            col = index % 3
            row = index // 3
            child = self.lib.XCreateSimpleWindow(
                self.display,
                self.parent,
                col * tile_w,
                row * tile_h,
                tile_w,
                tile_h,
                0,
                black,
                black,
            )
            if not child:
                raise RuntimeError(f"V11 Step1 V4 could not create X11 child {index}")
            self.lib.XMapWindow(self.display, child)
            self.children.append(int(child))

        self.lib.XFlush(self.display)

    def close(self) -> None:
        if not getattr(self, "display", None):
            return
        for child in reversed(getattr(self, "children", [])):
            try:
                self.lib.XDestroyWindow(self.display, ctypes.c_ulong(child))
            except Exception:
                pass
        try:
            self.lib.XDestroyWindow(self.display, ctypes.c_ulong(self.parent))
        except Exception:
            pass
        try:
            self.lib.XFlush(self.display)
            self.lib.XCloseDisplay(self.display)
        except Exception:
            pass
        self.display = None


class V11Step1IndependentEglV4:
    """V11 Step1 V4: six independent GPU display pipelines.

    There is deliberately no nvstreammux and no nvmultistreamtiler. Each camera
    owns its own pipeline, one-buffer downstream-leaky queue, GPU resize and
    nveglglessink bound directly to its own X11 child window via GstVideoOverlay.

    This removes the cross-camera batching dependency proven problematic by V2/V3.
    Detector/tracker/OSD/ReID/face/JPEG/OpenCV remain absent.
    """

    PTS_MATCH_TOLERANCE_NS = 5_000_000

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import GLib, Gst, GstVideo

        Gst.init(None)
        self.GLib = GLib
        self.Gst = Gst
        self.GstVideo = GstVideo

        self.settings = load_settings()
        self.cameras = list(self.settings.cameras)
        if len(self.cameras) != 6:
            raise RuntimeError(f"V11 Step1 V4 requires exactly 6 cameras, got {len(self.cameras)}")

        ds = self.settings.deepstream
        self.gpu_id = int(os.environ.get("V11_GPU_ID", ds.gpu_id))
        self.transport = os.environ.get("V11_RTSP_TRANSPORT", "tcp").strip().lower()
        if self.transport not in {"tcp", "udp", "auto"}:
            raise RuntimeError("V11_RTSP_TRANSPORT must be tcp, udp, or auto")
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

        self.tile_width = max(320, int(os.environ.get("V11_TILE_WIDTH", "640")))
        self.tile_height = max(180, int(os.environ.get("V11_TILE_HEIGHT", "360")))
        self.wall_width = self.tile_width * 3
        self.wall_height = self.tile_height * 2
        self.interpolation = max(0, min(6, int(os.environ.get("V11_SCALE_INTERPOLATION", "4"))))

        self.lock = threading.RLock()
        self.stats = {camera.camera_id: CameraStats() for camera in self.cameras}
        self.pipelines: dict[str, object] = {}
        self.sources: dict[str, object] = {}
        self.queues: dict[str, object] = {}
        self.converters: dict[str, object] = {}
        self.capsfilters: dict[str, object] = {}
        self.sinks: dict[str, object] = {}
        self._stopping = False

        self._preflight()
        self.wall = X11Wall(
            self.wall_width,
            self.wall_height,
            self.tile_width,
            self.tile_height,
            len(self.cameras),
        )
        self.loop = GLib.MainLoop()

        for index, camera in enumerate(self.cameras):
            self._build_camera(index, camera)

        print(
            "CAMERA_V11_STEP1V4_ARCH "
            "cameras=6 independent_pipelines=6 mux=0 tiler=0 detector=0 tracker=0 "
            "osd=0 reid=0 face=0 jpeg=0 opencv=0 queue=latest1/leaky-downstream "
            "sink=nveglglessink videooverlay=1",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V4_POLICY "
            f"transport={self.transport} latency_ms={self.latency_ms} "
            f"drop_on_latency={int(self.drop_on_latency)} extra_surfaces={self.extra_surfaces} "
            f"tile={self.tile_width}x{self.tile_height} wall={self.wall_width}x{self.wall_height} "
            f"interpolation={self.interpolation} sink_sync=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V4_QUALITY "
            f"interpolation={self.interpolation} gpu_scaling=1 single_resize=1 jpeg=0 "
            f"main_streams={int(all(self._is_main_stream(c.uri) for c in self.cameras))} "
            f"tile={self.tile_width}x{self.tile_height} mux=0 tiler=0 independent=1 videooverlay=1",
            flush=True,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _is_main_stream(uri: str) -> bool:
        tail = uri.rstrip("/").split("/")[-1]
        return len(tail) >= 3 and tail.endswith("01")

    def _preflight(self) -> None:
        required = ("nvurisrcbin", "queue", "nvvideoconvert", "capsfilter", "nveglglessink", "rtspsrc")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("V11 Step1 V4 missing GStreamer/DeepStream plugins: " + ", ".join(missing))
        try:
            ctypes.CDLL("libX11.so.6")
        except OSError as exc:
            raise RuntimeError(f"V11 Step1 V4 requires libX11.so.6: {exc}") from exc

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
            raise RuntimeError(f"V11 Step1 V4 link failed: {label}")

    def _configure_latest_queue(self, queue) -> None:
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
        if self.transport == "tcp":
            self._set_if(element, "protocols", 4)
        elif self.transport == "udp":
            self._set_if(element, "protocols", 1)
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", self.drop_on_latency)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "do-rtsp-keep-alive", True)

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        safe = cid.lower().replace("-", "_")
        pipeline = self.Gst.Pipeline.new(f"v11_step1v4_{safe}")
        if pipeline is None:
            raise RuntimeError(f"{cid}: could not create pipeline")

        source = self._make("nvurisrcbin", f"source_{safe}")
        queue = self._make("queue", f"latest_{safe}")
        convert = self._make("nvvideoconvert", f"scale_{safe}")
        capsfilter = self._make("capsfilter", f"caps_{safe}")
        sink = self._make("nveglglessink", f"sink_{safe}")

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

        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "nvbuf-memory-type", 0)
        self._set_if(convert, "interpolation-method", self.interpolation)

        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.tile_width},height={self.tile_height}"
            ),
        )

        self._set_if(sink, "sync", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "enable-last-sample", False)
        self._set_if(sink, "force-aspect-ratio", False)

        for element in (source, queue, convert, capsfilter, sink):
            pipeline.add(element)
        self._require_link(queue, convert, f"{cid}:queue->convert")
        self._require_link(convert, capsfilter, f"{cid}:convert->caps")
        self._require_link(capsfilter, sink, f"{cid}:caps->sink")

        sink_pad = sink.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError(f"{cid}: sink pad missing")
        sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe, cid)

        try:
            self.GstVideo.VideoOverlay.set_window_handle(sink, int(self.wall.children[index]))
            overlay_ok = 1
        except Exception as exc:
            overlay_ok = 0
            raise RuntimeError(f"{cid}: GstVideoOverlay window binding failed: {exc}") from exc

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, cid)

        self.pipelines[cid] = pipeline
        self.sources[cid] = source
        self.queues[cid] = queue
        self.converters[cid] = convert
        self.capsfilters[cid] = capsfilter
        self.sinks[cid] = sink

        print(
            f"CAMERA_V11_STEP1V4_WINDOW camera={cid} xid={self.wall.children[index]} "
            f"overlay={overlay_ok} tile={self.tile_width}x{self.tile_height}",
            flush=True,
        )

    def _on_source_pad_added(self, _source, pad, cid: str) -> None:
        queue = self.queues.get(cid)
        if queue is None:
            return
        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            with self.lock:
                self.stats[cid].errors += 1
            result_name = result.value_nick if hasattr(result, "value_nick") else str(result)
            print(f"CAMERA_V11_STEP1V4_LINK camera={cid} status=error result={result_name}", flush=True)
            return
        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        with self.lock:
            self.stats[cid].source_linked = True
        caps = pad.get_current_caps()
        print(
            f"CAMERA_V11_STEP1V4_LINK camera={cid} status=OK pad={pad.get_name()} "
            f"caps={caps.to_string() if caps else 'pending'}",
            flush=True,
        )

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

            if pts is not None:
                if st.last_pts_ns is not None and pts >= st.last_pts_ns:
                    dt = (pts - st.last_pts_ns) / 1_000_000.0
                    if 0.0 <= dt <= 5000.0:
                        st.pts_dt_ms.append(dt)
                st.last_pts_ns = pts
                st.pts_history.append((pts, now))
        return self.Gst.PadProbeReturn.OK

    def _match_pts_arrival(self, st: CameraStats, pts: int) -> float | None:
        best: tuple[int, float] | None = None
        best_delta = self.PTS_MATCH_TOLERANCE_NS + 1
        for src_pts, arrival in reversed(st.pts_history):
            delta = abs(src_pts - pts)
            if delta < best_delta:
                best_delta = delta
                best = (src_pts, arrival)
            if src_pts < pts - self.PTS_MATCH_TOLERANCE_NS:
                break
        if best is None or best_delta > self.PTS_MATCH_TOLERANCE_NS:
            return None
        return best[1]

    def _render_probe(self, _pad, info, cid: str):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        pts = int(buffer.pts) if buffer.pts != self.Gst.CLOCK_TIME_NONE else None
        with self.lock:
            st = self.stats[cid]
            st.rendered += 1
            if st.last_render_mono is not None:
                gap = (now - st.last_render_mono) * 1000.0
                if 0.0 <= gap <= 5000.0:
                    st.render_gap_ms.append(gap)
            st.last_render_mono = now

            if pts is not None:
                arrival = self._match_pts_arrival(st, pts)
                if arrival is None:
                    st.pts_match_miss += 1
                else:
                    age = (now - arrival) * 1000.0
                    if 0.0 <= age <= 5000.0:
                        st.display_age_ms.append(age)
                        st.display_samples += 1
        return self.Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus, message, cid: str) -> None:
        if message.type not in {self.Gst.MessageType.ERROR, self.Gst.MessageType.WARNING}:
            return
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            kind = "ERROR"
        else:
            err, debug = message.parse_warning()
            kind = "WARNING"
        with self.lock:
            if kind == "ERROR":
                self.stats[cid].errors += 1
            else:
                self.stats[cid].warnings += 1
        src_name = message.src.get_name() if message.src is not None else "unknown"
        print(
            f"CAMERA_V11_STEP1V4_{kind} camera={cid} source={src_name} message={err} debug={debug}",
            flush=True,
        )

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
        rows = []
        with self.lock:
            for camera in self.cameras:
                cid = camera.camera_id
                st = self.stats[cid]
                elapsed = max(0.001, now - st.stat_mono)
                source_fps = (st.decoded - st.decoded_last) / elapsed
                render_fps = (st.rendered - st.rendered_last) / elapsed
                st.decoded_last = st.decoded
                st.rendered_last = st.rendered
                st.stat_mono = now
                q = int(self.queues[cid].get_property("current-level-buffers"))
                st.input_qmax = max(st.input_qmax, q)
                rows.append(
                    (
                        cid,
                        source_fps,
                        render_fps,
                        list(st.wall_dt_ms),
                        list(st.pts_dt_ms),
                        list(st.display_age_ms),
                        list(st.render_gap_ms),
                        st.display_samples,
                        st.pts_match_miss,
                        q,
                        st.input_qmax,
                        st.errors,
                        st.warnings,
                    )
                )

        for (
            cid,
            source_fps,
            render_fps,
            wall,
            pts,
            display_age,
            render_gap,
            samples,
            misses,
            q,
            qmax,
            errors,
            warnings,
        ) in rows:
            print(
                "CAMERA_V11_STEP1V4_CAMERA "
                f"camera={cid} source_fps={source_fps:.2f} render_fps={render_fps:.2f} "
                f"wall_p95={self._pct(wall, 0.95):.0f}ms pts_p95={self._pct(pts, 0.95):.0f}ms "
                f"display_age_p95={self._pct(display_age, 0.95):.0f}ms "
                f"render_gap_p95={self._pct(render_gap, 0.95):.0f}ms "
                f"display_samples={samples} pts_match_miss={misses} "
                f"input_q={q} input_qmax={qmax} errors={errors} warnings={warnings}",
                flush=True,
            )
        return True

    def _start_pipelines(self) -> None:
        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            delay_ms = max(1, int(round(index * self.startup_stagger * 1000.0)))

            def _start(camera_id=cid):
                if self._stopping:
                    return False
                result = self.pipelines[camera_id].set_state(self.Gst.State.PLAYING)
                name = result.value_nick if hasattr(result, "value_nick") else str(result)
                print(f"CAMERA_V11_STEP1V4_START camera={camera_id} state={name}", flush=True)
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
        self._start_pipelines()
        self.GLib.timeout_add_seconds(self.stats_interval, self._print_stats)
        try:
            self.loop.run()
        finally:
            for pipeline in self.pipelines.values():
                try:
                    pipeline.set_state(self.Gst.State.NULL)
                except Exception:
                    pass
            self.wall.close()
        return 0


def main() -> int:
    return V11Step1IndependentEglV4().run()


if __name__ == "__main__":
    raise SystemExit(main())
