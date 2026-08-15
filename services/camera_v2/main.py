from __future__ import annotations

import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit, urlunsplit

# DeepStream 7.1: use the legacy nvstreammux in this camera-only baseline.
# It gives us deterministic width/height, live-source, GPU memory and pool knobs.
os.environ.pop("USE_NEW_NVSTREAMMUX", None)

from services.ml_service.app.config import CameraConfig, load_settings


@dataclass
class SourceRuntime:
    frames: int = 0
    last_frames: int = 0
    last_stat_time: float = field(default_factory=time.monotonic)
    last_pts_ns: int | None = None
    intervals_ms: deque[float] = field(default_factory=lambda: deque(maxlen=240))
    caps: str = "pending"


class CameraWallV2:
    """Clean six-camera GPU-only baseline.

    RTSP -> nvurisrcbin/NVDEC -> latest queue(1) -> nvstreammux(batch=6)
         -> nvmultistreamtiler -> latest wall queue(1) -> nveglglessink

    No UI framework, appsink, NumPy, JPEG, mmap, detector, tracker or OSD.
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
        if len(self.cameras) != 6:
            raise RuntimeError(f"Camera V2 requires exactly 6 cameras, found {len(self.cameras)}")

        ds = self.settings.deepstream
        self.gpu_id = int(os.environ.get("CAMERA_V2_GPU_ID", ds.gpu_id))
        self.rtsp_latency_ms = max(40, int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "100")))
        self.udp_buffer_size = max(
            1_048_576,
            int(os.environ.get("CAMERA_V2_UDP_BUFFER_SIZE", str(8 * 1024 * 1024))),
        )
        self.extra_surfaces = max(2, min(16, int(os.environ.get("CAMERA_V2_EXTRA_SURFACES", "6"))))
        self.low_latency_mode = os.environ.get("CAMERA_V2_LOW_LATENCY_MODE", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

        # The NVR is currently delivering ~20 FPS on all six channels. Matching
        # the mux timeout to one frame period avoids needless partial-batch bursts.
        self.source_fps = max(1, int(os.environ.get("CAMERA_V2_SOURCE_FPS", "20")))
        self.mux_timeout_us = max(
            5_000,
            int(os.environ.get("CAMERA_V2_MUX_TIMEOUT_US", str(round(1_000_000 / self.source_fps)))),
        )

        # Keep decode/mux surfaces at the camera's native working resolution.
        self.frame_width = max(320, int(os.environ.get("CAMERA_V2_FRAME_WIDTH", "1280")))
        self.frame_height = max(180, int(os.environ.get("CAMERA_V2_FRAME_HEIGHT", "720")))

        # Smoothness-first wall. 1920x720 is 4x fewer output pixels than the old
        # 3840x1440 experiment while keeping a clean 3x2 16:9 layout.
        self.wall_width = max(960, int(os.environ.get("CAMERA_V2_WALL_WIDTH", "1920")))
        self.wall_height = max(360, int(os.environ.get("CAMERA_V2_WALL_HEIGHT", "720")))

        self.stats = {cam.camera_id: SourceRuntime() for cam in self.cameras}
        self.queues: dict[str, object] = {}
        self.sources: dict[str, object] = {}
        self._request_pads: list[object] = []
        self._warning_last: dict[str, float] = {}
        self._stopping = False

        self._preflight()
        self.pipeline = Gst.Pipeline.new("camera-v2-gpu-wall")
        if self.pipeline is None:
            raise RuntimeError("Could not create GStreamer pipeline")

        self.mux = self._make("nvstreammux", "camera_v2_mux")
        self.tiler = self._make("nvmultistreamtiler", "camera_v2_tiler")
        self.wall_queue = self._make("queue", "camera_v2_wall_queue")
        self.sink = self._make("nveglglessink", "camera_v2_sink")

        self._configure_mux()
        self._configure_tiler()
        self._configure_wall_queue()
        self._configure_sink()

        for element in (self.mux, self.tiler, self.wall_queue, self.sink):
            self.pipeline.add(element)

        self._require_link(self.mux, self.tiler, "nvstreammux -> nvmultistreamtiler")
        self._require_link(self.tiler, self.wall_queue, "nvmultistreamtiler -> wall queue")
        self._require_link(self.wall_queue, self.sink, "wall queue -> nveglglessink")

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()
        GLib.timeout_add_seconds(5, self._print_stats)

    def _preflight(self) -> None:
        required = ("nvurisrcbin", "nvstreammux", "nvmultistreamtiler", "nveglglessink", "queue")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("Missing DeepStream/GStreamer plugins: " + ", ".join(missing))

        try:
            with open("/proc/sys/net/core/rmem_max", "r", encoding="utf-8") as f:
                rmem_max = int(f.read().strip())
            if rmem_max < self.udp_buffer_size:
                print(
                    f"CAMERA_V2 WARNING kernel rmem_max={rmem_max} < requested UDP buffer "
                    f"{self.udp_buffer_size}. For UDP-heavy RTSP, raise net.core.rmem_max.",
                    flush=True,
                )
        except Exception:
            pass

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Could not create {factory}")
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

    def _configure_mux(self) -> None:
        self._set_if(self.mux, "batch-size", 6)
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", self.frame_width)
        self._set_if(self.mux, "height", self.frame_height)
        self._set_if(self.mux, "enable-padding", False)
        self._set_if(self.mux, "batched-push-timeout", self.mux_timeout_us)

        # Do not let one camera pause the wall. The tiler keeps a cached previous
        # frame for a slower source, so independent live motion remains responsive.
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "max-latency", 0)
        self._set_if(self.mux, "buffer-pool-size", 8)
        self._set_if(self.mux, "nvbuf-memory-type", 2)  # CUDA device memory
        self._set_if(self.mux, "gpu-id", self.gpu_id)

    def _configure_tiler(self) -> None:
        self._set_if(self.tiler, "rows", 2)
        self._set_if(self.tiler, "columns", 3)
        self._set_if(self.tiler, "width", self.wall_width)
        self._set_if(self.tiler, "height", self.wall_height)
        self._set_if(self.tiler, "gpu-id", self.gpu_id)
        self._set_if(self.tiler, "nvbuf-memory-type", 2)

    def _configure_wall_queue(self) -> None:
        self._set_if(self.wall_queue, "max-size-buffers", 1)
        self._set_if(self.wall_queue, "max-size-bytes", 0)
        self._set_if(self.wall_queue, "max-size-time", 0)
        self._set_if(self.wall_queue, "leaky", 2)  # downstream: keep newest wall only
        self._set_if(self.wall_queue, "silent", True)

    def _configure_sink(self) -> None:
        # NVIDIA's DeepStream FAQ recommends sync=0/qos=0 for jittery live camera
        # display. It prevents GstBaseSink from declaring frames late and dropping
        # them as happened in the previous experimental branch.
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

    def _auth_uri(self, camera: CameraConfig) -> str:
        if not camera.username or not camera.password:
            return camera.uri
        parsed = urlsplit(camera.uri)
        if "@" in parsed.netloc:
            return camera.uri
        auth = f"{quote(camera.username, safe='')}:{quote(camera.password, safe='')}@"
        return urlunsplit((parsed.scheme, auth + parsed.netloc, parsed.path, parsed.query, parsed.fragment))

    def _request_mux_pad(self, index: int):
        name = f"sink_{index}"
        request_simple = getattr(self.mux, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            pad = self.mux.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"nvstreammux could not allocate {name}")
        self._request_pads.append(pad)
        return pad

    def _add_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        source = self._make("nvurisrcbin", f"camera_v2_source_{index}")
        queue = self._make("queue", f"camera_v2_queue_{index}")

        # Never print this URI; credentials may be embedded here.
        source.set_property("uri", self._auth_uri(camera))
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 0)  # rtp-multi/automatic
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", self.low_latency_mode)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "gpu-id", self.gpu_id)

        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

        self.pipeline.add(source)
        self.pipeline.add(queue)

        mux_pad = self._request_mux_pad(index)
        qsrc = queue.get_static_pad("src")
        if qsrc.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: queue -> nvstreammux link failed")

        qsrc.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        source.connect("pad-added", self._source_pad_added, queue, cid)
        self.sources[cid] = source
        self.queues[cid] = queue

    def _source_pad_added(self, _source, pad, queue, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        if not str(structure.get_name()).startswith("video/"):
            return

        sink_pad = queue.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"CAMERA_V2 {cid} source link failed: {result}", file=sys.stderr, flush=True)

    def _source_probe(self, pad, info, cid: str):
        runtime = self.stats[cid]
        runtime.frames += 1
        buffer = info.get_buffer()

        if runtime.caps == "pending":
            caps = pad.get_current_caps()
            if caps is not None:
                runtime.caps = caps.to_string()
                print(f"CAMERA_V2 {cid} negotiated={runtime.caps}", flush=True)

        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts_ns = int(buffer.pts)
            previous = runtime.last_pts_ns
            runtime.last_pts_ns = pts_ns
            if previous is not None and pts_ns > previous:
                interval = (pts_ns - previous) / 1_000_000.0
                if 0.0 < interval < 500.0:
                    runtime.intervals_ms.append(interval)
        return self.Gst.PadProbeReturn.OK

    @staticmethod
    def _percentile(values: deque[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
        return ordered[index]

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
        now = time.monotonic()
        parts: list[str] = []
        for cid in [camera.camera_id for camera in self.cameras]:
            runtime = self.stats[cid]
            elapsed = max(0.001, now - runtime.last_stat_time)
            fps = (runtime.frames - runtime.last_frames) / elapsed
            runtime.last_frames = runtime.frames
            runtime.last_stat_time = now
            p50 = self._percentile(runtime.intervals_ms, 0.50)
            p95 = self._percentile(runtime.intervals_ms, 0.95)
            q = int(self.queues[cid].get_property("current-level-buffers"))
            cadence = "?" if p50 is None else f"{p50:.1f}/{p95:.1f}ms"
            parts.append(f"{cid}:{fps:.1f}fps pts50/95={cadence} q={q}")

        rendered, dropped = self._sink_stats()
        sink_text = "sink_stats=n/a"
        if rendered is not None or dropped is not None:
            sink_text = f"rendered={rendered if rendered is not None else '?'} dropped={dropped if dropped is not None else '?'}"

        print("CAMERA_V2 " + " | ".join(parts) + " || " + sink_text, flush=True)
        return not self._stopping

    def _redact(self, text: str) -> str:
        # Avoid exposing credentials if a GStreamer debug string happens to include
        # the full RTSP location.
        if not text:
            return ""
        import re

        return re.sub(r"(rtsps?://)[^/@\s]+:[^/@\s]+@", r"\1***:***@", text)

    def _warn_once(self, key: str, message: str) -> None:
        now = time.monotonic()
        previous = self._warning_last.get(key, 0.0)
        if now - previous >= 10.0:
            self._warning_last[key] = now
            print(message, flush=True)

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src else "unknown"
            print(
                f"CAMERA_V2 ERROR source={src} message={err.message} debug={self._redact(debug or '')}",
                file=sys.stderr,
                flush=True,
            )
            # nvurisrcbin owns RTSP reconnection. Do not kill all six streams for a
            # transient single-source error unless the pipeline itself reaches EOS.
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            src = message.src.get_name() if message.src else "unknown"
            key = f"{src}:{err.message}"
            self._warn_once(
                key,
                f"CAMERA_V2 WARNING source={src} message={err.message} debug={self._redact(debug or '')}",
            )
        elif message.type == self.Gst.MessageType.EOS:
            print("CAMERA_V2 EOS", file=sys.stderr, flush=True)
            self.stop()

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            self.loop.quit()
        except Exception:
            pass

    def run(self) -> int:
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("Camera V2 pipeline failed to enter PLAYING")

        print(
            "CAMERA_V2 started: 6x RTSP -> nvurisrcbin/NVDEC -> queue(1) -> "
            "nvstreammux -> nvmultistreamtiler -> queue(1) -> nveglglessink; "
            f"gpu={self.gpu_id} latency={self.rtsp_latency_ms}ms mux_timeout={self.mux_timeout_us}us "
            f"source={self.frame_width}x{self.frame_height}@~{self.source_fps} wall={self.wall_width}x{self.wall_height} "
            f"extra_surfaces={self.extra_surfaces} low_latency_mode={int(self.low_latency_mode)} sync=0 qos=0",
            flush=True,
        )

        def _signal(_signum, _frame):
            self.stop()

        old_int = signal.signal(signal.SIGINT, _signal)
        old_term = signal.signal(signal.SIGTERM, _signal)
        try:
            self.loop.run()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            self._stopping = True
            self.pipeline.set_state(self.Gst.State.NULL)
            for pad in self._request_pads:
                try:
                    self.mux.release_request_pad(pad)
                except Exception:
                    pass
        return 0


def main() -> int:
    return CameraWallV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
