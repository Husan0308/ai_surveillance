from __future__ import annotations

import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field

from .config import CameraConfig, load_settings


@dataclass
class CameraStats:
    frames: int = 0
    last_frames: int = 0
    last_stat_at: float = field(default_factory=time.monotonic)
    last_pts_ns: int | None = None
    intervals_ms: deque[float] = field(default_factory=lambda: deque(maxlen=240))
    caps: str = "pending"


class CameraServiceRuntime:
    """AI-free six-camera data plane.

    Production/headless mode:
        RTSP/NVDEC -> latest-only queue -> fakesink

    Optional debug-wall mode:
        RTSP/NVDEC -> latest-only queue -> nvstreammux -> tiler -> EGL

    The production camera service must not spend GPU compute on presentation.
    UI/video presentation is a separate service boundary. This process imports no
    detector, TensorRT, tracker, ReID, identity, API or frontend code.
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
        self.headless = os.environ.get("CAMERA_SERVICE_HEADLESS", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.stats = {camera.camera_id: CameraStats() for camera in self.cameras}
        self.sources: dict[str, object] = {}
        self.queues: dict[str, object] = {}
        self.headless_sinks: dict[str, object] = {}
        self.request_pads: list[tuple[object, object]] = []
        self.stopping = False

        required = ["nvurisrcbin", "queue"]
        if self.headless:
            required.append("fakesink")
        else:
            required.extend(
                [
                    "nvstreammux",
                    "nvmultistreamtiler",
                    "nvvideoconvert",
                    "capsfilter",
                    "nveglglessink",
                ]
            )
        missing = [name for name in required if Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("camera_service missing plugins: " + ", ".join(missing))

        self.pipeline = Gst.Pipeline.new("camera-service")
        if self.pipeline is None:
            raise RuntimeError("could not create camera-service pipeline")

        self.mux = None
        self.tiler = None
        self.wall_queue = None
        self.convert = None
        self.caps = None
        self.sink = None

        if not self.headless:
            self.mux = self._make("nvstreammux", "camera_service_mux")
            self.tiler = self._make("nvmultistreamtiler", "camera_service_tiler")
            self.wall_queue = self._make("queue", "camera_service_wall_queue")
            self.convert = self._make("nvvideoconvert", "camera_service_convert")
            self.caps = self._make("capsfilter", "camera_service_rgba")
            self.sink = self._make("nveglglessink", "camera_service_sink")

            self._configure_graph()
            for element in (
                self.mux,
                self.tiler,
                self.wall_queue,
                self.convert,
                self.caps,
                self.sink,
            ):
                self.pipeline.add(element)
            self._link(self.mux, self.tiler, "mux->tiler")
            self._link(self.tiler, self.wall_queue, "tiler->wall_queue")
            self._link(self.wall_queue, self.convert, "wall_queue->convert")
            self._link(self.convert, self.caps, "convert->rgba")
            self._link(self.caps, self.sink, "rgba->egl")

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()

        if self.headless:
            path = "RTSP/NVDEC->latest->fakesink"
            mode = "headless render=0 tiler=0 egl=0"
        else:
            path = "RTSP/NVDEC->latest->mux->tiler->EGL"
            mode = (
                f"debug-wall render=1 display={self.settings.display_width}x{self.settings.display_height} "
                f"wall={self.settings.wall_width}x{self.settings.wall_height}"
            )
        print(
            "CAMERA_SERVICE_ARCH "
            f"sources={len(self.cameras)} ai=0 tracker=0 detector=0 reid=0 "
            f"source_target={self.settings.source_fps}fps mode={mode} path={path}",
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

    def _configure_fakesink(self, sink) -> None:
        self._set_if(sink, "sync", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "enable-last-sample", False)

    def _configure_graph(self) -> None:
        if self.headless:
            return
        s = self.settings
        self._set_if(self.mux, "batch-size", len(self.cameras))
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", s.display_width)
        self._set_if(self.mux, "height", s.display_height)
        self._set_if(self.mux, "enable-padding", False)
        self._set_if(self.mux, "batched-push-timeout", round(1_000_000 / s.source_fps))
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "max-latency", 0)
        self._set_if(self.mux, "buffer-pool-size", max(8, len(self.cameras) + 2))
        self._set_if(self.mux, "nvbuf-memory-type", 2)
        self._set_if(self.mux, "gpu-id", s.gpu_id)
        self._set_if(self.mux, "compute-hw", 1)
        self._set_if(self.mux, "interpolation-method", 2)

        rows = max(1, math.ceil(len(self.cameras) / 3))
        cols = min(3, len(self.cameras))
        self._set_if(self.tiler, "rows", rows)
        self._set_if(self.tiler, "columns", cols)
        self._set_if(self.tiler, "width", s.wall_width)
        self._set_if(self.tiler, "height", s.wall_height)
        self._set_if(self.tiler, "gpu-id", s.gpu_id)
        self._set_if(self.tiler, "nvbuf-memory-type", 2)
        self._set_if(self.tiler, "compute-hw", 1)
        self._set_if(self.tiler, "interpolation-method", 2)

        self._latest_queue(self.wall_queue)
        self._set_if(self.convert, "gpu-id", s.gpu_id)
        self.caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "async", False)
        self._set_if(self.sink, "enable-last-sample", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "render-delay", 0)
        self._set_if(self.sink, "throttle-time", 0)
        self._set_if(self.sink, "force-aspect-ratio", True)
        self._set_if(self.sink, "gpu-id", s.gpu_id)

    def _request_mux_pad(self, index: int):
        if self.mux is None:
            raise RuntimeError("mux pad requested in headless camera mode")
        name = f"sink_{index}"
        request_simple = getattr(self.mux, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            pad = self.mux.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"could not allocate mux {name}")
        self.request_pads.append((self.mux, pad))
        return pad

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera: CameraConfig) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)
        if self.settings.rtsp_transport == "tcp":
            self._set_if(element, "protocols", 4)
            self._set_if(element, "tcp-timestamp", True)
        elif self.settings.rtsp_transport == "udp":
            self._set_if(element, "protocols", 1)
        self._set_if(element, "latency", self.settings.latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "buffer-mode", 3)
        self._set_if(element, "do-rtsp-keep-alive", True)

    def _add_camera(self, index: int, camera: CameraConfig) -> None:
        source = self._make("nvurisrcbin", f"camera_service_source_{index}")
        queue = self._make("queue", f"camera_service_q_{index}")
        self._latest_queue(queue)

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 4 if self.settings.rtsp_transport == "tcp" else 0)
        self._set_if(source, "latency", self.settings.latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", self.settings.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", 3)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)
        self._set_if(source, "gpu-id", self.settings.gpu_id)

        self.pipeline.add(source)
        self.pipeline.add(queue)
        if self.headless:
            sink = self._make("fakesink", f"camera_service_fakesink_{index}")
            self._configure_fakesink(sink)
            self.pipeline.add(sink)
            self._link(queue, sink, f"{camera.camera_id}:queue->fakesink")
            self.headless_sinks[camera.camera_id] = sink
        else:
            mux_pad = self._request_mux_pad(index)
            if queue.get_static_pad("src").link(mux_pad) != self.Gst.PadLinkReturn.OK:
                raise RuntimeError(f"{camera.camera_id}: queue->mux failed")

        queue.get_static_pad("sink").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._source_probe,
            camera.camera_id,
        )
        source.connect("pad-added", self._source_pad_added, queue, camera.camera_id)
        self.sources[camera.camera_id] = source
        self.queues[camera.camera_id] = queue

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
            print(f"CAMERA_SERVICE {cid} source link failed result={result}", file=sys.stderr, flush=True)

    def _source_probe(self, pad, info, cid: str):
        stat = self.stats[cid]
        stat.frames += 1
        if stat.caps == "pending":
            caps = pad.get_current_caps()
            if caps is not None:
                stat.caps = caps.to_string()
                print(f"CAMERA_SERVICE_SOURCE {cid} negotiated={stat.caps}", flush=True)
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

    @staticmethod
    def _percentile(values, p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
        return float(ordered[idx])

    def _print_stats(self) -> bool:
        if self.stopping:
            return False
        now = time.monotonic()
        parts = []
        for camera in self.cameras:
            stat = self.stats[camera.camera_id]
            elapsed = max(0.001, now - stat.last_stat_at)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_stat_at = now
            p50 = self._percentile(stat.intervals_ms, 0.50)
            p95 = self._percentile(stat.intervals_ms, 0.95)
            cadence = "?" if p50 is None else f"{p50:.0f}/{p95:.0f}ms"
            q = int(self.queues[camera.camera_id].get_property("current-level-buffers"))
            parts.append(f"{camera.camera_id}:{fps:.1f}fps pts={cadence} q={q}")
        print("CAMERA_SERVICE_STATS " + " | ".join(parts), flush=True)
        return True

    def _start_sources(self) -> None:
        for source in self.sources.values():
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)
        for index, camera in enumerate(self.cameras):
            delay_ms = max(1, int(round(index * self.settings.startup_stagger_sec * 1000.0)))

            def _start(cid=camera.camera_id, ordinal=index):
                if self.stopping:
                    return False
                source = self.sources[cid]
                source.set_locked_state(False)
                ok = bool(source.sync_state_with_parent())
                print(f"CAMERA_SERVICE_START cid={cid} index={ordinal} sync={int(ok)}", flush=True)
                return False

            self.GLib.timeout_add(delay_ms, _start)

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"CAMERA_SERVICE_ERROR {err} debug={debug}", file=sys.stderr, flush=True)
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"CAMERA_SERVICE_WARNING {err} debug={debug}", file=sys.stderr, flush=True)
        elif message.type == self.Gst.MessageType.EOS:
            print("CAMERA_SERVICE_EOS", file=sys.stderr, flush=True)

    def stop(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        try:
            self.pipeline.set_state(self.Gst.State.NULL)
        finally:
            try:
                self.loop.quit()
            except Exception:
                pass

    def run(self) -> int:
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("camera_service failed to enter PLAYING")
        self._start_sources()
        self.GLib.timeout_add_seconds(5, self._print_stats)
        try:
            self.loop.run()
        finally:
            self.stop()
            for mux, pad in self.request_pads:
                try:
                    mux.release_request_pad(pad)
                except Exception:
                    pass
        return 0


def main() -> int:
    return CameraServiceRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
