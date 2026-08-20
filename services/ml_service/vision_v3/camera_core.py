from __future__ import annotations

import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# DeepStream 7.1: keep the legacy nvstreammux. It gives deterministic width,
# height, live-source and memory/pool controls on the target machine.
os.environ.pop("USE_NEW_NVSTREAMMUX", None)

ROOT = Path(__file__).resolve().parents[3]
CAMERAS_PATH = ROOT / "config" / "cameras.yaml"
CORE_PATH = ROOT / "config" / "vision_v3_camera.yaml"


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    uri: str
    username: str = ""
    password: str = ""


@dataclass
class SourceStats:
    frames: int = 0
    last_frames: int = 0
    last_stat_t: float = field(default_factory=time.monotonic)
    last_pts_ns: int | None = None
    intervals_ms: deque[float] = field(default_factory=lambda: deque(maxlen=240))
    negotiated_caps: str = "pending"


class SixCameraCore:
    """Single production camera owner for Vision V3.

    Hot path:
      6x RTSP -> nvurisrcbin/NVDEC -> queue(1, leaky) -> nvstreammux(batch=6)
      -> nvmultistreamtiler -> queue(1, leaky) -> nveglglessink

    There is intentionally no detector, tracker, ReID, face recognition, JPEG,
    NumPy frame copy, HTTP or Qt logic in this module. The sole acceptance target
    is a smooth, bounded-latency six-camera GPU-native wall.
    """

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self.Gst = Gst
        self.GLib = GLib

        self.cameras = self._load_cameras()
        self.cfg = self._load_core_config()
        if len(self.cameras) != 6:
            raise RuntimeError(f"Vision V3 camera core requires exactly 6 enabled cameras, found {len(self.cameras)}")

        self.gpu_id = int(self.cfg.get("gpu_id", 0))
        self.source_fps = max(1, int(self.cfg.get("source_fps", 20)))
        self.rtsp_latency_ms = max(40, int(self.cfg.get("rtsp_latency_ms", 100)))
        self.udp_buffer_size = max(1_048_576, int(self.cfg.get("udp_buffer_size", 8 * 1024 * 1024)))
        self.extra_surfaces = max(2, min(16, int(self.cfg.get("decoder_extra_surfaces", 6))))
        self.low_latency_mode = bool(self.cfg.get("low_latency_mode", False))
        self.working_width = max(320, int(self.cfg.get("working_width", 1280)))
        self.working_height = max(180, int(self.cfg.get("working_height", 720)))
        self.tiler_columns = int(self.cfg.get("tiler_columns", 3))
        self.tiler_rows = int(self.cfg.get("tiler_rows", 2))
        self.wall_width = max(960, int(self.cfg.get("wall_width", 1920)))
        self.wall_height = max(360, int(self.cfg.get("wall_height", 720)))
        self.stats_interval = max(1, int(self.cfg.get("stats_interval_sec", 5)))
        self.queue_buffers = max(1, int(self.cfg.get("queue_buffers", 1)))
        self.sync_inputs = bool(self.cfg.get("sync_inputs", False))
        self.sink_sync = bool(self.cfg.get("sink_sync", False))
        self.sink_qos = bool(self.cfg.get("sink_qos", False))
        self.mux_timeout_us = max(5_000, round(1_000_000 / self.source_fps))

        self._stopping = False
        self._request_pads: list[Any] = []
        self.sources: dict[str, Any] = {}
        self.queues: dict[str, Any] = {}
        self.stats = {cam.camera_id: SourceStats() for cam in self.cameras}

        self._preflight()
        self.pipeline = Gst.Pipeline.new("vision-v3-six-camera-core")
        if self.pipeline is None:
            raise RuntimeError("Could not create GStreamer pipeline")

        self.mux = self._make("nvstreammux", "v3_mux")
        self.tiler = self._make("nvmultistreamtiler", "v3_tiler")
        self.wall_queue = self._make("queue", "v3_wall_queue")
        self.sink = self._make("nveglglessink", "v3_sink")

        self._configure_mux()
        self._configure_tiler()
        self._configure_queue(self.wall_queue)
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
        GLib.timeout_add_seconds(self.stats_interval, self._print_stats)

    @staticmethod
    def _yaml(path: Path) -> dict:
        if not path.exists():
            raise RuntimeError(f"missing config: {path}")
        with path.open("r", encoding="utf-8") as fh:
            value = yaml.safe_load(fh) or {}
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid YAML root in {path}")
        return value

    def _load_core_config(self) -> dict:
        return dict(self._yaml(CORE_PATH).get("camera_core") or {})

    def _load_cameras(self) -> list[CameraSpec]:
        data = self._yaml(CAMERAS_PATH)
        rows = data.get("cameras") or []
        global_user = os.environ.get("RTSP_USERNAME", "")
        global_pw = os.environ.get("RTSP_PASSWORD", "")
        cameras: list[CameraSpec] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("enabled", True) is False:
                continue
            camera_id = str(row.get("id") or row.get("camera_id") or "").strip()
            uri = str(row.get("uri") or "").strip()
            if not camera_id or not uri:
                raise RuntimeError("every enabled camera needs id and uri")
            username = str(row.get("username") or global_user or "")
            password = str(row.get("password") or global_pw or "")
            cameras.append(CameraSpec(camera_id, uri, username, password))
        return cameras

    def _preflight(self) -> None:
        required = ("nvurisrcbin", "nvstreammux", "nvmultistreamtiler", "nveglglessink", "queue")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("Missing DeepStream/GStreamer plugins: " + ", ".join(missing))

        try:
            with open("/proc/sys/net/core/rmem_max", "r", encoding="utf-8") as fh:
                rmem_max = int(fh.read().strip())
            if rmem_max < self.udp_buffer_size:
                print(
                    f"V3 WARNING kernel rmem_max={rmem_max} < requested UDP buffer {self.udp_buffer_size}",
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

    def _configure_queue(self, queue) -> None:
        self._set_if(queue, "max-size-buffers", self.queue_buffers)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)  # downstream: keep newest buffer
        self._set_if(queue, "silent", True)

    def _configure_mux(self) -> None:
        self._set_if(self.mux, "batch-size", len(self.cameras))
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", self.working_width)
        self._set_if(self.mux, "height", self.working_height)
        self._set_if(self.mux, "enable-padding", False)
        self._set_if(self.mux, "batched-push-timeout", self.mux_timeout_us)
        self._set_if(self.mux, "sync-inputs", self.sync_inputs)
        self._set_if(self.mux, "max-latency", 0)
        self._set_if(self.mux, "buffer-pool-size", max(8, len(self.cameras) + 2))
        self._set_if(self.mux, "nvbuf-memory-type", 2)
        self._set_if(self.mux, "gpu-id", self.gpu_id)

    def _configure_tiler(self) -> None:
        self._set_if(self.tiler, "rows", self.tiler_rows)
        self._set_if(self.tiler, "columns", self.tiler_columns)
        self._set_if(self.tiler, "width", self.wall_width)
        self._set_if(self.tiler, "height", self.wall_height)
        self._set_if(self.tiler, "gpu-id", self.gpu_id)
        self._set_if(self.tiler, "nvbuf-memory-type", 2)

    def _configure_sink(self) -> None:
        self._set_if(self.sink, "sync", self.sink_sync)
        self._set_if(self.sink, "qos", self.sink_qos)
        self._set_if(self.sink, "async", False)
        self._set_if(self.sink, "enable-last-sample", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "render-delay", 0)
        self._set_if(self.sink, "throttle-time", 0)
        self._set_if(self.sink, "force-aspect-ratio", True)
        self._set_if(self.sink, "gpu-id", self.gpu_id)

    def _request_mux_pad(self, index: int):
        name = f"sink_{index}"
        request_simple = getattr(self.mux, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else self.mux.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"nvstreammux could not allocate {name}")
        self._request_pads.append(pad)
        return pad

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera: CameraSpec) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)
        self._set_if(element, "latency", self.rtsp_latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "buffer-mode", 3)
        print(
            f"V3 {camera.camera_id} rtspsrc configured auth={'yes' if camera.username else 'no'} latency={self.rtsp_latency_ms}ms",
            flush=True,
        )

    def _add_camera(self, index: int, camera: CameraSpec) -> None:
        source = self._make("nvurisrcbin", f"v3_source_{index}")
        queue = self._make("queue", f"v3_source_queue_{index}")
        self._configure_queue(queue)

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 0)  # automatic UDP/TCP selection
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", self.low_latency_mode)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "gpu-id", self.gpu_id)

        self.pipeline.add(source)
        self.pipeline.add(queue)

        mux_pad = self._request_mux_pad(index)
        qsrc = queue.get_static_pad("src")
        if qsrc.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera.camera_id}: queue -> nvstreammux link failed")

        qsrc.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, camera.camera_id)
        source.connect("pad-added", self._source_pad_added, queue, camera.camera_id)
        self.sources[camera.camera_id] = source
        self.queues[camera.camera_id] = queue

    def _source_pad_added(self, _source, pad, queue, camera_id: str) -> None:
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
            print(f"V3 {camera_id} source link failed: {result}", file=sys.stderr, flush=True)

    def _source_probe(self, pad, info, camera_id: str):
        row = self.stats[camera_id]
        row.frames += 1
        buffer = info.get_buffer()

        if row.negotiated_caps == "pending":
            caps = pad.get_current_caps()
            if caps is not None:
                row.negotiated_caps = caps.to_string()
                print(f"V3 {camera_id} negotiated={row.negotiated_caps}", flush=True)

        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts_ns = int(buffer.pts)
            previous = row.last_pts_ns
            row.last_pts_ns = pts_ns
            if previous is not None and pts_ns > previous:
                interval = (pts_ns - previous) / 1_000_000.0
                if 0.0 < interval < 500.0:
                    row.intervals_ms.append(interval)
        return self.Gst.PadProbeReturn.OK

    @staticmethod
    def _percentile(values: deque[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
        return ordered[index]

    def _print_stats(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        pieces: list[str] = []
        for camera in self.cameras:
            cid = camera.camera_id
            row = self.stats[cid]
            elapsed = max(0.001, now - row.last_stat_t)
            fps = (row.frames - row.last_frames) / elapsed
            row.last_frames = row.frames
            row.last_stat_t = now
            p50 = self._percentile(row.intervals_ms, 0.50)
            p95 = self._percentile(row.intervals_ms, 0.95)
            q = int(self.queues[cid].get_property("current-level-buffers"))
            cadence = "?" if p50 is None else f"{p50:.1f}/{p95:.1f}ms"
            pieces.append(f"{cid}:{fps:.1f}fps pts50/95={cadence} q={q}")
        print("V3_STATS " + " | ".join(pieces), flush=True)
        return True

    def _on_bus_message(self, _bus, message) -> None:
        kind = message.type
        if kind == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            source_name = message.src.get_name() if message.src else "unknown"
            print(f"V3 ERROR source={source_name}: {err}; debug={debug}", file=sys.stderr, flush=True)
        elif kind == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            source_name = message.src.get_name() if message.src else "unknown"
            print(f"V3 WARNING source={source_name}: {err}; debug={debug}", file=sys.stderr, flush=True)
        elif kind == self.Gst.MessageType.EOS:
            print("V3 EOS", file=sys.stderr, flush=True)

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            self.pipeline.set_state(self.Gst.State.NULL)
        finally:
            for pad in self._request_pads:
                try:
                    self.mux.release_request_pad(pad)
                except Exception:
                    pass
            if self.loop.is_running():
                self.loop.quit()

    def run(self) -> int:
        print(
            "V3_CAMERA_CORE starting cameras="
            + ",".join(cam.camera_id for cam in self.cameras)
            + f" work={self.working_width}x{self.working_height} wall={self.wall_width}x{self.wall_height}",
            flush=True,
        )
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("DeepStream pipeline failed to enter PLAYING")

        def _signal(_signum, _frame):
            self.GLib.idle_add(self._stop_from_glib)

        signal.signal(signal.SIGINT, _signal)
        signal.signal(signal.SIGTERM, _signal)
        try:
            self.loop.run()
        finally:
            self.stop()
        return 0

    def _stop_from_glib(self) -> bool:
        self.stop()
        return False


def main() -> int:
    try:
        return SixCameraCore().run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"V3_FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
