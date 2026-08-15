from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from shared.config import camera_config
from services.ml_service.cameras.gstreamer import (
    _gstreamer,
    authenticated_source,
)


CAMERA_IDS = [f"CAM-{index:02d}" for index in range(1, 7)]


@dataclass
class SourceStats:
    frames: int = 0
    last_frames: int = 0
    last_print: float = 0.0
    last_pts_ns: int | None = None
    interval_ms_ema: float | None = None


class DeepStreamGpuWall:
    """Pure-GPU six-camera display baseline.

    RTSP -> nvurisrcbin/NVDEC -> nvstreammux -> nvmultistreamtiler ->
    nvvideoconvert -> nveglglessink

    No appsink, NumPy, BGR host mapping, mmap frame transport, JPEG encoding,
    detector, tracker, ReID or face model exists in this path. This is the
    baseline used to decide whether remaining stutter comes from RTSP/NVDEC or
    from the Python/Qt transport path.
    """

    def __init__(self):
        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.pipeline = Gst.Pipeline.new("deepstream-gpu-wall")
        if self.pipeline is None:
            raise RuntimeError("failed to create GStreamer pipeline")

        self.cameras = [
            dict(item)
            for item in camera_config().get("cameras", [])
            if item.get("online", True)
        ]
        if not self.cameras:
            raise RuntimeError("no enabled cameras in config/cameras.yaml")

        self.stats: dict[str, SourceStats] = {
            str(item["id"]): SourceStats(last_print=time.monotonic())
            for item in self.cameras
        }
        self.queues = {}
        self.sources = {}
        self._request_pads = []

        self.mux = self._make("nvstreammux", "mux")
        self.tiler = self._make("nvmultistreamtiler", "tiler")
        self.convert = self._make("nvvideoconvert", "wall_convert")
        self.capsfilter = self._make("capsfilter", "wall_caps")
        self.sink_queue = self._make("queue", "wall_queue")
        self.sink = self._make("nveglglessink", "wall_sink")

        count = len(self.cameras)
        self._set_if(self.mux, "batch-size", count)
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", 1280)
        self._set_if(self.mux, "height", 720)
        # 30 FPS fastest configured camera -> ~33.3 ms. Do not wait for a slow
        # camera to fill a complete batch.
        self._set_if(self.mux, "batched-push-timeout", 33333)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "buffer-pool-size", 8)

        rows = 2 if count > 3 else 1
        columns = 3 if count > 1 else 1
        self._set_if(self.tiler, "rows", rows)
        self._set_if(self.tiler, "columns", columns)
        # 3x2 cells remain 16:9: 640x360 each.
        self._set_if(self.tiler, "width", 1920)
        self._set_if(self.tiler, "height", 720)
        self._set_if(self.tiler, "gpu-id", 0)

        self.capsfilter.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(self.sink_queue, "max-size-buffers", 2)
        self._set_if(self.sink_queue, "max-size-bytes", 0)
        self._set_if(self.sink_queue, "max-size-time", 0)
        self._set_if(self.sink_queue, "leaky", 2)  # downstream: discard old
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "enable-last-sample", False)

        for element in (
            self.mux,
            self.tiler,
            self.convert,
            self.capsfilter,
            self.sink_queue,
            self.sink,
        ):
            self.pipeline.add(element)

        if not self.mux.link(self.tiler):
            raise RuntimeError("failed to link nvstreammux -> nvmultistreamtiler")
        if not self.tiler.link(self.convert):
            raise RuntimeError("failed to link tiler -> nvvideoconvert")
        if not self.convert.link(self.capsfilter):
            raise RuntimeError("failed to link converter -> RGBA NVMM caps")
        if not self.capsfilter.link(self.sink_queue):
            raise RuntimeError("failed to link caps -> sink queue")
        if not self.sink_queue.link(self.sink):
            raise RuntimeError("failed to link queue -> nveglglessink")

        for index, camera in enumerate(self.cameras):
            self._add_source(index, camera)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()
        GLib.timeout_add_seconds(5, self._print_stats)

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(
                f"required GStreamer/DeepStream element is missing: {factory}"
            )
        return element

    @staticmethod
    def _set_if(element, name: str, value) -> bool:
        if element.find_property(name) is None:
            return False
        element.set_property(name, value)
        return True

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

    def _add_source(self, index: int, camera: dict) -> None:
        camera_id = str(camera["id"])
        uri = authenticated_source(
            {
                **camera,
                "source": camera.get("display_source") or camera.get("source"),
            }
        )
        if not isinstance(uri, str) or not uri.startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"{camera_id}: invalid RTSP source")

        source = self._make("nvurisrcbin", f"src_{index}")
        queue = self._make("queue", f"src_queue_{index}")

        source.set_property("uri", uri)
        self._set_if(source, "disable-audio", True)
        # rtp-multi lets GStreamer use UDP on a clean LAN and fall back to TCP;
        # forcing TCP can turn one lost packet into head-of-line blocking.
        self._set_if(source, "select-rtp-protocol", 0)
        self._set_if(source, "latency", 100)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", 6)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", 4 * 1024 * 1024)
        self._set_if(source, "rtsp-reconnect-interval", 5)
        self._set_if(source, "rtsp-reconnect-attempts", -1)

        # One newest decoded buffer may wait for streammux. This is intentionally
        # tiny: a slow consumer must drop stale frames rather than build latency.
        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)

        self.pipeline.add(source)
        self.pipeline.add(queue)

        mux_pad = self._request_mux_pad(index)
        qsrc = queue.get_static_pad("src")
        if qsrc.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera_id}: failed queue -> streammux link")

        qsrc.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._source_probe,
            camera_id,
        )
        source.connect("pad-added", self._on_source_pad_added, queue, camera_id)

        self.sources[camera_id] = source
        self.queues[camera_id] = queue

    def _on_source_pad_added(self, _source, pad, queue, camera_id: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        media = caps.get_structure(0).get_name()
        if not str(media).startswith("video/"):
            return
        sinkpad = queue.get_static_pad("sink")
        if sinkpad.is_linked():
            return
        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"GPU_WALL {camera_id} dynamic-link failed: {result}", flush=True)

    def _source_probe(self, _pad, info, camera_id: str):
        buffer = info.get_buffer()
        stat = self.stats[camera_id]
        stat.frames += 1
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts = int(buffer.pts)
            if stat.last_pts_ns is not None and pts > stat.last_pts_ns:
                interval_ms = (pts - stat.last_pts_ns) / 1_000_000.0
                if 0.0 < interval_ms < 2000.0:
                    if stat.interval_ms_ema is None:
                        stat.interval_ms_ema = interval_ms
                    else:
                        stat.interval_ms_ema = (
                            stat.interval_ms_ema * 0.90 + interval_ms * 0.10
                        )
            stat.last_pts_ns = pts
        return self.Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"GPU_WALL ERROR source={source} message={err.message} debug={debug or ''}",
                flush=True,
            )
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"GPU_WALL WARNING source={source} message={err.message} debug={debug or ''}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.EOS:
            print("GPU_WALL EOS", flush=True)
            self.loop.quit()

    def _print_stats(self) -> bool:
        now = time.monotonic()
        parts = []
        for camera in self.cameras:
            camera_id = str(camera["id"])
            stat = self.stats[camera_id]
            elapsed = max(0.001, now - stat.last_print)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_print = now
            queue = self.queues[camera_id]
            queued = int(queue.get_property("current-level-buffers"))
            pts_text = (
                "n/a"
                if stat.interval_ms_ema is None
                else f"{stat.interval_ms_ema:.1f}ms"
            )
            parts.append(
                f"{camera_id}:{fps:.1f}fps pts={pts_text} q={queued}"
            )
        print("GPU_WALL " + " | ".join(parts), flush=True)
        return True

    def run(self) -> int:
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("DeepStream GPU wall failed to enter PLAYING")
        print(
            "GPU_WALL started: RTSP -> NVDEC -> nvstreammux -> tiler -> EGL; "
            "no Python frame copies",
            flush=True,
        )
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(self.Gst.State.NULL)
            for pad in self._request_pads:
                try:
                    self.mux.release_request_pad(pad)
                except Exception:
                    pass
        return 0


def run() -> int:
    return DeepStreamGpuWall().run()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"GPU_WALL FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
