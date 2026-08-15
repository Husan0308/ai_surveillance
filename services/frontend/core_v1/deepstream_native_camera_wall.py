from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

# Hybrid Intel/NVIDIA laptops often need PRIME offload for the EGL renderer.
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

from shared.config import camera_config
from services.ml_service.cameras.gstreamer import _gstreamer, authenticated_source


@dataclass
class SourceStats:
    frames: int = 0
    last_frames: int = 0
    last_print: float = 0.0
    last_pts_ns: int | None = None
    pts_ms_ema: float | None = None
    caps_text: str = "unknown"


class NativeCameraWall:
    """UI-free, AI-free, low-latency six-camera DeepStream wall.

    Hot path:
        RTSP -> nvurisrcbin/NVDEC -> latest-only queue -> nvstreammux
             -> nvmultistreamtiler -> latest-only queue -> nveglglessink

    There is intentionally no Qt, mmap, JPEG, appsink, NumPy, PyTorch,
    TensorRT inference, tracker, ReID, face model, or nvvideoconvert here.
    """

    def __init__(self):
        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.pipeline = Gst.Pipeline.new("native-camera-wall")
        if self.pipeline is None:
            raise RuntimeError("failed to create GStreamer pipeline")

        self.cameras = [
            dict(item)
            for item in camera_config().get("cameras", [])
            if item.get("online", True)
        ]
        if not self.cameras:
            raise RuntimeError("no enabled cameras")

        self.stats = {
            str(camera["id"]): SourceStats(last_print=time.monotonic())
            for camera in self.cameras
        }
        self.queues = {}
        self._request_pads = []

        self.rtsp_latency_ms = max(20, int(os.environ.get("AI_RTSP_LATENCY_MS", "80")))
        self.mux_timeout_us = max(1000, int(os.environ.get("AI_MUX_TIMEOUT_US", "15000")))
        self.udp_buffer = max(524288, int(os.environ.get("AI_RTSP_UDP_BUFFER", str(8 * 1024 * 1024))))
        self.extra_surfaces = max(1, int(os.environ.get("AI_DECODER_EXTRA_SURFACES", "8")))
        self.sink_sync = os.environ.get("AI_WALL_SINK_SYNC", "1").strip().lower() not in {"0", "false", "no"}

        # Keep the mux canvas at the configured camera resolution. Current project
        # streams are configured as 1280x720. Override only if the actual RTSP
        # caps printed at startup prove otherwise.
        self.frame_width = max(320, int(os.environ.get("AI_CAMERA_WIDTH", "1280")))
        self.frame_height = max(180, int(os.environ.get("AI_CAMERA_HEIGHT", "720")))

        count = len(self.cameras)
        rows = 2 if count > 3 else 1
        columns = 3 if count > 1 else 1

        # Native-detail wall: each 3x2 tile gets a full 1280x720 surface instead
        # of the old 640x360 tile. The window compositor may scale the final wall
        # to the monitor, but DeepStream itself no longer destroys detail first.
        self.wall_width = max(self.frame_width * columns, int(os.environ.get("AI_WALL_WIDTH", str(self.frame_width * columns))))
        self.wall_height = max(self.frame_height * rows, int(os.environ.get("AI_WALL_HEIGHT", str(self.frame_height * rows))))

        self.mux = self._make("nvstreammux", "mux")
        self.tiler = self._make("nvmultistreamtiler", "tiler")
        self.wall_queue = self._make("queue", "wall_queue")
        self.sink = self._make("nveglglessink", "wall_sink")

        self._set_if(self.mux, "batch-size", count)
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", self.frame_width)
        self._set_if(self.mux, "height", self.frame_height)
        # Low latency: do not wait a whole 20 FPS frame period for every camera.
        # nvmultistreamtiler caches the previous frame for a source when needed.
        self._set_if(self.mux, "batched-push-timeout", self.mux_timeout_us)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "max-latency", 0)
        self._set_if(self.mux, "buffer-pool-size", 8)
        self._set_if(self.mux, "nvbuf-memory-type", 2)  # CUDA device memory
        self._set_if(self.mux, "gpu-id", 0)

        self._set_if(self.tiler, "rows", rows)
        self._set_if(self.tiler, "columns", columns)
        self._set_if(self.tiler, "width", self.wall_width)
        self._set_if(self.tiler, "height", self.wall_height)
        self._set_if(self.tiler, "gpu-id", 0)
        self._set_if(self.tiler, "nvbuf-memory-type", 2)

        # One output buffer only. If rendering ever stalls, discard the old wall
        # instead of accumulating visible latency.
        self._set_if(self.wall_queue, "max-size-buffers", 1)
        self._set_if(self.wall_queue, "max-size-bytes", 0)
        self._set_if(self.wall_queue, "max-size-time", 0)
        self._set_if(self.wall_queue, "leaky", 2)

        # PTS pacing gives smoother 20 FPS motion without the old cross-camera
        # 300 ms synchronization buffer. Set AI_WALL_SINK_SYNC=0 for absolute
        # minimum latency if needed.
        self._set_if(self.sink, "sync", self.sink_sync)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "enable-last-sample", False)
        self._set_if(self.sink, "gpu-id", 0)

        for element in (self.mux, self.tiler, self.wall_queue, self.sink):
            self.pipeline.add(element)

        if not self.mux.link(self.tiler):
            raise RuntimeError("failed nvstreammux -> nvmultistreamtiler")
        if not self.tiler.link(self.wall_queue):
            raise RuntimeError("failed nvmultistreamtiler -> wall queue")
        if not self.wall_queue.link(self.sink):
            raise RuntimeError("failed wall queue -> nveglglessink")

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
            raise RuntimeError(f"missing GStreamer/DeepStream element: {factory}")
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
        cid = str(camera["id"])
        uri = authenticated_source(
            {**camera, "source": camera.get("display_source") or camera.get("source")}
        )
        if not isinstance(uri, str) or not uri.startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"{cid}: invalid RTSP source")

        source = self._make("nvurisrcbin", f"src_{index}")
        queue = self._make("queue", f"src_queue_{index}")

        source.set_property("uri", uri)
        self._set_if(source, "disable-audio", True)
        # rtp-multi: UDP on a clean LAN, with TCP available as fallback.
        self._set_if(source, "select-rtp-protocol", 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", True)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer)
        self._set_if(source, "rtsp-reconnect-interval", 5)
        self._set_if(source, "rtsp-reconnect-attempts", -1)

        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)

        self.pipeline.add(source)
        self.pipeline.add(queue)

        mux_pad = self._request_mux_pad(index)
        qsrc = queue.get_static_pad("src")
        if qsrc.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: queue -> streammux failed")

        qsrc.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        source.connect("pad-added", self._on_source_pad_added, queue, cid)
        self.queues[cid] = queue

    def _on_source_pad_added(self, _source, pad, queue, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        media = structure.get_name()
        if not str(media).startswith("video/"):
            return

        self.stats[cid].caps_text = caps.to_string()
        print(f"NATIVE_CAMERA {cid} decoded_caps={caps.to_string()}", flush=True)

        sinkpad = queue.get_static_pad("sink")
        if sinkpad.is_linked():
            return
        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"NATIVE_CAMERA {cid} source link failed: {result}", flush=True)

    def _source_probe(self, _pad, info, cid: str):
        buffer = info.get_buffer()
        stat = self.stats[cid]
        stat.frames += 1
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts = int(buffer.pts)
            if stat.last_pts_ns is not None and pts > stat.last_pts_ns:
                interval_ms = (pts - stat.last_pts_ns) / 1_000_000.0
                if 0.0 < interval_ms < 1000.0:
                    stat.pts_ms_ema = (
                        interval_ms
                        if stat.pts_ms_ema is None
                        else stat.pts_ms_ema * 0.9 + interval_ms * 0.1
                    )
            stat.last_pts_ns = pts
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        now = time.monotonic()
        parts = []
        for cid, stat in self.stats.items():
            elapsed = max(0.001, now - stat.last_print)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_print = now
            q = int(self.queues[cid].get_property("current-level-buffers"))
            pts = f"{stat.pts_ms_ema:.1f}ms" if stat.pts_ms_ema is not None else "?"
            parts.append(f"{cid}:{fps:.1f}fps pts={pts} q={q}")
        print("NATIVE_CAMERA " + " | ".join(parts), flush=True)
        return True

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"NATIVE_CAMERA ERROR source={source} message={err.message} debug={debug or ''}",
                file=sys.stderr,
                flush=True,
            )
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            source = message.src.get_name() if message.src else "unknown"
            # DeepStream 7.1 nvv4l2decoder can emit a harmless capability-query
            # warning on dGPU. Keep warnings visible but do not reconnect sources.
            print(
                f"NATIVE_CAMERA WARNING source={source} message={err.message} debug={debug or ''}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.EOS:
            self.loop.quit()

    def run(self) -> int:
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("native camera pipeline failed to PLAY")

        print(
            "NATIVE_CAMERA started: UI=off AI=off mmap=off JPEG=off; "
            f"RTSP latency={self.rtsp_latency_ms}ms low_latency_mode=1 "
            f"mux_timeout={self.mux_timeout_us}us sink_sync={int(self.sink_sync)} "
            f"wall={self.wall_width}x{self.wall_height}",
            flush=True,
        )
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(self.Gst.State.NULL)
        return 0


def run() -> int:
    return NativeCameraWall().run()


if __name__ == "__main__":
    raise SystemExit(run())
