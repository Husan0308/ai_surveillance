from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from shared.config import camera_config
from services.ml_service.cameras.gstreamer import _gstreamer, authenticated_source


ROOT = Path(__file__).resolve().parents[3]
BATCH_SIZE = 6
MODEL_HEIGHT = 416
MODEL_WIDTH = 736
ONNX_PATH = ROOT / "models" / "yolo26m_b6_416x736.onnx"
ENGINE_PATH = ROOT / "models" / "yolo26m_b6_416x736_fp16.engine"


@dataclass
class SourceStats:
    frames: int = 0
    last_frames: int = 0
    last_print: float = 0.0
    last_pts_ns: int | None = None
    interval_ms_ema: float | None = None


class DeepStreamBatch6Wall:
    """Six RTSP cameras -> one nvstreammux batch -> one nvinfer call -> GPU wall.

    Hot path stays on NVIDIA memory until rendering:
        RTSP -> nvurisrcbin/NVDEC -> nvstreammux(batch=6)
             -> nvinfer(batch=6, TensorRT FP16)
             -> nvmultistreamtiler -> nvvideoconvert -> nveglglessink

    nvinfer runs YOLO26 as network-type=Other and attaches the tiny raw output
    tensor as metadata. Image frames are never copied to NumPy/Python here.
    Person-only tensor parsing/OSD can be added after this batch baseline is
    proven stable without changing the inference architecture.
    """

    def __init__(self):
        if not ONNX_PATH.is_file():
            raise RuntimeError(
                f"batch-6 ONNX model not found: {ONNX_PATH}\n"
                "Run: python scripts/export_yolo26_batch6.py"
            )

        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.pipeline = Gst.Pipeline.new("deepstream-batch6-wall")
        if self.pipeline is None:
            raise RuntimeError("failed to create GStreamer pipeline")

        self.cameras = [
            dict(item)
            for item in camera_config().get("cameras", [])
            if item.get("online", True)
        ]
        if len(self.cameras) != BATCH_SIZE:
            raise RuntimeError(
                f"batch6 mode requires exactly {BATCH_SIZE} enabled cameras; "
                f"found {len(self.cameras)}"
            )

        self.stats = {
            str(camera["id"]): SourceStats(last_print=time.monotonic())
            for camera in self.cameras
        }
        self.queues = {}
        self._request_pads = []

        self.infer_buffers = 0
        self.last_infer_buffers = 0
        self.infer_frames = 0
        self.last_infer_frames = 0
        self.full_batches = 0
        self.partial_batches = 0
        self._last_infer_print = time.monotonic()

        try:
            import pyds
        except Exception:
            pyds = None
        self.pyds = pyds

        self.mux = self._make("nvstreammux", "mux")
        self.pgie = self._make("nvinfer", "primary_yolo26_batch6")
        self.tiler = self._make("nvmultistreamtiler", "tiler")
        self.convert = self._make("nvvideoconvert", "wall_convert")
        self.capsfilter = self._make("capsfilter", "wall_caps")
        self.sink_queue = self._make("queue", "wall_queue")
        self.sink = self._make("nveglglessink", "wall_sink")

        self._configure_mux()
        infer_config = self._write_nvinfer_config()
        self._infer_config_path = infer_config
        self.pgie.set_property("config-file-path", str(infer_config))
        self._set_if(self.pgie, "batch-size", BATCH_SIZE)
        self._set_if(self.pgie, "interval", 0)
        self._set_if(self.pgie, "gpu-id", 0)

        self._set_if(self.tiler, "rows", 2)
        self._set_if(self.tiler, "columns", 3)
        self._set_if(self.tiler, "width", 1920)
        self._set_if(self.tiler, "height", 720)
        self._set_if(self.tiler, "gpu-id", 0)

        self.capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        )
        self._set_if(self.sink_queue, "max-size-buffers", 2)
        self._set_if(self.sink_queue, "max-size-bytes", 0)
        self._set_if(self.sink_queue, "max-size-time", 0)
        self._set_if(self.sink_queue, "leaky", 2)
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "enable-last-sample", False)

        for element in (
            self.mux,
            self.pgie,
            self.tiler,
            self.convert,
            self.capsfilter,
            self.sink_queue,
            self.sink,
        ):
            self.pipeline.add(element)

        self._link(self.mux, self.pgie, "nvstreammux -> nvinfer")
        self._link(self.pgie, self.tiler, "nvinfer -> nvmultistreamtiler")
        self._link(self.tiler, self.convert, "tiler -> nvvideoconvert")
        self._link(self.convert, self.capsfilter, "convert -> RGBA NVMM caps")
        self._link(self.capsfilter, self.sink_queue, "caps -> sink queue")
        self._link(self.sink_queue, self.sink, "queue -> nveglglessink")

        pgie_src = self.pgie.get_static_pad("src")
        if pgie_src is None:
            raise RuntimeError("nvinfer has no src pad")
        pgie_src.add_probe(Gst.PadProbeType.BUFFER, self._infer_probe)

        for index, camera in enumerate(self.cameras):
            self._add_source(index, camera)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()
        GLib.timeout_add_seconds(5, self._print_stats)

    def _configure_mux(self) -> None:
        self._set_if(self.mux, "batch-size", BATCH_SIZE)
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", 1280)
        self._set_if(self.mux, "height", 720)
        # Slowest configured source is 20 FPS (50 ms/frame). Wait at most one
        # such frame period for a full six-camera batch; never wait forever.
        self._set_if(self.mux, "batched-push-timeout", 50000)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "buffer-pool-size", 8)

    def _write_nvinfer_config(self) -> Path:
        ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        config = f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file={ONNX_PATH}
model-engine-file={ENGINE_PATH}
batch-size={BATCH_SIZE}
network-mode=2
interval=0
gie-unique-id=1
process-mode=1
network-type=100
output-tensor-meta=1
maintain-aspect-ratio=1
symmetric-padding=1
workspace-size=1024
"""
        path = Path(tempfile.gettempdir()) / "ai_surveillance_yolo26_batch6_nvinfer.txt"
        path.write_text(config, encoding="utf-8")
        return path

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"required GStreamer/DeepStream element missing: {factory}")
        return element

    @staticmethod
    def _set_if(element, name: str, value) -> bool:
        if element.find_property(name) is None:
            return False
        element.set_property(name, value)
        return True

    @staticmethod
    def _link(a, b, label: str) -> None:
        if not a.link(b):
            raise RuntimeError(f"failed to link {label}")

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
            {**camera, "source": camera.get("display_source") or camera.get("source")}
        )
        if not isinstance(uri, str) or not uri.startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"{camera_id}: invalid RTSP source")

        source = self._make("nvurisrcbin", f"src_{index}")
        queue = self._make("queue", f"src_queue_{index}")

        source.set_property("uri", uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 0)
        self._set_if(source, "latency", 100)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", 6)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", 4 * 1024 * 1024)
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
        if qsrc is None or qsrc.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera_id}: failed queue -> streammux link")

        qsrc.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, camera_id)
        source.connect("pad-added", self._on_source_pad_added, queue, camera_id)
        self.queues[camera_id] = queue

    def _on_source_pad_added(self, _source, pad, queue, camera_id: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        media = caps.get_structure(0).get_name()
        if not str(media).startswith("video/"):
            return
        sinkpad = queue.get_static_pad("sink")
        if sinkpad is None or sinkpad.is_linked():
            return
        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"BATCH6 {camera_id} dynamic-link failed: {result}", flush=True)

    def _source_probe(self, _pad, info, camera_id: str):
        buffer = info.get_buffer()
        stat = self.stats[camera_id]
        stat.frames += 1
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts = int(buffer.pts)
            if stat.last_pts_ns is not None and pts > stat.last_pts_ns:
                interval_ms = (pts - stat.last_pts_ns) / 1_000_000.0
                if 0.0 < interval_ms < 2000.0:
                    stat.interval_ms_ema = (
                        interval_ms
                        if stat.interval_ms_ema is None
                        else stat.interval_ms_ema * 0.90 + interval_ms * 0.10
                    )
            stat.last_pts_ns = pts
        return self.Gst.PadProbeReturn.OK

    def _infer_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        self.infer_buffers += 1
        frames_in_batch = 0
        if self.pyds is not None:
            try:
                batch_meta = self.pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                if batch_meta is not None:
                    frames_in_batch = int(batch_meta.num_frames_in_batch)
            except Exception:
                frames_in_batch = 0

        if frames_in_batch > 0:
            self.infer_frames += frames_in_batch
            if frames_in_batch == BATCH_SIZE:
                self.full_batches += 1
            else:
                self.partial_batches += 1
        return self.Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"BATCH6 ERROR source={source} message={err.message} debug={debug or ''}",
                flush=True,
            )
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"BATCH6 WARNING source={source} message={err.message} debug={debug or ''}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.EOS:
            print("BATCH6 EOS", flush=True)
            self.loop.quit()

    def _print_stats(self) -> bool:
        now = time.monotonic()
        source_parts = []
        for camera in self.cameras:
            camera_id = str(camera["id"])
            stat = self.stats[camera_id]
            elapsed = max(0.001, now - stat.last_print)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_print = now
            queued = int(self.queues[camera_id].get_property("current-level-buffers"))
            source_parts.append(f"{camera_id}:{fps:.1f}fps q={queued}")

        infer_elapsed = max(0.001, now - self._last_infer_print)
        infer_rate = (self.infer_buffers - self.last_infer_buffers) / infer_elapsed
        frame_rate = (self.infer_frames - self.last_infer_frames) / infer_elapsed
        self.last_infer_buffers = self.infer_buffers
        self.last_infer_frames = self.infer_frames
        self._last_infer_print = now

        if self.pyds is not None:
            total_known = self.full_batches + self.partial_batches
            full_ratio = (100.0 * self.full_batches / total_known) if total_known else 0.0
            batch_text = (
                f"gpu_calls={infer_rate:.1f}/s infer_frames={frame_rate:.1f}/s "
                f"full6={self.full_batches} partial={self.partial_batches} "
                f"full_ratio={full_ratio:.1f}%"
            )
        else:
            batch_text = f"gpu_calls={infer_rate:.1f}/s batch_meta=pyds-unavailable"

        print("BATCH6 " + " | ".join(source_parts) + " || " + batch_text, flush=True)
        return True

    def run(self) -> int:
        print(f"BATCH6 model={ONNX_PATH}", flush=True)
        print(f"BATCH6 engine={ENGINE_PATH}", flush=True)
        print(
            "BATCH6 pipeline: 6x RTSP -> NVDEC -> nvstreammux(batch=6) -> "
            "nvinfer(batch=6, FP16) -> tiler -> EGL",
            flush=True,
        )
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("DeepStream batch6 pipeline failed to enter PLAYING")
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
            try:
                os.unlink(self._infer_config_path)
            except OSError:
                pass
        return 0


def run() -> int:
    return DeepStreamBatch6Wall().run()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"BATCH6 FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
