from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from shared.config import camera_config
from services.ml_service.cameras.gstreamer import (
    _gstreamer,
    authenticated_source,
    owned_bgr_from_mapped,
)

ROOT = Path(__file__).resolve().parents[3]
BATCH_SIZE = 6
INFER_WIDTH = 736
INFER_HEIGHT = 416
MODEL_PATH = ROOT / os.environ.get("AI_BATCH6_MODEL", "yolo26m.pt")
HEADLESS = os.environ.get("AI_BATCH6_HEADLESS", "0").strip().lower() in {"1", "true", "yes"}


@dataclass
class SourceStats:
    frames: int = 0
    last_frames: int = 0
    last_print: float = 0.0


class LatestBatchFrames:
    def __init__(self, camera_ids: list[str]):
        self.camera_ids = list(camera_ids)
        self._condition = threading.Condition()
        self._frames: dict[str, tuple[int, float, object] | None] = {
            cid: None for cid in self.camera_ids
        }
        self._versions = {cid: 0 for cid in self.camera_ids}
        self._closed = False

    def put(self, camera_id: str, captured_mono: float, frame) -> None:
        with self._condition:
            self._versions[camera_id] += 1
            self._frames[camera_id] = (
                self._versions[camera_id],
                float(captured_mono),
                frame,
            )
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def wait_full_new_batch(
        self,
        last_versions: dict[str, int],
        timeout: float = 0.5,
    ):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed:
                ready = all(
                    self._frames[cid] is not None
                    and self._frames[cid][0] > last_versions[cid]
                    for cid in self.camera_ids
                )
                if ready:
                    return [self._frames[cid] for cid in self.camera_ids]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None


class DeepStreamTorchBatch6Wall:
    """Single-decode GPU wall + one strict PyTorch CUDA batch of six frames.

    Display path:
      RTSP -> nvurisrcbin/NVDEC -> tee -> nvstreammux -> tiler -> EGL

    Inference side path (same decode session):
      tee -> leaky queue -> nvvideoconvert -> 736x416 BGRx appsink
          -> six latest frames -> ONE Ultralytics/PyTorch CUDA predict call

    TensorRT/nvinfer is intentionally not used because current TensorRT releases
    do not support Pascal SM 6.1 (GTX 1050 Ti).
    """

    def __init__(self):
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"YOLO model not found: {MODEL_PATH}")

        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.pipeline = Gst.Pipeline.new("deepstream-torch-batch6-wall")
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

        self.camera_ids = [str(camera["id"]) for camera in self.cameras]
        self.latest = LatestBatchFrames(self.camera_ids)
        self.stop_event = threading.Event()
        self.infer_thread: threading.Thread | None = None

        self.source_stats = {
            cid: SourceStats(last_print=time.monotonic()) for cid in self.camera_ids
        }
        self.display_queues = {}
        self.infer_queues = {}
        self._mux_request_pads = []
        self._tee_request_pads = []

        self.batch_calls = 0
        self.batch_inputs = 0
        self.total_detections = 0
        self.last_batch_ms = 0.0
        self.last_batch_age_ms = 0.0
        self.batch_errors = 0
        self._metrics_lock = threading.Lock()
        self._metrics_started = time.monotonic()

        self.mux = self._make("nvstreammux", "mux")
        self.tiler = self._make("nvmultistreamtiler", "tiler")
        self.wall_convert = self._make("nvvideoconvert", "wall_convert")
        self.wall_caps = self._make("capsfilter", "wall_caps")
        self.wall_queue = self._make("queue", "wall_queue")
        self.sink = self._make("fakesink" if HEADLESS else "nveglglessink", "wall_sink")

        self._set_if(self.mux, "batch-size", BATCH_SIZE)
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", 1280)
        self._set_if(self.mux, "height", 720)
        self._set_if(self.mux, "batched-push-timeout", 50000)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "buffer-pool-size", 8)

        self._set_if(self.tiler, "rows", 2)
        self._set_if(self.tiler, "columns", 3)
        self._set_if(self.tiler, "width", 1920)
        self._set_if(self.tiler, "height", 720)
        self._set_if(self.tiler, "gpu-id", 0)

        self.wall_caps.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(self.wall_queue, "max-size-buffers", 2)
        self._set_if(self.wall_queue, "max-size-bytes", 0)
        self._set_if(self.wall_queue, "max-size-time", 0)
        self._set_if(self.wall_queue, "leaky", 2)
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "enable-last-sample", False)

        for element in (
            self.mux,
            self.tiler,
            self.wall_convert,
            self.wall_caps,
            self.wall_queue,
            self.sink,
        ):
            self.pipeline.add(element)

        if not self.mux.link(self.tiler):
            raise RuntimeError("failed nvstreammux -> tiler")
        if not self.tiler.link(self.wall_convert):
            raise RuntimeError("failed tiler -> wall_convert")
        if not self.wall_convert.link(self.wall_caps):
            raise RuntimeError("failed wall_convert -> wall_caps")
        if not self.wall_caps.link(self.wall_queue):
            raise RuntimeError("failed wall_caps -> wall_queue")
        if not self.wall_queue.link(self.sink):
            raise RuntimeError("failed wall_queue -> sink")

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

    def _request_pad(self, element, name: str):
        request_simple = getattr(element, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            pad = element.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"{element.get_name()} could not allocate {name}")
        return pad

    def _link_tee_to_queue(self, tee, queue, camera_id: str, branch: str):
        tee_pad = self._request_pad(tee, "src_%u")
        sink_pad = queue.get_static_pad("sink")
        if tee_pad.link(sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera_id}: tee -> {branch} queue failed")
        self._tee_request_pads.append((tee, tee_pad))

    def _add_source(self, index: int, camera: dict) -> None:
        cid = str(camera["id"])
        uri = authenticated_source(
            {**camera, "source": camera.get("display_source") or camera.get("source")}
        )
        if not isinstance(uri, str) or not uri.startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"{cid}: invalid RTSP source")

        source = self._make("nvurisrcbin", f"src_{index}")
        tee = self._make("tee", f"tee_{index}")
        display_q = self._make("queue", f"display_q_{index}")
        infer_q = self._make("queue", f"infer_q_{index}")
        infer_convert = self._make("nvvideoconvert", f"infer_convert_{index}")
        infer_caps = self._make("capsfilter", f"infer_caps_{index}")
        appsink = self._make("appsink", f"infer_sink_{index}")

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

        for q in (display_q, infer_q):
            self._set_if(q, "max-size-buffers", 1)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)

        infer_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,width={INFER_WIDTH},height={INFER_HEIGHT},format=BGRx"
            ),
        )
        self._set_if(appsink, "drop", True)
        self._set_if(appsink, "max-buffers", 1)
        self._set_if(appsink, "sync", False)
        self._set_if(appsink, "emit-signals", True)
        self._set_if(appsink, "wait-on-eos", False)
        self._set_if(appsink, "enable-last-sample", False)

        for element in (
            source,
            tee,
            display_q,
            infer_q,
            infer_convert,
            infer_caps,
            appsink,
        ):
            self.pipeline.add(element)

        self._link_tee_to_queue(tee, display_q, cid, "display")
        self._link_tee_to_queue(tee, infer_q, cid, "infer")

        mux_pad = self._request_pad(self.mux, f"sink_{index}")
        self._mux_request_pads.append(mux_pad)
        if display_q.get_static_pad("src").link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display queue -> streammux failed")

        if not infer_q.link(infer_convert):
            raise RuntimeError(f"{cid}: infer_q -> infer_convert failed")
        if not infer_convert.link(infer_caps):
            raise RuntimeError(f"{cid}: infer_convert -> infer_caps failed")
        if not infer_caps.link(appsink):
            raise RuntimeError(f"{cid}: infer_caps -> appsink failed")

        display_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._display_probe,
            cid,
        )
        appsink.connect("new-sample", self._on_new_sample, cid)
        source.connect("pad-added", self._on_source_pad_added, tee, cid)

        self.display_queues[cid] = display_q
        self.infer_queues[cid] = infer_q

    def _on_source_pad_added(self, _source, pad, tee, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        media = caps.get_structure(0).get_name()
        if not str(media).startswith("video/"):
            return
        sinkpad = tee.get_static_pad("sink")
        if sinkpad.is_linked():
            return
        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"TORCH_BATCH6 {cid} source -> tee failed: {result}", flush=True)

    def _display_probe(self, _pad, info, cid: str):
        if info.get_buffer() is not None:
            self.source_stats[cid].frames += 1
        return self.Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK

        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        fmt = str(caps.get_value("format"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.OK
        try:
            frame = owned_bgr_from_mapped(mapped.data, width, height, fmt)
        finally:
            buffer.unmap(mapped)

        self.latest.put(cid, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _infer_loop(self) -> None:
        try:
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is not available")
            capability = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)
            torch.cuda.set_device(0)
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True

            print(
                f"TORCH_BATCH6 cuda={torch.version.cuda} device={device_name} "
                f"sm={capability[0]}.{capability[1]} model={MODEL_PATH}",
                flush=True,
            )

            model = YOLO(str(MODEL_PATH))
            import numpy as np
            warm = [
                np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8)
                for _ in range(BATCH_SIZE)
            ]
            model.predict(
                source=warm,
                imgsz=(INFER_HEIGHT, INFER_WIDTH),
                classes=[0],
                conf=0.08,
                iou=0.50,
                max_det=30,
                device="cuda:0",
                half=False,
                verbose=False,
            )
            print("TORCH_BATCH6 warmup complete: strict batch=6", flush=True)
        except BaseException as exc:
            print(
                f"TORCH_BATCH6 startup error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self.stop_event.set()
            self.loop.quit()
            return

        last_versions = {cid: 0 for cid in self.camera_ids}
        while not self.stop_event.is_set():
            rows = self.latest.wait_full_new_batch(last_versions, timeout=0.5)
            if rows is None:
                continue

            versions = {}
            frames = []
            captured = []
            for cid, row in zip(self.camera_ids, rows):
                version, captured_mono, frame = row
                versions[cid] = int(version)
                frames.append(frame)
                captured.append(float(captured_mono))

            started = time.perf_counter()
            try:
                predictions = model.predict(
                    source=frames,
                    imgsz=(INFER_HEIGHT, INFER_WIDTH),
                    classes=[0],
                    conf=0.08,
                    iou=0.50,
                    max_det=30,
                    device="cuda:0",
                    half=False,
                    verbose=False,
                )
                detections = 0
                for prediction in predictions:
                    boxes = getattr(prediction, "boxes", None)
                    if boxes is not None:
                        detections += len(boxes)
            except BaseException as exc:
                with self._metrics_lock:
                    self.batch_errors += 1
                print(
                    f"TORCH_BATCH6 batch error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if "out of memory" in str(exc).lower():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            ended = time.monotonic()
            batch_ms = (time.perf_counter() - started) * 1000.0
            age_ms = max(0.0, (ended - min(captured)) * 1000.0)
            last_versions.update(versions)

            with self._metrics_lock:
                self.batch_calls += 1
                self.batch_inputs += BATCH_SIZE
                self.total_detections += int(detections)
                self.last_batch_ms = batch_ms
                self.last_batch_age_ms = age_ms

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"TORCH_BATCH6 ERROR source={source} message={err.message} "
                f"debug={debug or ''}",
                file=sys.stderr,
                flush=True,
            )
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"TORCH_BATCH6 WARNING source={source} message={err.message} "
                f"debug={debug or ''}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.EOS:
            self.loop.quit()

    def _print_stats(self) -> bool:
        now = time.monotonic()
        source_parts = []
        for cid in self.camera_ids:
            stat = self.source_stats[cid]
            elapsed = max(0.001, now - stat.last_print)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_print = now
            dq = int(self.display_queues[cid].get_property("current-level-buffers"))
            iq = int(self.infer_queues[cid].get_property("current-level-buffers"))
            source_parts.append(f"{cid}:{fps:.1f}fps dq={dq} iq={iq}")

        with self._metrics_lock:
            elapsed = max(0.001, now - self._metrics_started)
            calls = self.batch_calls
            inputs = self.batch_inputs
            detections = self.total_detections
            batch_ms = self.last_batch_ms
            age_ms = self.last_batch_age_ms
            errors = self.batch_errors

        print(
            "TORCH_BATCH6 "
            + " | ".join(source_parts)
            + f" || gpu_batches={calls/elapsed:.2f}/s "
            + f"inputs={inputs/elapsed:.1f}/s "
            + f"last_batch={batch_ms:.1f}ms "
            + f"age={age_ms:.1f}ms det={detections} errors={errors}",
            flush=True,
        )
        return True

    def run(self) -> int:
        self.infer_thread = threading.Thread(
            target=self._infer_loop,
            name="torch-batch6-infer",
            daemon=True,
        )
        self.infer_thread.start()

        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.stop_event.set()
            self.latest.close()
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("DeepStream Torch batch6 pipeline failed to PLAY")

        print(
            "TORCH_BATCH6 started: one NVDEC session per camera; GPU-native wall; "
            "strict 6-frame PyTorch CUDA inference batch; TensorRT disabled",
            flush=True,
        )
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self.latest.close()
            self.pipeline.set_state(self.Gst.State.NULL)
            if self.infer_thread:
                self.infer_thread.join(3.0)
            for pad in self._mux_request_pads:
                try:
                    self.mux.release_request_pad(pad)
                except Exception:
                    pass
            for tee, pad in self._tee_request_pads:
                try:
                    tee.release_request_pad(pad)
                except Exception:
                    pass
        return 0


def run() -> int:
    return DeepStreamTorchBatch6Wall().run()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"TORCH_BATCH6 FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
