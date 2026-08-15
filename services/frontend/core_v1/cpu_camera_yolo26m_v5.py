from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from shared.config import camera_config
from services.ml_service.cameras.gstreamer import _gstreamer, authenticated_source

ROOT = Path(__file__).resolve().parents[3]
MODEL_SPEC = os.environ.get("AI_YOLO_MODEL", "yolo26m.pt")
INFER_WIDTH = max(320, int(os.environ.get("AI_YOLO_INFER_WIDTH", "448")))
INFER_HEIGHT = max(192, int(os.environ.get("AI_YOLO_INFER_HEIGHT", "256")))
BATCH_SIZE = 6


@dataclass
class CameraStat:
    frames: int = 0
    last_frames: int = 0
    last_time: float = 0.0
    fps: float = 0.0


class LatestFrames:
    def __init__(self):
        self.condition = threading.Condition()
        self.rows: dict[str, tuple[int, float, object]] = {}
        self.versions: dict[str, int] = {}
        self.closed = False

    def put(self, cid: str, captured: float, frame) -> None:
        with self.condition:
            version = self.versions.get(cid, 0) + 1
            self.versions[cid] = version
            self.rows[cid] = (version, captured, frame)
            self.condition.notify_all()

    def wait_six(self, ids: list[str], previous: dict[str, int], timeout: float):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.closed:
                if all(
                    cid in self.rows and self.rows[cid][0] > previous.get(cid, 0)
                    for cid in ids
                ):
                    return [self.rows[cid] for cid in ids]
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return None
                self.condition.wait(remain)
        return None

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def _resolve_model_spec(spec: str) -> str:
    candidate = Path(spec)
    if candidate.is_file():
        return str(candidate)
    rooted = ROOT / spec
    if rooted.is_file():
        return str(rooted)
    return spec


def _yolo_worker(job_q, result_q, model_spec: str, infer_w: int, infer_h: int, conf: float):
    """Separate spawned process: all PyTorch/Ultralytics work lives here."""
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(0)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        model_path = _resolve_model_spec(model_spec)
        device = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        model = YOLO(model_path)
        kwargs = {
            "imgsz": (infer_h, infer_w),
            "rect": True,
            "classes": [0],
            "conf": float(conf),
            "iou": 0.50,
            "max_det": 20,
            "device": "cuda:0",
            "verbose": False,
            "stream": False,
        }
        warm = [np.zeros((infer_h, infer_w, 3), dtype=np.uint8) for _ in range(BATCH_SIZE)]
        with torch.inference_mode():
            model.predict(source=warm, **kwargs)

        result_q.put({
            "type": "ready",
            "device": device,
            "sm": f"{capability[0]}.{capability[1]}",
            "model": model_path,
        })

        while True:
            job = job_q.get()
            if job is None:
                return
            seq = int(job["seq"])
            frames = job["frames"]
            captured = job["captured"]
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    preds = model.predict(source=frames, **kwargs)
                ended = time.monotonic()
                snapshots = {}
                counts = {}
                for cid, frame, pred, captured_mono in zip(
                    job["camera_ids"], frames, preds, captured
                ):
                    items = []
                    boxes = getattr(pred, "boxes", None)
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().tolist()
                        confs = boxes.conf.detach().cpu().tolist()
                        for coords, confidence in zip(xyxy, confs):
                            items.append({
                                "xyxy": [float(v) for v in coords],
                                "confidence": float(confidence),
                            })
                    counts[cid] = len(items)
                    snapshots[cid] = {
                        "captured_mono": float(captured_mono),
                        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                        "boxes": items,
                    }
                result_q.put({
                    "type": "result",
                    "seq": seq,
                    "batch_ms": (ended - started) * 1000.0,
                    "spread_ms": (max(captured) - min(captured)) * 1000.0,
                    "snapshots": snapshots,
                    "counts": counts,
                })
            except BaseException as exc:
                result_q.put({
                    "type": "error",
                    "seq": seq,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    except BaseException as exc:
        try:
            result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


class CpuCameraYolo26mV5:
    """Smoothness-first fallback for systems where no Intel DRM render node exists.

    One RTSP/decode session per camera. Camera decode/composition/display use
    native GStreamer + libav on CPU. NVIDIA is touched only by the spawned
    YOLO26m process. Heavy Python inference never shares the display process/GIL.
    """

    def __init__(self):
        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.pipeline = Gst.Pipeline.new("cpu-camera-yolo26m-v5")
        if self.pipeline is None:
            raise RuntimeError("failed to create CPU camera pipeline")

        self.cameras = [
            dict(item)
            for item in camera_config().get("cameras", [])
            if item.get("online", True)
        ]
        if len(self.cameras) != BATCH_SIZE:
            raise RuntimeError(f"strict batch=6 requires six cameras; found {len(self.cameras)}")
        self.camera_ids = [str(c["id"]) for c in self.cameras]

        self.wall_width = max(960, int(os.environ.get("AI_CPU_WALL_WIDTH", "1920")))
        self.wall_height = max(360, int(os.environ.get("AI_CPU_WALL_HEIGHT", "720")))
        self.tile_width = self.wall_width // 3
        self.tile_height = self.wall_height // 2
        self.rtsp_latency_ms = max(60, int(os.environ.get("AI_CPU_RTSP_LATENCY_MS", "100")))
        self.decoder_threads = max(1, min(2, int(os.environ.get("AI_CPU_DECODER_THREADS", "1"))))
        self.batch_fps = max(0.25, float(os.environ.get("AI_YOLO_BATCH_FPS", "0.70")))
        self.capture_timeout = max(0.10, float(os.environ.get("AI_YOLO_CAPTURE_TIMEOUT", "0.40")))
        self.conf = min(0.9, max(0.05, float(os.environ.get("AI_YOLO_CONF", "0.16"))))
        self.box_hold_ms = max(800.0, float(os.environ.get("AI_DETECTION_BOX_HOLD_MS", "3000")))

        self.stop_event = threading.Event()
        self.latest = LatestFrames()
        self.ticket_lock = threading.Lock()
        self.tickets = {cid: False for cid in self.camera_ids}
        self.det_lock = threading.Lock()
        self.latest_detections: dict[str, dict] = {}
        self.last_counts = {cid: 0 for cid in self.camera_ids}
        self.detector_ready = False
        self.detector_error = ""
        self.batch_calls = 0
        self.batch_errors = 0
        self.capture_timeouts = 0
        self.last_batch_ms = 0.0
        self.last_spread_ms = 0.0
        self.detector_started = 0.0
        self.detector_thread: threading.Thread | None = None

        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.worker = ctx.Process(
            target=_yolo_worker,
            args=(self.job_q, self.result_q, MODEL_SPEC, INFER_WIDTH, INFER_HEIGHT, self.conf),
            name="yolo26m-cuda-worker",
            daemon=True,
        )

        now = time.monotonic()
        self.stats = {cid: CameraStat(last_time=now) for cid in self.camera_ids}
        self.display_queues = {}
        self.decoders = {}
        self._tee_pads = []
        self._comp_pads = []
        self.wall_frames = 0
        self.wall_last_frames = 0
        self.wall_last_time = now

        self.compositor = self._make("compositor", "cpu_compositor")
        self._set_if(self.compositor, "ignore-inactive-pads", True)
        self._set_if(self.compositor, "background", 1)
        self._set_if(self.compositor, "latency", 50_000_000)
        self.pipeline.add(self.compositor)

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        self.wall_convert = self._make("videoconvert", "wall_convert")
        self.wall_scale = self._make("videoscale", "wall_scale")
        self.wall_caps = self._make("capsfilter", "wall_caps")
        self.overlay = self._make("cairooverlay", "bbox_overlay")
        self.wall_queue = self._make("queue", "wall_queue")
        self.sink = self._choose_sink()

        self.wall_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={self.wall_width},height={self.wall_height},framerate=20/1"
            ),
        )
        self._set_if(self.wall_scale, "method", 1)
        self._set_if(self.wall_queue, "max-size-buffers", 1)
        self._set_if(self.wall_queue, "max-size-bytes", 0)
        self._set_if(self.wall_queue, "max-size-time", 0)
        self._set_if(self.wall_queue, "leaky", 2)

        # No clock-based late-frame policy at the final renderer. Latest frame wins.
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "async", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "enable-last-sample", False)
        self._set_if(self.sink, "force-aspect-ratio", True)

        for element in (
            self.wall_convert, self.wall_scale, self.wall_caps,
            self.overlay, self.wall_queue, self.sink,
        ):
            self.pipeline.add(element)

        if not self.compositor.link(self.wall_convert):
            raise RuntimeError("compositor -> videoconvert failed")
        if not self.wall_convert.link(self.wall_scale):
            raise RuntimeError("videoconvert -> videoscale failed")
        if not self.wall_scale.link(self.wall_caps):
            raise RuntimeError("videoscale -> wall caps failed")
        if not self.wall_caps.link(self.overlay):
            raise RuntimeError("wall caps -> cairooverlay failed")
        if not self.overlay.link(self.wall_queue):
            raise RuntimeError("cairooverlay -> wall queue failed")
        if not self.wall_queue.link(self.sink):
            raise RuntimeError("wall queue -> video sink failed")

        self.overlay.connect("draw", self._draw_overlay)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()
        GLib.timeout_add_seconds(5, self._print_stats)

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"missing GStreamer element: {factory}")
        return element

    @staticmethod
    def _set_if(element, name: str, value) -> bool:
        if element.find_property(name) is None:
            return False
        element.set_property(name, value)
        return True

    def _choose_sink(self):
        # Explicitly avoid GL/EGL sinks: they would put visible rendering back on NVIDIA.
        if self.Gst.ElementFactory.find("ximagesink") is not None and os.environ.get("DISPLAY"):
            return self._make("ximagesink", "cpu_wall_sink")
        if self.Gst.ElementFactory.find("waylandsink") is not None:
            return self._make("waylandsink", "cpu_wall_sink")
        raise RuntimeError("need ximagesink or waylandsink; refusing GL/EGL fallback")

    def _request_tee_pad(self, tee):
        request = getattr(tee, "request_pad_simple", None)
        pad = request("src_%u") if request else None
        if pad is None:
            pad = tee.get_request_pad("src_%u")
        if pad is None:
            raise RuntimeError("tee request pad failed")
        self._tee_pads.append((tee, pad))
        return pad

    def _link_tee(self, tee, queue_element, cid: str, branch: str) -> None:
        src = self._request_tee_pad(tee)
        dst = queue_element.get_static_pad("sink")
        if src.link(dst) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> {branch} failed")

    def _request_comp_pad(self, index: int):
        request = getattr(self.compositor, "request_pad_simple", None)
        pad = request("sink_%u") if request else None
        if pad is None:
            pad = self.compositor.get_request_pad("sink_%u")
        if pad is None:
            raise RuntimeError("compositor request pad failed")
        row, col = divmod(index, 3)
        for key, value in (
            ("xpos", col * self.tile_width),
            ("ypos", row * self.tile_height),
            ("width", self.tile_width),
            ("height", self.tile_height),
        ):
            if pad.find_property(key) is not None:
                pad.set_property(key, value)
        self._comp_pads.append(pad)
        return pad

    def _add_camera(self, index: int, camera: dict) -> None:
        cid = str(camera["id"])
        codec = str(camera.get("display_codec") or camera.get("codec") or "h264").lower()
        if codec not in {"h264", "h265"}:
            raise RuntimeError(f"{cid}: unsupported codec={codec}")

        uri = authenticated_source(
            {**camera, "source": camera.get("display_source") or camera.get("source")}
        )
        if not isinstance(uri, str) or not uri.startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"{cid}: invalid RTSP source")

        source = self._make("rtspsrc", f"rtsp_{index}")
        depay = self._make(f"rtp{codec}depay", f"depay_{index}")
        parser = self._make(f"{codec}parse", f"parse_{index}")
        decoder = self._make(f"avdec_{codec}", f"cpu_decode_{index}")
        tee = self._make("tee", f"decoded_tee_{index}")
        display_q = self._make("queue", f"display_q_{index}")
        infer_q = self._make("queue", f"infer_q_{index}")
        infer_scale = self._make("videoscale", f"infer_scale_{index}")
        infer_convert = self._make("videoconvert", f"infer_convert_{index}")
        infer_caps = self._make("capsfilter", f"infer_caps_{index}")
        appsink = self._make("appsink", f"infer_sink_{index}")

        source.set_property("location", uri)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "tcp-timeout", 5_000_000)

        self._set_if(decoder, "max-threads", self.decoder_threads)
        self._set_if(decoder, "direct-rendering", True)
        self._set_if(decoder, "output-corrupt", False)

        for q, buffers in ((display_q, 2), (infer_q, 1)):
            self._set_if(q, "max-size-buffers", buffers)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)

        self._set_if(infer_scale, "method", 1)
        infer_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INFER_WIDTH},height={INFER_HEIGHT}"
            ),
        )
        self._set_if(appsink, "drop", True)
        self._set_if(appsink, "max-buffers", 1)
        self._set_if(appsink, "sync", False)
        self._set_if(appsink, "emit-signals", True)
        self._set_if(appsink, "wait-on-eos", False)
        self._set_if(appsink, "enable-last-sample", False)

        for element in (
            source, depay, parser, decoder, tee, display_q,
            infer_q, infer_scale, infer_convert, infer_caps, appsink,
        ):
            self.pipeline.add(element)

        if not depay.link(parser):
            raise RuntimeError(f"{cid}: depay -> parser failed")
        if not parser.link(decoder):
            raise RuntimeError(f"{cid}: parser -> avdec failed")
        if not decoder.link(tee):
            raise RuntimeError(f"{cid}: avdec -> tee failed")

        self._link_tee(tee, display_q, cid, "display")
        self._link_tee(tee, infer_q, cid, "infer")

        comp_pad = self._request_comp_pad(index)
        display_src = display_q.get_static_pad("src")
        if display_src.link(comp_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display -> compositor failed")
        display_src.add_probe(self.Gst.PadProbeType.BUFFER, self._display_probe, cid)

        infer_src = infer_q.get_static_pad("src")
        infer_src.add_probe(self.Gst.PadProbeType.BUFFER, self._ticket_probe, cid)
        if not infer_q.link(infer_scale):
            raise RuntimeError(f"{cid}: infer queue -> videoscale failed")
        if not infer_scale.link(infer_convert):
            raise RuntimeError(f"{cid}: videoscale -> videoconvert failed")
        if not infer_convert.link(infer_caps):
            raise RuntimeError(f"{cid}: videoconvert -> infer caps failed")
        if not infer_caps.link(appsink):
            raise RuntimeError(f"{cid}: infer caps -> appsink failed")
        appsink.connect("new-sample", self._on_infer_sample, cid)

        source.connect("pad-added", self._on_rtsp_pad, depay, codec, cid)
        self.display_queues[cid] = display_q
        self.decoders[cid] = decoder

    def _on_rtsp_pad(self, _source, pad, depay, codec: str, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        s = caps.get_structure(0)
        if str(s.get_name()) != "application/x-rtp":
            return
        media = str(s.get_string("media") or "").lower()
        encoding = str(s.get_string("encoding-name") or "").lower()
        codec_ok = (codec == "h264" and "h264" in encoding) or (
            codec == "h265" and ("h265" in encoding or "hevc" in encoding)
        )
        if media != "video" or not codec_ok:
            return
        sinkpad = depay.get_static_pad("sink")
        if sinkpad.is_linked():
            return
        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"CPU_V5 {cid} RTSP -> depay failed: {result}", flush=True)

    def _display_probe(self, _pad, _info, cid: str):
        self.stats[cid].frames += 1
        return self.Gst.PadProbeReturn.OK

    def _ticket_probe(self, _pad, _info, cid: str):
        with self.ticket_lock:
            if not self.tickets.get(cid, False):
                return self.Gst.PadProbeReturn.DROP
            self.tickets[cid] = False
        return self.Gst.PadProbeReturn.OK

    def _request_six(self) -> None:
        with self.ticket_lock:
            for cid in self.camera_ids:
                self.tickets[cid] = True

    def _clear_tickets(self) -> None:
        with self.ticket_lock:
            for cid in self.camera_ids:
                self.tickets[cid] = False

    @staticmethod
    def _bgrx_to_bgr(data, width: int, height: int):
        import numpy as np
        raw = np.frombuffer(data, dtype=np.uint8)
        stride = raw.size // max(1, int(height))
        minimum = int(width) * 4
        if stride < minimum:
            raise ValueError(f"BGRx stride {stride} < {minimum}")
        rows = raw[: stride * int(height)].reshape((int(height), stride))
        pixels = rows[:, :minimum].reshape((int(height), int(width), 4))
        return pixels[..., :3].copy()

    def _on_infer_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            with self.ticket_lock:
                self.tickets[cid] = True
            return self.Gst.FlowReturn.OK
        s = sample.get_caps().get_structure(0)
        width = int(s.get_value("width"))
        height = int(s.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            with self.ticket_lock:
                self.tickets[cid] = True
            return self.Gst.FlowReturn.OK
        try:
            frame = self._bgrx_to_bgr(mapped.data, width, height)
        except Exception:
            with self.ticket_lock:
                self.tickets[cid] = True
            return self.Gst.FlowReturn.OK
        finally:
            buffer.unmap(mapped)
        self.latest.put(cid, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _detector_coordinator(self) -> None:
        # Wait for worker readiness while camera pipeline is already free to run.
        ready_deadline = time.monotonic() + 15.0
        while not self.stop_event.is_set() and time.monotonic() < ready_deadline:
            try:
                msg = self.result_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if msg.get("type") == "ready":
                with self.det_lock:
                    self.detector_ready = True
                    self.detector_started = time.monotonic()
                print(
                    "YOLO26M_V5 ready "
                    f"model={msg.get('model')} device={msg.get('device')} sm={msg.get('sm')} "
                    f"strict_batch=6 input={INFER_WIDTH}x{INFER_HEIGHT} process_isolated=1",
                    flush=True,
                )
                break
            if msg.get("type") == "fatal":
                with self.det_lock:
                    self.detector_error = str(msg.get("error") or "worker fatal")
                return
        else:
            with self.det_lock:
                self.detector_error = "YOLO worker readiness timeout"
            return

        previous = {cid: 0 for cid in self.camera_ids}
        interval = 1.0 / self.batch_fps
        seq = 0
        next_at = time.monotonic()

        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < next_at:
                if self.stop_event.wait(min(0.05, next_at - now)):
                    break
                continue

            self._request_six()
            rows = self.latest.wait_six(self.camera_ids, previous, self.capture_timeout)
            self._clear_tickets()
            if rows is None:
                with self.det_lock:
                    self.capture_timeouts += 1
                next_at = time.monotonic() + 0.10
                continue

            frames = []
            captured = []
            for cid, (version, captured_mono, frame) in zip(self.camera_ids, rows):
                previous[cid] = int(version)
                frames.append(frame)
                captured.append(float(captured_mono))

            seq += 1
            sent = time.monotonic()
            try:
                self.job_q.put({
                    "seq": seq,
                    "camera_ids": self.camera_ids,
                    "frames": frames,
                    "captured": captured,
                }, timeout=0.2)
            except queue.Full:
                next_at = time.monotonic() + 0.10
                continue

            try:
                msg = self.result_q.get(timeout=max(2.0, interval * 2.0))
            except queue.Empty:
                with self.det_lock:
                    self.batch_errors += 1
                    self.detector_error = "YOLO result timeout"
                next_at = time.monotonic() + 0.25
                continue

            kind = msg.get("type")
            if kind == "result":
                with self.det_lock:
                    self.batch_calls += 1
                    self.last_batch_ms = float(msg.get("batch_ms") or 0.0)
                    self.last_spread_ms = float(msg.get("spread_ms") or 0.0)
                    self.last_counts = dict(msg.get("counts") or {})
                    self.latest_detections = dict(msg.get("snapshots") or {})
                    self.detector_error = ""
            elif kind in {"error", "fatal"}:
                with self.det_lock:
                    self.batch_errors += 1
                    self.detector_error = str(msg.get("error") or kind)
                if kind == "fatal":
                    return

            next_at = max(time.monotonic(), sent + interval)

    def _draw_overlay(self, _overlay, cr, _timestamp, _duration) -> None:
        self.wall_frames += 1
        now = time.monotonic()
        with self.det_lock:
            snapshots = {
                cid: dict(value)
                for cid, value in self.latest_detections.items()
                if value
            }

        for index, cid in enumerate(self.camera_ids):
            snapshot = snapshots.get(cid)
            if not snapshot:
                continue
            captured = float(snapshot.get("captured_mono") or 0.0)
            if captured <= 0.0 or (now - captured) * 1000.0 > self.box_hold_ms:
                continue
            fw, fh = snapshot.get("frame_size") or [INFER_WIDTH, INFER_HEIGHT]
            fw = max(1.0, float(fw))
            fh = max(1.0, float(fh))
            row, col = divmod(index, 3)
            ox = float(col * self.tile_width)
            oy = float(row * self.tile_height)
            sx = self.tile_width / fw
            sy = self.tile_height / fh
            for item in snapshot.get("boxes") or []:
                x1, y1, x2, y2 = [float(v) for v in item.get("xyxy", [0, 0, 1, 1])]
                x1 = max(0.0, min(fw, x1))
                y1 = max(0.0, min(fh, y1))
                x2 = max(x1 + 1.0, min(fw, x2))
                y2 = max(y1 + 1.0, min(fh, y2))
                cr.set_source_rgba(0.0, 1.0, 0.10, 1.0)
                cr.set_line_width(3.0)
                cr.rectangle(
                    ox + x1 * sx,
                    oy + y1 * sy,
                    (x2 - x1) * sx,
                    (y2 - y1) * sy,
                )
                cr.stroke()

    def _sink_stats(self) -> tuple[int, int]:
        try:
            stats = self.sink.get_property("stats")
            rendered = int(stats.get_value("rendered")) if stats.has_field("rendered") else -1
            dropped = int(stats.get_value("dropped")) if stats.has_field("dropped") else -1
            return rendered, dropped
        except Exception:
            return -1, -1

    def _print_stats(self) -> bool:
        now = time.monotonic()
        parts = []
        fps_values = []
        for cid in self.camera_ids:
            stat = self.stats[cid]
            elapsed = max(0.001, now - stat.last_time)
            stat.fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_time = now
            fps_values.append(stat.fps)
            qlevel = int(self.display_queues[cid].get_property("current-level-buffers"))
            parts.append(f"{cid}:{stat.fps:.1f}fps q={qlevel}")

        wall_elapsed = max(0.001, now - self.wall_last_time)
        wall_fps = (self.wall_frames - self.wall_last_frames) / wall_elapsed
        self.wall_last_frames = self.wall_frames
        self.wall_last_time = now
        rendered, dropped = self._sink_stats()

        with self.det_lock:
            ready = self.detector_ready
            error = self.detector_error
            calls = self.batch_calls
            batch_ms = self.last_batch_ms
            spread_ms = self.last_spread_ms
            errors = self.batch_errors
            timeouts = self.capture_timeouts
            counts = dict(self.last_counts)
            started = self.detector_started
        elapsed_det = max(0.001, now - started) if started else 1.0
        rate = calls / elapsed_det if ready else 0.0
        count_text = " ".join(f"{cid}:{counts.get(cid, 0)}" for cid in self.camera_ids)

        print("CPU_V5 " + " | ".join(parts), flush=True)
        print(
            f"CPU_V5 wall_fps={wall_fps:.1f} sink={self.sink.get_factory().get_name()} "
            f"rendered={rendered} dropped={dropped} decoder_threads={self.decoder_threads} "
            f"wall={self.wall_width}x{self.wall_height}",
            flush=True,
        )
        print(
            f"YOLO26M_V5 ready={int(ready)} batches={rate:.2f}/s batch={batch_ms:.1f}ms "
            f"spread={spread_ms:.1f}ms timeouts={timeouts} errors={errors} "
            f"persons=[{count_text}] process_isolated=1"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        return True

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src else "unknown"
            print(
                f"CPU_V5 ERROR source={src} message={err.message} debug={debug or ''}",
                file=sys.stderr,
                flush=True,
            )
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            src = message.src.get_name() if message.src else "unknown"
            print(
                f"CPU_V5 WARNING source={src} message={err.message} debug={debug or ''}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.EOS:
            self.loop.quit()

    def run(self) -> int:
        self.worker.start()

        state = self.pipeline.set_state(self.Gst.State.PLAYING)
        if state == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            try:
                self.job_q.put_nowait(None)
            except Exception:
                pass
            self.worker.join(timeout=2.0)
            raise RuntimeError("CPU V5 camera pipeline failed to PLAY")

        self.detector_thread = threading.Thread(
            target=self._detector_coordinator,
            name="yolo26m-batch-coordinator",
            daemon=True,
        )
        self.detector_thread.start()

        print(
            "CPU_V5 started: camera=CPU-libav/compositor display=no-GL/no-EGL; "
            f"NVIDIA=YOLO26m-only strict_batch=6 process_isolated=1 "
            f"infer={INFER_WIDTH}x{INFER_HEIGHT} batch_cap={self.batch_fps:.2f}/s "
            f"wall={self.wall_width}x{self.wall_height} sink={self.sink.get_factory().get_name()}",
            flush=True,
        )
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self._clear_tickets()
            self.latest.close()
            self.pipeline.set_state(self.Gst.State.NULL)
            try:
                self.job_q.put_nowait(None)
            except Exception:
                pass
            if self.detector_thread is not None:
                self.detector_thread.join(timeout=2.0)
            self.worker.join(timeout=3.0)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=1.0)
        return 0


def missing_plugins() -> list[str]:
    Gst = _gstreamer()
    required = [
        "rtspsrc", "rtph264depay", "rtph265depay", "h264parse", "h265parse",
        "avdec_h264", "avdec_h265", "tee", "queue", "compositor", "videoscale",
        "videoconvert", "appsink", "cairooverlay",
    ]
    if not (
        (Gst.ElementFactory.find("ximagesink") is not None and os.environ.get("DISPLAY"))
        or Gst.ElementFactory.find("waylandsink") is not None
    ):
        required.append("ximagesink-or-waylandsink")
    return [name for name in required if name == "ximagesink-or-waylandsink" or Gst.ElementFactory.find(name) is None]


def run() -> int:
    missing = missing_plugins()
    if missing:
        raise RuntimeError(
            "CPU_V5 missing GStreamer plugins: " + ", ".join(missing)
            + ". On Ubuntu install gstreamer1.0-libav gstreamer1.0-plugins-good "
              "gstreamer1.0-plugins-base gstreamer1.0-x."
        )
    return CpuCameraYolo26mV5().run()


if __name__ == "__main__":
    raise SystemExit(run())
