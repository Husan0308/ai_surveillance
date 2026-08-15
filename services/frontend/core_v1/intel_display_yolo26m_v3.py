from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# IMPORTANT: visible video never touches NVIDIA/GL/EGL in this runtime.
# Intel Ice Lake VA-API handles RTSP decode, scale and 3x2 composition.
# NVIDIA GTX 1050 Ti is reserved for YOLO26m CUDA only.
os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
os.environ.pop("__NV_PRIME_RENDER_OFFLOAD", None)
os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)

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


class IntelDisplayYolo26mV3:
    """One RTSP/decode session per camera; Intel display + NVIDIA YOLO26m.

    Per camera:
      RTSP -> depay -> parse -> Intel VA decoder -> tee
        display:   latest queue -> Intel VA compositor
        inference: ticket gate -> Intel VA resize/download -> 448x256 BGRA appsink

    Wall:
      VA compositor -> one BGRA download -> cairooverlay -> ximagesink/waylandsink

    Detector:
      exactly six fresh BGR frames -> one YOLO26m CUDA model.predict call

    No DeepStream tiler/EGL, no NVIDIA graphics, no duplicate RTSP sessions,
    no tracker/ReID/face/pose/heatmap/UI.
    """

    def __init__(self):
        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.pipeline = Gst.Pipeline.new("intel-display-yolo26m-v3")
        if self.pipeline is None:
            raise RuntimeError("failed to create pipeline")

        self.cameras = [
            dict(item)
            for item in camera_config().get("cameras", [])
            if item.get("online", True)
        ]
        if len(self.cameras) != BATCH_SIZE:
            raise RuntimeError(f"strict batch=6 needs six cameras; found {len(self.cameras)}")
        self.camera_ids = [str(c["id"]) for c in self.cameras]

        self.wall_width = max(960, int(os.environ.get("AI_INTEL_WALL_WIDTH", "1920")))
        self.wall_height = max(360, int(os.environ.get("AI_INTEL_WALL_HEIGHT", "720")))
        self.tile_width = self.wall_width // 3
        self.tile_height = self.wall_height // 2
        self.rtsp_latency_ms = max(60, int(os.environ.get("AI_INTEL_RTSP_LATENCY_MS", "120")))
        self.box_hold_ms = max(800.0, float(os.environ.get("AI_DETECTION_BOX_HOLD_MS", "2500")))
        self.batch_fps = max(0.25, float(os.environ.get("AI_YOLO_BATCH_FPS", "0.80")))
        self.capture_timeout = max(0.10, float(os.environ.get("AI_YOLO_CAPTURE_TIMEOUT", "0.35")))
        self.conf = min(0.90, max(0.05, float(os.environ.get("AI_YOLO_CONF", "0.16"))))

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
        self.detector_start = 0.0
        self.detector_thread: threading.Thread | None = None

        now = time.monotonic()
        self.stats = {cid: CameraStat(last_time=now) for cid in self.camera_ids}
        self.queues = {}
        self.decoders = {}
        self._tee_pads = []
        self._comp_pads = []
        self.wall_frames = 0
        self.wall_last_frames = 0
        self.wall_last_time = now

        self.use_va_compositor = Gst.ElementFactory.find("vacompositor") is not None
        comp_name = "vacompositor" if self.use_va_compositor else "compositor"
        self.compositor = self._make(comp_name, "intel_compositor")
        self._set_if(self.compositor, "ignore-inactive-pads", True)
        self._set_if(self.compositor, "background", 1)
        self.pipeline.add(self.compositor)

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        # Compose in Intel VA memory, then download ONCE for the whole 1920x720
        # wall. Cairo draws a handful of boxes in system memory. This completely
        # avoids NVIDIA graphics/EGL contention with CUDA inference.
        post_name = "vapostproc" if self.use_va_compositor else "videoconvert"
        self.wall_post = self._make(post_name, "intel_wall_post")
        self.wall_caps = self._make("capsfilter", "intel_wall_caps")
        self.overlay = self._make("cairooverlay", "detection_overlay")
        self.wall_queue = self._make("queue", "intel_wall_queue")
        self.sink = self._choose_sink()

        self.wall_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGRA,width={self.wall_width},height={self.wall_height},framerate=20/1"
            ),
        )
        self._set_if(self.wall_queue, "max-size-buffers", 1)
        self._set_if(self.wall_queue, "max-size-bytes", 0)
        self._set_if(self.wall_queue, "max-size-time", 0)
        self._set_if(self.wall_queue, "leaky", 2)

        # Never let display timestamps create a second drop policy. The compositor
        # already produces a fixed 20 FPS wall from latest camera frames.
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "async", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "enable-last-sample", False)

        for element in (self.wall_post, self.wall_caps, self.overlay, self.wall_queue, self.sink):
            self.pipeline.add(element)
        if not self.compositor.link(self.wall_post):
            raise RuntimeError(f"{comp_name} -> wall post failed")
        if not self.wall_post.link(self.wall_caps):
            raise RuntimeError("wall post -> BGRA caps failed")
        if not self.wall_caps.link(self.overlay):
            raise RuntimeError("BGRA caps -> cairooverlay failed")
        if not self.overlay.link(self.wall_queue):
            raise RuntimeError("cairooverlay -> wall queue failed")
        if not self.wall_queue.link(self.sink):
            raise RuntimeError("wall queue -> display sink failed")

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
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session != "wayland" and self.Gst.ElementFactory.find("ximagesink") is not None:
            return self._make("ximagesink", "intel_wall_sink")
        if self.Gst.ElementFactory.find("waylandsink") is not None:
            return self._make("waylandsink", "intel_wall_sink")
        return self._make("autovideosink", "intel_wall_sink")

    def _request_tee_pad(self, tee):
        request = getattr(tee, "request_pad_simple", None)
        pad = request("src_%u") if request else None
        if pad is None:
            pad = tee.get_request_pad("src_%u")
        if pad is None:
            raise RuntimeError("tee request pad failed")
        self._tee_pads.append((tee, pad))
        return pad

    def _request_comp_pad(self, index: int):
        request = getattr(self.compositor, "request_pad_simple", None)
        pad = request("sink_%u") if request else None
        if pad is None:
            pad = self.compositor.get_request_pad("sink_%u")
        if pad is None:
            raise RuntimeError("compositor request pad failed")

        row, col = divmod(index, 3)
        values = {
            "xpos": col * self.tile_width,
            "ypos": row * self.tile_height,
            "width": self.tile_width,
            "height": self.tile_height,
        }
        for key, value in values.items():
            if pad.find_property(key) is not None:
                pad.set_property(key, value)
        self._comp_pads.append(pad)
        return pad

    def _link_tee(self, tee, queue, cid: str, branch: str) -> None:
        src = self._request_tee_pad(tee)
        dst = queue.get_static_pad("sink")
        if src.link(dst) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> {branch} failed")

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
        decoder = self._make(f"va{codec}dec", f"va_decode_{index}")
        tee = self._make("tee", f"decoded_tee_{index}")
        display_q = self._make("queue", f"display_q_{index}")
        infer_q = self._make("queue", f"infer_q_{index}")
        infer_post = self._make("vapostproc", f"infer_post_{index}")
        infer_caps = self._make("capsfilter", f"infer_caps_{index}")
        appsink = self._make("appsink", f"infer_sink_{index}")

        source.set_property("location", uri)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "tcp-timeout", 5_000_000)

        for q, buffers in ((display_q, 2), (infer_q, 1)):
            self._set_if(q, "max-size-buffers", buffers)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)

        self._set_if(infer_post, "disable-passthrough", True)
        infer_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRA,width={INFER_WIDTH},height={INFER_HEIGHT}"
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
            infer_q, infer_post, infer_caps, appsink,
        ):
            self.pipeline.add(element)

        if not depay.link(parser):
            raise RuntimeError(f"{cid}: depay -> parser failed")
        if not parser.link(decoder):
            raise RuntimeError(f"{cid}: parser -> Intel decoder failed")
        if not decoder.link(tee):
            raise RuntimeError(f"{cid}: Intel decoder -> tee failed")

        self._link_tee(tee, display_q, cid, "display")
        self._link_tee(tee, infer_q, cid, "infer")

        comp_pad = self._request_comp_pad(index)
        display_src = display_q.get_static_pad("src")
        if display_src.link(comp_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display -> compositor failed")
        display_src.add_probe(self.Gst.PadProbeType.BUFFER, self._display_probe, cid)

        infer_src = infer_q.get_static_pad("src")
        infer_src.add_probe(self.Gst.PadProbeType.BUFFER, self._ticket_probe, cid)
        if not infer_q.link(infer_post):
            raise RuntimeError(f"{cid}: infer queue -> VA postproc failed")
        if not infer_post.link(infer_caps):
            raise RuntimeError(f"{cid}: VA postproc -> infer caps failed")
        if not infer_caps.link(appsink):
            raise RuntimeError(f"{cid}: infer caps -> appsink failed")
        appsink.connect("new-sample", self._on_infer_sample, cid)

        source.connect("pad-added", self._on_rtsp_pad, depay, codec, cid)
        self.queues[cid] = display_q
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
            print(f"INTEL_V3 {cid} RTSP -> depay failed: {result}", flush=True)

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
    def _bgra_to_bgr(data, width: int, height: int):
        import numpy as np

        raw = np.frombuffer(data, dtype=np.uint8)
        # VA/system buffers can have row padding. Derive stride from mapped size.
        stride = raw.size // max(1, int(height))
        if stride < int(width) * 4:
            raise ValueError(f"BGRA stride too small: {stride} for width={width}")
        rows = raw[: stride * int(height)].reshape((int(height), stride))
        pixels = rows[:, : int(width) * 4].reshape((int(height), int(width), 4))
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
            frame = self._bgra_to_bgr(mapped.data, width, height)
        except Exception:
            with self.ticket_lock:
                self.tickets[cid] = True
            return self.Gst.FlowReturn.OK
        finally:
            buffer.unmap(mapped)
        self.latest.put(cid, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _resolve_model(self) -> str:
        p = Path(MODEL_SPEC)
        if p.is_file():
            return str(p)
        p = ROOT / MODEL_SPEC
        return str(p) if p.is_file() else MODEL_SPEC

    def _infer_loop(self) -> None:
        if self.stop_event.wait(2.0):
            return
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

            model_spec = self._resolve_model()
            device = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(
                f"YOLO26M_V3 loading model={model_spec} device={device} "
                f"sm={capability[0]}.{capability[1]} input={INFER_WIDTH}x{INFER_HEIGHT} strict_batch=6",
                flush=True,
            )
            model = YOLO(model_spec)
            kwargs = {
                "imgsz": (INFER_HEIGHT, INFER_WIDTH),
                "rect": True,
                "classes": [0],
                "conf": self.conf,
                "iou": 0.50,
                "max_det": 20,
                "device": "cuda:0",
                "verbose": False,
                "stream": False,
            }
            warm = [
                np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8)
                for _ in self.camera_ids
            ]
            with torch.inference_mode():
                model.predict(source=warm, **kwargs)

            with self.det_lock:
                self.detector_ready = True
                self.detector_start = time.monotonic()
            print(
                f"YOLO26M_V3 ready model=YOLO26m strict_batch=6 batch_cap={self.batch_fps:.2f}/s; "
                "NVIDIA graphics/decode=off",
                flush=True,
            )
        except BaseException as exc:
            with self.det_lock:
                self.detector_error = f"{type(exc).__name__}: {exc}"
            print(f"YOLO26M_V3 disabled: {self.detector_error}", file=sys.stderr, flush=True)
            return

        previous = {cid: 0 for cid in self.camera_ids}
        interval = 1.0 / self.batch_fps
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
                next_at = time.monotonic() + 0.15
                continue

            frames = []
            captured = []
            for cid, (version, captured_mono, frame) in zip(self.camera_ids, rows):
                previous[cid] = int(version)
                frames.append(frame)
                captured.append(float(captured_mono))

            started = time.monotonic()
            try:
                with torch.inference_mode():
                    predictions = model.predict(source=frames, **kwargs)
                ended = time.monotonic()
                snapshots = {}
                counts = {}
                for cid, frame, pred, captured_mono in zip(
                    self.camera_ids, frames, predictions, captured
                ):
                    items = []
                    boxes = getattr(pred, "boxes", None)
                    if boxes is not None and len(boxes):
                        coords_all = boxes.xyxy.detach().cpu().tolist()
                        conf_all = boxes.conf.detach().cpu().tolist()
                        for coords, confidence in zip(coords_all, conf_all):
                            items.append({
                                "xyxy": [float(v) for v in coords],
                                "confidence": float(confidence),
                            })
                    counts[cid] = len(items)
                    snapshots[cid] = {
                        "captured_mono": captured_mono,
                        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                        "boxes": items,
                    }

                with self.det_lock:
                    self.batch_calls += 1
                    self.last_batch_ms = (ended - started) * 1000.0
                    self.last_spread_ms = (max(captured) - min(captured)) * 1000.0
                    self.last_counts = counts
                    self.latest_detections = snapshots
                next_at = max(ended, started + interval)
            except BaseException as exc:
                with self.det_lock:
                    self.batch_errors += 1
                    self.detector_error = f"{type(exc).__name__}: {exc}"
                print(f"YOLO26M_V3 batch error: {self.detector_error}", file=sys.stderr, flush=True)
                next_at = time.monotonic() + 0.5

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

    def _print_stats(self) -> bool:
        now = time.monotonic()
        source_parts = []
        source_fps = []
        for cid in self.camera_ids:
            stat = self.stats[cid]
            elapsed = max(0.001, now - stat.last_time)
            stat.fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_time = now
            source_fps.append(stat.fps)
            q = int(self.queues[cid].get_property("current-level-buffers"))
            source_parts.append(f"{cid}:{stat.fps:.1f}fps q={q}")

        wall_elapsed = max(0.001, now - self.wall_last_time)
        wall_fps = (self.wall_frames - self.wall_last_frames) / wall_elapsed
        self.wall_last_frames = self.wall_frames
        self.wall_last_time = now

        devices = []
        for cid, decoder in self.decoders.items():
            device = "?"
            if decoder.find_property("device-path") is not None:
                try:
                    device = str(decoder.get_property("device-path") or "?")
                except Exception:
                    pass
            devices.append(f"{cid}:{device}")

        with self.det_lock:
            ready = self.detector_ready
            error = self.detector_error
            calls = self.batch_calls
            batch_ms = self.last_batch_ms
            spread_ms = self.last_spread_ms
            errors = self.batch_errors
            timeouts = self.capture_timeouts
            counts = dict(self.last_counts)
            started = self.detector_start

        det_elapsed = max(0.001, now - started) if started else 1.0
        batch_rate = calls / det_elapsed if ready else 0.0
        count_text = " ".join(f"{cid}:{counts.get(cid, 0)}" for cid in self.camera_ids)
        print("INTEL_V3 " + " | ".join(source_parts), flush=True)
        print(
            f"INTEL_V3 wall_fps={wall_fps:.1f} wall={self.wall_width}x{self.wall_height} "
            f"va_compositor={int(self.use_va_compositor)} devices=[{' '.join(devices)}]",
            flush=True,
        )
        print(
            f"YOLO26M_V3 ready={int(ready)} batches={batch_rate:.2f}/s batch={batch_ms:.1f}ms "
            f"spread={spread_ms:.1f}ms timeouts={timeouts} errors={errors} persons=[{count_text}]"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        return True

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"INTEL_V3 ERROR source={source} message={err.message} debug={debug or ''}",
                file=sys.stderr,
                flush=True,
            )
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            source = message.src.get_name() if message.src else "unknown"
            print(
                f"INTEL_V3 WARNING source={source} message={err.message} debug={debug or ''}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.EOS:
            self.loop.quit()

    def run(self) -> int:
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("Intel V3 camera pipeline failed to PLAY")

        self.detector_thread = threading.Thread(
            target=self._infer_loop,
            name="yolo26m-strict-batch6",
            daemon=True,
        )
        self.detector_thread.start()

        print(
            "INTEL_V3 started: one RTSP/decode per camera; Intel VA display/decode; "
            f"NVIDIA=YOLO26m-only strict_batch=6 infer={INFER_WIDTH}x{INFER_HEIGHT} "
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
            if self.detector_thread is not None:
                self.detector_thread.join(timeout=3.0)
        return 0


def missing_plugins() -> list[str]:
    Gst = _gstreamer()
    required = [
        "rtspsrc", "rtph264depay", "rtph265depay", "h264parse", "h265parse",
        "vah264dec", "vah265dec", "vapostproc", "cairooverlay",
    ]
    if Gst.ElementFactory.find("vacompositor") is not None:
        required.append("vacompositor")
    else:
        required.extend(["compositor", "videoconvert"])
    return [name for name in required if Gst.ElementFactory.find(name) is None]


def run() -> int:
    missing = missing_plugins()
    if missing:
        raise RuntimeError(
            "Intel smooth-display path cannot start; missing GStreamer plugins: "
            + ", ".join(missing)
            + ". Install the VA/GStreamer packages instead of falling back to NVIDIA rendering."
        )
    return IntelDisplayYolo26mV3().run()


if __name__ == "__main__":
    raise SystemExit(run())
