from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Cairo must be registered as a foreign GI type before Gst/cairooverlay is used.
# Without this, cairooverlay's cairo_t* arrives in Python as opaque GBoxed.
CAIRO_IMPORT_ERROR = ""
try:
    import gi
    gi.require_foreign("cairo")
    import cairo  # noqa: F401
except Exception as exc:  # validated in run(), before any pipeline starts
    CAIRO_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from shared.config import camera_config
from services.ml_service.cameras.gstreamer import _gstreamer, authenticated_source

ROOT = Path(__file__).resolve().parents[3]
MODEL_SPEC = os.environ.get("AI_YOLO_MODEL", "yolo26m.pt")
BATCH_SIZE = 6
DISPLAY_FPS = max(10, int(os.environ.get("AI_DISPLAY_FPS", "20")))
WALL_WIDTH = max(960, int(os.environ.get("AI_WALL_WIDTH", "1920")))
WALL_HEIGHT = max(360, int(os.environ.get("AI_WALL_HEIGHT", "720")))
TILE_WIDTH = WALL_WIDTH // 3
TILE_HEIGHT = WALL_HEIGHT // 2
INFER_WIDTH = max(320, int(os.environ.get("AI_YOLO_INFER_WIDTH", "448")))
INFER_HEIGHT = max(192, int(os.environ.get("AI_YOLO_INFER_HEIGHT", "256")))


@dataclass
class CameraRuntime:
    cid: str
    videorate: object
    display_queue: object
    decoder: object


class FreshSix:
    """Latest-only six-camera inference mailbox."""

    def __init__(self):
        self.cv = threading.Condition()
        self.rows: dict[str, tuple[int, float, object]] = {}
        self.versions: dict[str, int] = {}
        self.closed = False

    def put(self, cid: str, captured: float, frame) -> None:
        with self.cv:
            version = self.versions.get(cid, 0) + 1
            self.versions[cid] = version
            self.rows[cid] = (version, float(captured), frame)
            self.cv.notify_all()

    def wait_new(self, ids: list[str], old: dict[str, int], timeout: float):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not self.closed:
                if all(cid in self.rows and self.rows[cid][0] > old.get(cid, 0) for cid in ids):
                    return [self.rows[cid] for cid in ids]
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return None
                self.cv.wait(remain)
        return None

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()


def _model_path(spec: str) -> str:
    p = Path(spec)
    if p.is_file():
        return str(p)
    p = ROOT / spec
    return str(p) if p.is_file() else spec


def _yolo_worker(job_q, result_q, worker_cpus, model_spec: str, width: int, height: int, conf: float):
    """All PyTorch/Ultralytics work is isolated from the camera process."""
    try:
        if worker_cpus and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(worker_cpus))
            except Exception:
                pass
        try:
            os.nice(10)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

        import numpy as np
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        torch.cuda.set_device(0)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        model_path = _model_path(model_spec)
        model = YOLO(model_path)
        kwargs = {
            "imgsz": (height, width),
            "rect": True,
            "classes": [0],
            "conf": float(conf),
            "iou": 0.50,
            "max_det": 20,
            "device": "cuda:0",
            "verbose": False,
            "stream": False,
        }
        warm = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(BATCH_SIZE)]
        with torch.inference_mode():
            model.predict(source=warm, **kwargs)

        result_q.put_nowait({
            "type": "ready",
            "device": torch.cuda.get_device_name(0),
            "sm": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "model": model_path,
        })

        while True:
            job = job_q.get()
            if job is None:
                return
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    preds = model.predict(source=job["frames"], **kwargs)
                ended = time.monotonic()
                snapshots = {}
                counts = {}
                for cid, frame, pred, captured in zip(
                    job["camera_ids"], job["frames"], preds, job["captured"]
                ):
                    items = []
                    boxes = getattr(pred, "boxes", None)
                    if boxes is not None and len(boxes):
                        coords_all = boxes.xyxy.detach().cpu().tolist()
                        conf_all = boxes.conf.detach().cpu().tolist()
                        for coords, score in zip(coords_all, conf_all):
                            items.append({
                                "xyxy": [float(x) for x in coords],
                                "confidence": float(score),
                            })
                    counts[cid] = len(items)
                    snapshots[cid] = {
                        "captured_mono": float(captured),
                        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                        "boxes": items,
                    }
                payload = {
                    "type": "result",
                    "seq": int(job["seq"]),
                    "batch_ms": (ended - started) * 1000.0,
                    "spread_ms": (max(job["captured"]) - min(job["captured"])) * 1000.0,
                    "snapshots": snapshots,
                    "counts": counts,
                }
                try:
                    result_q.put_nowait(payload)
                except queue.Full:
                    try:
                        result_q.get_nowait()
                    except Exception:
                        pass
                    result_q.put_nowait(payload)
            except BaseException as exc:
                try:
                    result_q.put_nowait({
                        "type": "error",
                        "seq": int(job.get("seq", -1)),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                except Exception:
                    pass
    except BaseException as exc:
        try:
            result_q.put_nowait({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


class SmoothCameraYolo26mV6:
    """Fresh architecture: NVDEC decode, CPU wall, process-isolated YOLO26m.

    Camera path (20 FPS, bounded):
      RTSP -> depay -> parse -> NVDEC -> system NV12 -> tee
        display -> queue(3) -> videorate 20 -> 640x360 BGRA -> CPU compositor
                -> cairooverlay -> queue(2) -> ximagesink sync=1 max-lateness=-1
        infer   -> queue(1) -> ticket gate -> CPU scale 448x256 BGRx -> appsink

    Detection path:
      exactly six fresh ticket frames -> multiprocessing queue -> separate
      YOLO26m CUDA process -> rectangle snapshots back to cairooverlay.

    There is no nvstreammux, nvmultistreamtiler, nveglglessink, nvdsosd,
    software video decoder, Qt UI, tracker, ReID, face, pose or heatmap.
    """

    def __init__(self):
        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.pipeline = Gst.Pipeline.new("smooth-camera-yolo26m-v6")
        if self.pipeline is None:
            raise RuntimeError("failed to create V6 pipeline")

        self.cameras = [dict(c) for c in camera_config().get("cameras", []) if c.get("online", True)]
        if len(self.cameras) != BATCH_SIZE:
            raise RuntimeError(f"V6 strict batch=6 requires exactly six enabled cameras, found {len(self.cameras)}")
        self.camera_ids = [str(c["id"]) for c in self.cameras]

        self.rtsp_latency_ms = max(60, int(os.environ.get("AI_RTSP_LATENCY_MS", "120")))
        self.udp_buffer = max(1_048_576, int(os.environ.get("AI_RTSP_UDP_BUFFER", str(8 * 1024 * 1024))))
        self.conf = min(0.9, max(0.05, float(os.environ.get("AI_YOLO_CONF", "0.16"))))
        self.batch_target = max(0.25, float(os.environ.get("AI_YOLO_BATCH_FPS", "0.65")))
        self.batch_floor = max(0.20, float(os.environ.get("AI_YOLO_BATCH_FPS_MIN", "0.35")))
        self.batch_current = self.batch_target
        self.capture_timeout = max(0.15, float(os.environ.get("AI_YOLO_CAPTURE_TIMEOUT", "0.45")))
        self.box_hold_ms = max(900.0, float(os.environ.get("AI_DETECTION_BOX_HOLD_MS", "2200")))

        self.stop_event = threading.Event()
        self.mailbox = FreshSix()
        self.ticket_lock = threading.Lock()
        self.tickets = {cid: False for cid in self.camera_ids}
        self.det_lock = threading.RLock()
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
        self.overlay_errors = 0

        self.runtimes: dict[str, CameraRuntime] = {}
        self.infer_queues = {}
        self._tee_pads = []
        self._comp_pads = []
        self._scheduler_thread: threading.Thread | None = None
        self._worker = None
        self._job_q = None
        self._result_q = None
        self._job_seq = 0
        self._worker_cpus: list[int] = []

        # Prefer nvcodec because it can negotiate plain system-memory NV12
        # directly. Fall back to DeepStream NVDEC + one NVMM->system conversion.
        self.decoder_backend = self._choose_decoder_backend()

        self.compositor = self._make("compositor", "wall_compositor")
        self._set_if(self.compositor, "background", 1)
        self._set_if(self.compositor, "ignore-inactive-pads", True)
        self._set_if(self.compositor, "max-threads", min(4, max(1, os.cpu_count() or 1)))
        self._set_if(self.compositor, "latency", 60_000_000)
        self.pipeline.add(self.compositor)

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        self.wall_caps = self._make("capsfilter", "wall_caps")
        self.overlay = self._make("cairooverlay", "person_boxes")
        self.wall_queue = self._make("queue", "wall_queue")
        self.sink = self._choose_sink()

        self.wall_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGRA,width={WALL_WIDTH},height={WALL_HEIGHT},framerate={DISPLAY_FPS}/1,pixel-aspect-ratio=1/1"
            ),
        )
        self._queue_latest(self.wall_queue, 2)
        self._set_if(self.sink, "sync", True)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "enable-last-sample", False)

        for element in (self.wall_caps, self.overlay, self.wall_queue, self.sink):
            self.pipeline.add(element)
        if not self.compositor.link(self.wall_caps):
            raise RuntimeError("compositor -> wall caps failed")
        if not self.wall_caps.link(self.overlay):
            raise RuntimeError("wall caps -> cairooverlay failed")
        if not self.overlay.link(self.wall_queue):
            raise RuntimeError("cairooverlay -> wall queue failed")
        if not self.wall_queue.link(self.sink):
            raise RuntimeError("wall queue -> display sink failed")

        self.overlay.connect("draw", self._draw_overlay)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()
        GLib.timeout_add(50, self._drain_worker_results)
        GLib.timeout_add_seconds(5, self._print_stats)

    def _choose_decoder_backend(self) -> str:
        h264 = self.Gst.ElementFactory.find("nvh264dec") is not None
        h265 = self.Gst.ElementFactory.find("nvh265dec") is not None
        if h264 and h265:
            return "nvcodec-system"
        if (
            self.Gst.ElementFactory.find("nvv4l2decoder") is not None
            and self.Gst.ElementFactory.find("nvvideoconvert") is not None
        ):
            return "deepstream-nvdec-system"
        raise RuntimeError(
            "No NVIDIA hardware decoder path found. Need nvh264dec+nvh265dec or nvv4l2decoder+nvvideoconvert."
        )

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"missing GStreamer element: {factory}")
        return element

    @staticmethod
    def _set_if(element, name: str, value) -> bool:
        pspec = element.find_property(name)
        if pspec is None:
            return False
        try:
            element.set_property(name, value)
            return True
        except Exception:
            return False

    def _queue_latest(self, element, buffers: int) -> None:
        self._set_if(element, "max-size-buffers", buffers)
        self._set_if(element, "max-size-bytes", 0)
        self._set_if(element, "max-size-time", 0)
        self._set_if(element, "leaky", 2)

    def _choose_sink(self):
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session != "wayland" and self.Gst.ElementFactory.find("ximagesink") is not None:
            return self._make("ximagesink", "wall_sink")
        if self.Gst.ElementFactory.find("waylandsink") is not None:
            return self._make("waylandsink", "wall_sink")
        return self._make("autovideosink", "wall_sink")

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
        for key, value in {
            "xpos": col * TILE_WIDTH,
            "ypos": row * TILE_HEIGHT,
            "width": TILE_WIDTH,
            "height": TILE_HEIGHT,
        }.items():
            if pad.find_property(key) is not None:
                pad.set_property(key, value)
        self._comp_pads.append(pad)
        return pad

    def _add_camera(self, index: int, camera: dict) -> None:
        cid = str(camera["id"])
        codec = str(camera.get("display_codec") or camera.get("codec") or "h264").lower()
        if codec not in {"h264", "h265"}:
            raise RuntimeError(f"{cid}: unsupported codec {codec}")
        uri = authenticated_source({**camera, "source": camera.get("display_source") or camera.get("source")})
        if not isinstance(uri, str) or not uri.startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"{cid}: invalid RTSP URL")

        source = self._make("rtspsrc", f"rtsp_{index}")
        depay = self._make(f"rtp{codec}depay", f"depay_{index}")
        parser = self._make(f"{codec}parse", f"parse_{index}")
        decoder_factory = f"nv{codec}dec" if self.decoder_backend == "nvcodec-system" else "nvv4l2decoder"
        decoder = self._make(decoder_factory, f"decoder_{index}")
        sys_caps = self._make("capsfilter", f"system_nv12_{index}")
        tee = self._make("tee", f"decoded_tee_{index}")

        source.set_property("location", uri)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "udp-buffer-size", self.udp_buffer)
        self._set_if(source, "tcp-timeout", 5_000_000)

        if self.decoder_backend == "nvcodec-system":
            self._set_if(decoder, "max-display-delay", 0)
            self._set_if(decoder, "num-output-surfaces", 2)
            sys_caps.set_property("caps", self.Gst.Caps.from_string("video/x-raw,format=NV12"))
            decode_tail = [decoder, sys_caps]
        else:
            self._set_if(decoder, "cudadec-memtype", 1)  # pinned host decode buffers on dGPU
            self._set_if(decoder, "num-extra-surfaces", 8)
            convert = self._make("nvvideoconvert", f"download_{index}")
            self._set_if(convert, "gpu-id", 0)
            sys_caps.set_property("caps", self.Gst.Caps.from_string("video/x-raw,format=NV12"))
            decode_tail = [decoder, convert, sys_caps]

        display_q = self._make("queue", f"display_q_{index}")
        rate = self._make("videorate", f"rate_{index}")
        display_convert = self._make("videoconvertscale", f"display_convert_{index}")
        display_caps = self._make("capsfilter", f"display_caps_{index}")
        self._queue_latest(display_q, 3)
        self._set_if(rate, "silent", True)
        self._set_if(rate, "drop-only", False)
        self._set_if(rate, "max-duplication-time", 150_000_000)
        display_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRA,width={TILE_WIDTH},height={TILE_HEIGHT},framerate={DISPLAY_FPS}/1,pixel-aspect-ratio=1/1"
            ),
        )

        infer_q = self._make("queue", f"infer_q_{index}")
        infer_convert = self._make("videoconvertscale", f"infer_convert_{index}")
        infer_caps = self._make("capsfilter", f"infer_caps_{index}")
        appsink = self._make("appsink", f"infer_sink_{index}")
        self._queue_latest(infer_q, 1)
        infer_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INFER_WIDTH},height={INFER_HEIGHT},pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(appsink, "drop", True)
        self._set_if(appsink, "max-buffers", 1)
        self._set_if(appsink, "sync", False)
        self._set_if(appsink, "emit-signals", True)
        self._set_if(appsink, "wait-on-eos", False)
        self._set_if(appsink, "enable-last-sample", False)

        elements = [source, depay, parser, *decode_tail, tee, display_q, rate, display_convert, display_caps,
                    infer_q, infer_convert, infer_caps, appsink]
        for element in elements:
            self.pipeline.add(element)

        if not depay.link(parser):
            raise RuntimeError(f"{cid}: depay -> parser failed")
        if not parser.link(decoder):
            raise RuntimeError(f"{cid}: parser -> decoder failed")
        if self.decoder_backend == "nvcodec-system":
            if not decoder.link(sys_caps) or not sys_caps.link(tee):
                raise RuntimeError(f"{cid}: NVDEC -> system NV12 -> tee failed")
        else:
            convert = decode_tail[1]
            if not decoder.link(convert) or not convert.link(sys_caps) or not sys_caps.link(tee):
                raise RuntimeError(f"{cid}: DeepStream NVDEC download -> tee failed")

        self._link_tee(tee, display_q, cid, "display")
        self._link_tee(tee, infer_q, cid, "infer")

        if not display_q.link(rate):
            raise RuntimeError(f"{cid}: display queue -> videorate failed")
        if not rate.link(display_convert):
            raise RuntimeError(f"{cid}: videorate -> display convert failed")
        if not display_convert.link(display_caps):
            raise RuntimeError(f"{cid}: display convert -> caps failed")
        comp_pad = self._request_comp_pad(index)
        if display_caps.get_static_pad("src").link(comp_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display -> compositor failed")

        infer_q.get_static_pad("src").add_probe(self.Gst.PadProbeType.BUFFER, self._ticket_probe, cid)
        if not infer_q.link(infer_convert) or not infer_convert.link(infer_caps) or not infer_caps.link(appsink):
            raise RuntimeError(f"{cid}: inference branch link failed")
        appsink.connect("new-sample", self._on_infer_sample, cid)
        source.connect("pad-added", self._on_rtsp_pad, depay, codec, cid)

        self.runtimes[cid] = CameraRuntime(cid=cid, videorate=rate, display_queue=display_q, decoder=decoder)
        self.infer_queues[cid] = infer_q

    def _on_rtsp_pad(self, _source, pad, depay, codec: str, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        s = caps.get_structure(0)
        if str(s.get_name()) != "application/x-rtp":
            return
        media = str(s.get_string("media") or "").lower()
        enc = str(s.get_string("encoding-name") or "").lower()
        ok_codec = (codec == "h264" and "h264" in enc) or (codec == "h265" and ("h265" in enc or "hevc" in enc))
        if media != "video" or not ok_codec:
            return
        sink = depay.get_static_pad("sink")
        if sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"V6 {cid} RTSP link failed: {result}", flush=True)

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
        stride = raw.size // max(1, height)
        if stride < width * 4:
            raise ValueError(f"invalid BGRx stride={stride}, width={width}")
        rows = raw[: stride * height].reshape((height, stride))
        pixels = rows[:, : width * 4].reshape((height, width, 4))
        return pixels[..., :3].copy()

    def _on_infer_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK
        s = sample.get_caps().get_structure(0)
        width = int(s.get_value("width"))
        height = int(s.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.OK
        try:
            frame = self._bgrx_to_bgr(mapped.data, width, height)
        finally:
            buffer.unmap(mapped)
        self.mailbox.put(cid, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _draw_overlay(self, _overlay, cr, _timestamp, _duration) -> None:
        # gi.require_foreign('cairo') guarantees cr is a pycairo.Context.
        if not hasattr(cr, "set_source_rgba"):
            self.overlay_errors += 1
            if self.overlay_errors == 1:
                print(
                    f"V6 overlay context is not pycairo: type={type(cr)!r}. Install python3-gi-cairo.",
                    file=sys.stderr,
                    flush=True,
                )
            return

        now = time.monotonic()
        with self.det_lock:
            snapshots = {cid: dict(v) for cid, v in self.latest_detections.items() if v}
        for index, cid in enumerate(self.camera_ids):
            snapshot = snapshots.get(cid)
            if not snapshot:
                continue
            captured = float(snapshot.get("captured_mono") or 0.0)
            if captured <= 0.0 or (now - captured) * 1000.0 > self.box_hold_ms:
                continue
            fw, fh = snapshot.get("frame_size") or [INFER_WIDTH, INFER_HEIGHT]
            fw = max(1.0, float(fw)); fh = max(1.0, float(fh))
            row, col = divmod(index, 3)
            ox = float(col * TILE_WIDTH); oy = float(row * TILE_HEIGHT)
            sx = TILE_WIDTH / fw; sy = TILE_HEIGHT / fh
            cr.set_source_rgba(0.0, 1.0, 0.10, 1.0)
            cr.set_line_width(3.0)
            for item in snapshot.get("boxes") or []:
                x1, y1, x2, y2 = [float(x) for x in item.get("xyxy", [0, 0, 1, 1])]
                x1 = max(0.0, min(fw, x1)); y1 = max(0.0, min(fh, y1))
                x2 = max(x1 + 1.0, min(fw, x2)); y2 = max(y1 + 1.0, min(fh, y2))
                cr.rectangle(ox + x1 * sx, oy + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
                cr.stroke()

    def _scheduler_loop(self) -> None:
        old = {cid: 0 for cid in self.camera_ids}
        next_at = time.monotonic() + 2.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < next_at:
                if self.stop_event.wait(min(0.05, next_at - now)):
                    return
                continue
            self._request_six()
            rows = self.mailbox.wait_new(self.camera_ids, old, self.capture_timeout)
            self._clear_tickets()
            if rows is None:
                self.capture_timeouts += 1
                next_at = time.monotonic() + 0.15
                continue
            frames, captured = [], []
            for cid, row in zip(self.camera_ids, rows):
                version, ts, frame = row
                old[cid] = int(version)
                frames.append(frame)
                captured.append(float(ts))
            self._job_seq += 1
            job = {
                "seq": self._job_seq,
                "camera_ids": list(self.camera_ids),
                "frames": frames,
                "captured": captured,
            }
            try:
                self._job_q.put_nowait(job)
            except queue.Full:
                # Never backpressure the camera process. Keep only the newest job.
                try:
                    self._job_q.get_nowait()
                except Exception:
                    pass
                try:
                    self._job_q.put_nowait(job)
                except Exception:
                    pass
            with self.det_lock:
                rate = self.batch_current
            next_at = time.monotonic() + (1.0 / max(self.batch_floor, rate))

    def _drain_worker_results(self) -> bool:
        if self._result_q is None:
            return True
        for _ in range(8):
            try:
                msg = self._result_q.get_nowait()
            except queue.Empty:
                break
            kind = msg.get("type")
            with self.det_lock:
                if kind == "ready":
                    self.detector_ready = True
                    self.detector_started = time.monotonic()
                    self.detector_error = ""
                    print(
                        f"YOLO26M_V6 ready device={msg.get('device')} sm={msg.get('sm')} model={msg.get('model')} "
                        f"strict_batch=6 input={INFER_WIDTH}x{INFER_HEIGHT} process_isolated=1",
                        flush=True,
                    )
                elif kind == "result":
                    self.batch_calls += 1
                    self.last_batch_ms = float(msg.get("batch_ms") or 0.0)
                    self.last_spread_ms = float(msg.get("spread_ms") or 0.0)
                    self.latest_detections = dict(msg.get("snapshots") or {})
                    self.last_counts = dict(msg.get("counts") or {})
                elif kind in {"error", "fatal"}:
                    self.batch_errors += 1
                    self.detector_error = str(msg.get("error") or "unknown worker error")
                    if kind == "fatal":
                        self.detector_ready = False
        return True

    def _sink_stats(self) -> tuple[int | None, int | None]:
        try:
            if self.sink.find_property("stats") is None:
                return None, None
            s = self.sink.get_property("stats")
            rendered = int(s.get_value("rendered")) if s.has_field("rendered") else None
            dropped = int(s.get_value("dropped")) if s.has_field("dropped") else None
            return rendered, dropped
        except Exception:
            return None, None

    def _print_stats(self) -> bool:
        parts = []
        total_dup = 0
        total_drop = 0
        for cid in self.camera_ids:
            rt = self.runtimes[cid]
            try:
                vin = int(rt.videorate.get_property("in"))
                vout = int(rt.videorate.get_property("out"))
                dup = int(rt.videorate.get_property("duplicate"))
                drop = int(rt.videorate.get_property("drop"))
                q = int(rt.display_queue.get_property("current-level-buffers"))
            except Exception:
                vin = vout = dup = drop = q = -1
            total_dup += max(0, dup); total_drop += max(0, drop)
            parts.append(f"{cid}:in={vin} out={vout} dup={dup} drop={drop} q={q}")

        rendered, sink_dropped = self._sink_stats()
        with self.det_lock:
            ready = self.detector_ready
            calls = self.batch_calls
            errors = self.batch_errors
            error = self.detector_error
            batch_ms = self.last_batch_ms
            spread = self.last_spread_ms
            counts = dict(self.last_counts)
            started = self.detector_started
            rate = self.batch_current

        now = time.monotonic()
        elapsed = max(0.001, now - started) if started else 1.0
        actual_rate = calls / elapsed if ready else 0.0
        count_text = " ".join(f"{cid}:{counts.get(cid, 0)}" for cid in self.camera_ids)

        # Smoothness controller: detector may become slower, never the wall.
        if sink_dropped not in (None, 0) or total_drop > total_dup + 30:
            with self.det_lock:
                self.batch_current = max(self.batch_floor, self.batch_current * 0.85)
        elif ready:
            with self.det_lock:
                self.batch_current = min(self.batch_target, self.batch_current + 0.03)

        print("V6_CAMERA " + " | ".join(parts), flush=True)
        print(
            f"V6_WALL backend={self.decoder_backend} fps_target={DISPLAY_FPS} wall={WALL_WIDTH}x{WALL_HEIGHT} "
            f"sink={self.sink.get_factory().get_name()} rendered={rendered} sink_dropped={sink_dropped} "
            f"videorate_dup={total_dup} videorate_drop={total_drop} overlay_errors={self.overlay_errors}",
            flush=True,
        )
        print(
            f"YOLO26M_V6 ready={int(ready)} batches={actual_rate:.2f}/s cap={rate:.2f}/s "
            f"batch={batch_ms:.1f}ms spread={spread:.1f}ms timeouts={self.capture_timeouts} "
            f"errors={errors} persons=[{count_text}]" + (f" error={error}" if error else ""),
            flush=True,
        )
        return True

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src else "unknown"
            print(f"V6 ERROR source={src} message={err.message} debug={debug or ''}", file=sys.stderr, flush=True)
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            src = message.src.get_name() if message.src else "unknown"
            print(f"V6 WARNING source={src} message={err.message} debug={debug or ''}", flush=True)
        elif message.type == self.Gst.MessageType.EOS:
            self.loop.quit()

    def _start_worker(self) -> None:
        ctx = mp.get_context("spawn")
        self._job_q = ctx.Queue(maxsize=1)
        self._result_q = ctx.Queue(maxsize=3)
        self._worker = ctx.Process(
            target=_yolo_worker,
            args=(self._job_q, self._result_q, self._worker_cpus, MODEL_SPEC, INFER_WIDTH, INFER_HEIGHT, self.conf),
            name="yolo26m-v6",
            daemon=True,
        )
        self._worker.start()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, name="v6-ticket-scheduler", daemon=True)
        self._scheduler_thread.start()

    def run(self) -> int:
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("V6 camera pipeline failed to PLAY")
        self._start_worker()
        print(
            "V6 started: NVDEC hardware decode -> system NV12; exact 20fps videorate -> CPU compositor -> "
            f"cairo bbox -> {self.sink.get_factory().get_name()}; NVIDIA CUDA=YOLO26m-only; "
            f"decoder_backend={self.decoder_backend} strict_batch=6 infer={INFER_WIDTH}x{INFER_HEIGHT} "
            f"wall={WALL_WIDTH}x{WALL_HEIGHT} rtsp_latency={self.rtsp_latency_ms}ms",
            flush=True,
        )
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self._clear_tickets()
            self.mailbox.close()
            if self._job_q is not None:
                try:
                    self._job_q.put_nowait(None)
                except Exception:
                    pass
            self.pipeline.set_state(self.Gst.State.NULL)
            if self._scheduler_thread is not None:
                self._scheduler_thread.join(timeout=2.0)
            if self._worker is not None:
                self._worker.join(timeout=3.0)
                if self._worker.is_alive():
                    self._worker.terminate()
        return 0


def _configure_affinity() -> list[int]:
    if os.environ.get("AI_CPU_AFFINITY", "1").strip().lower() in {"0", "false", "no"}:
        return []
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return []
    try:
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 6:
            return []
        worker = available[-2:]
        display = available[:-2]
        os.sched_setaffinity(0, set(display))
        print(f"V6 CPU affinity display={display} yolo_worker={worker}", flush=True)
        return worker
    except Exception as exc:
        print(f"V6 CPU affinity disabled: {type(exc).__name__}: {exc}", flush=True)
        return []


def _preflight() -> None:
    if CAIRO_IMPORT_ERROR:
        raise RuntimeError(
            "PyGObject Cairo integration is missing. This is why cairooverlay returned gobject.GBoxed instead of "
            "cairo.Context. Install python3-cairo and python3-gi-cairo, then run again. Detail: " + CAIRO_IMPORT_ERROR
        )
    Gst = _gstreamer()
    required = [
        "rtspsrc", "rtph264depay", "rtph265depay", "h264parse", "h265parse",
        "queue", "tee", "videorate", "videoconvertscale", "compositor", "cairooverlay", "appsink",
    ]
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError("Missing GStreamer plugins for V6: " + ", ".join(missing))


def run() -> int:
    _preflight()
    worker_cpus = _configure_affinity()
    app = SmoothCameraYolo26mV6()
    app._worker_cpus = worker_cpus
    return app.run()


if __name__ == "__main__":
    raise SystemExit(run())
