from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue as pyqueue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .camera_core import SixCameraCore
from .native_boxes import NativeBoxBridge

ROOT = Path(__file__).resolve().parents[3]
DETECTOR_CONFIG = ROOT / "config" / "vision_v3_detector.yaml"


def _load_detector_config() -> dict:
    with DETECTOR_CONFIG.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = dict(data.get("detector") or {})
    cfg["box"] = dict(cfg.get("box") or {})
    return cfg


def _rfdetr_worker(job_q, result_q, cfg: dict) -> None:
    """Isolated CUDA worker so detector failures never own the camera pipeline."""
    try:
        try:
            os.nice(8)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

        import torch
        from rfdetr import RFDETRSmall

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        device = str(cfg.get("device", "cuda:0"))
        if device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        model = RFDETRSmall(device=device)
        if bool(cfg.get("optimize_for_inference", False)):
            # RF-DETR optimizes in place; do not assign the return value.
            model.optimize_for_inference()

        threshold = float(cfg.get("threshold", 0.18))
        person_class_id = int(cfg.get("person_class_id", 1))
        warm = np.zeros(
            (
                int(cfg.get("capture_height", 432)),
                int(cfg.get("capture_width", 768)),
                3,
            ),
            dtype=np.uint8,
        )
        with torch.inference_mode():
            model.predict(warm[:, :, ::-1].copy(), threshold=threshold)

        result_q.put(
            {
                "type": "ready",
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda": str(torch.version.cuda),
                "model": "RF-DETR-Small",
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return
            started = time.monotonic()
            try:
                frames_rgb = [np.ascontiguousarray(frame[:, :, ::-1]) for frame in job["frames"]]
                with torch.inference_mode():
                    predictions = model.predict(frames_rgb, threshold=threshold)
                if not isinstance(predictions, list):
                    predictions = [predictions]

                output: dict[str, list[tuple[list[float], float]]] = {}
                for cid, detections in zip(job["cameras"], predictions):
                    rows: list[tuple[list[float], float]] = []
                    xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
                    class_ids = np.asarray(getattr(detections, "class_id", []))
                    confidences = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
                    for box, class_id, confidence in zip(xyxy, class_ids, confidences):
                        if int(class_id) != person_class_id:
                            continue
                        rows.append(([float(v) for v in box.tolist()], float(confidence)))
                    output[cid] = rows

                result_q.put(
                    {
                        "type": "result",
                        "cameras": list(job["cameras"]),
                        "captured": list(job["captured"]),
                        "boxes": output,
                        "batch_ms": (time.monotonic() - started) * 1000.0,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                result_q.put({"type": "batch_error", "error": f"CUDA OOM: {exc}"})
            except BaseException as exc:
                result_q.put({"type": "batch_error", "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})


class FreshFrameMailbox:
    """One latest detector frame per camera; never queues stale CCTV frames."""

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.rows: dict[str, tuple[int, float, np.ndarray]] = {}
        self.versions: dict[str, int] = {}
        self.closed = False

    def put(self, camera_id: str, captured: float, frame: np.ndarray) -> None:
        with self.cv:
            version = self.versions.get(camera_id, 0) + 1
            self.versions[camera_id] = version
            self.rows[camera_id] = (version, captured, frame)
            self.cv.notify_all()

    def wait_group(self, camera_ids: list[str], old: dict[str, int], timeout: float):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not self.closed:
                if all(cid in self.rows and self.rows[cid][0] > old.get(cid, 0) for cid in camera_ids):
                    return [self.rows[cid] for cid in camera_ids]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cv.wait(remaining)
        return None

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


def _state(box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5, max(2.0, x2 - x1), max(2.0, y2 - y1))


def _box(cx: float, cy: float, width: float, height: float):
    return (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5)


@dataclass
class EnvelopeTrack:
    key: int
    cx: float
    cy: float
    width: float
    height: float
    vx: float
    vy: float
    last_det_t: float
    confidence: float


class ProtectiveBoxManager:
    """Display-only full-body envelope around raw RF-DETR boxes.

    This deliberately does NOT alter the raw detector geometry that future NvDCF
    will consume. It adds asymmetric head/feet/side guard space, expands quickly,
    shrinks slowly, and predicts briefly between sparse RF-DETR observations. That
    keeps visible heads, feet and arms from being clipped by a tight/flickering OSD
    rectangle without teaching the tracker fake geometry.
    """

    def __init__(self, width: int, height: int, cfg: dict) -> None:
        self.frame_width = float(width)
        self.frame_height = float(height)
        self.side = float(cfg.get("side_margin", 0.10))
        self.top = float(cfg.get("top_margin", 0.08))
        self.bottom = float(cfg.get("bottom_margin", 0.12))
        self.sitting_extra_side = float(cfg.get("sitting_extra_side", 0.05))
        self.sitting_extra_bottom = float(cfg.get("sitting_extra_bottom", 0.04))
        self.sitting_aspect_threshold = float(cfg.get("sitting_aspect_threshold", 1.55))
        self.duplicate_iou = float(cfg.get("duplicate_iou", 0.80))
        self.hold_sec = float(cfg.get("hold_sec", 1.10))
        self.predict_sec = float(cfg.get("predict_sec", 0.70))
        self.expand_alpha = float(cfg.get("expand_alpha", 0.88))
        self.shrink_alpha = float(cfg.get("shrink_alpha", 0.20))
        self.position_alpha = float(cfg.get("position_alpha", 0.82))
        self.velocity_alpha = float(cfg.get("velocity_alpha", 0.38))
        self.min_width = float(cfg.get("min_width", 12))
        self.min_height = float(cfg.get("min_height", 24))
        self.lock = threading.RLock()
        self.tracks: dict[str, dict[int, EnvelopeTrack]] = {}
        self.next_key = 1

    def _clamp(self, box):
        x1, y1, x2, y2 = box
        return (
            max(0.0, min(self.frame_width - 1.0, x1)),
            max(0.0, min(self.frame_height - 1.0, y1)),
            max(0.0, min(self.frame_width - 1.0, x2)),
            max(0.0, min(self.frame_height - 1.0, y2)),
        )

    def _guard(self, box):
        x1, y1, x2, y2 = box
        width = max(self.min_width, x2 - x1)
        height = max(self.min_height, y2 - y1)
        aspect = height / max(1.0, width)
        side = self.side
        bottom = self.bottom
        if aspect < self.sitting_aspect_threshold:
            side += self.sitting_extra_side
            bottom += self.sitting_extra_bottom

        # Minimum pixel guard matters for small/far people where a percentage alone
        # can still clip a head, hand or shoe.
        pad_side = max(6.0, width * side)
        pad_top = max(6.0, height * self.top)
        pad_bottom = max(8.0, height * bottom)
        return self._clamp((x1 - pad_side, y1 - pad_top, x2 + pad_side, y2 + pad_bottom))

    def _dedupe(self, detections):
        ordered = sorted(detections, key=lambda row: row[1], reverse=True)
        kept = []
        for row in ordered:
            if any(_iou(row[0], old[0]) >= self.duplicate_iou for old in kept):
                continue
            kept.append(row)
        return kept

    def _predict(self, track: EnvelopeTrack, when: float):
        dt = min(self.predict_sec, max(0.0, when - track.last_det_t))
        damping = 1.0 / (1.0 + dt)
        cx = track.cx + track.vx * dt * damping
        cy = track.cy + track.vy * dt * damping
        box = _box(cx, cy, track.width, track.height)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        # Shift the whole envelope at image edges instead of shrinking it.
        sx = -x1 if x1 < 0.0 else ((self.frame_width - 1.0) - x2 if x2 > self.frame_width - 1.0 else 0.0)
        sy = -y1 if y1 < 0.0 else ((self.frame_height - 1.0) - y2 if y2 > self.frame_height - 1.0 else 0.0)
        return self._clamp((x1 + sx, y1 + sy, x1 + sx + width, y1 + sy + height))

    def update(self, camera_id: str, captured_t: float, detections) -> None:
        guarded = [(self._guard(box), conf) for box, conf in self._dedupe(detections)]
        with self.lock:
            current = self.tracks.setdefault(camera_id, {})
            candidates = []
            for key, track in current.items():
                pred = self._predict(track, captured_t)
                pcx, pcy, pw, ph = _state(pred)
                for di, (target, _conf) in enumerate(guarded):
                    tcx, tcy, tw, th = _state(target)
                    dist = math.hypot(tcx - pcx, tcy - pcy) / max(30.0, math.hypot(pw, ph))
                    score = _iou(pred, target) * 0.68 + max(0.0, 1.0 - dist) * 0.32
                    if score >= 0.14:
                        candidates.append((score, key, di))
            candidates.sort(reverse=True)

            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            for _score, key, di in candidates:
                if key in used_tracks or di in used_dets:
                    continue
                used_tracks.add(key)
                used_dets.add(di)
                track = current[key]
                target, conf = guarded[di]
                tcx, tcy, tw, th = _state(target)
                dt = max(0.04, captured_t - track.last_det_t)
                measured_vx = (tcx - track.cx) / dt
                measured_vy = (tcy - track.cy) / dt
                max_vx = self.frame_width * 0.9
                max_vy = self.frame_height * 0.9
                measured_vx = max(-max_vx, min(max_vx, measured_vx))
                measured_vy = max(-max_vy, min(max_vy, measured_vy))
                track.vx = track.vx * (1.0 - self.velocity_alpha) + measured_vx * self.velocity_alpha
                track.vy = track.vy * (1.0 - self.velocity_alpha) + measured_vy * self.velocity_alpha
                track.cx += (tcx - track.cx) * self.position_alpha
                track.cy += (tcy - track.cy) * self.position_alpha
                wa = self.expand_alpha if tw >= track.width else self.shrink_alpha
                ha = self.expand_alpha if th >= track.height else self.shrink_alpha
                track.width += (tw - track.width) * wa
                track.height += (th - track.height) * ha
                track.last_det_t = captured_t
                track.confidence = conf

            for di, (target, conf) in enumerate(guarded):
                if di in used_dets:
                    continue
                cx, cy, width, height = _state(target)
                key = self.next_key
                self.next_key += 1
                current[key] = EnvelopeTrack(key, cx, cy, width, height, 0.0, 0.0, captured_t, conf)

            stale = [key for key, track in current.items() if captured_t - track.last_det_t > self.hold_sec]
            for key in stale:
                current.pop(key, None)

    def render(self, camera_id: str, now: float):
        with self.lock:
            current = self.tracks.get(camera_id, {})
            rows = []
            stale = []
            for key, track in current.items():
                if now - track.last_det_t > self.hold_sec:
                    stale.append(key)
                    continue
                x1, y1, x2, y2 = self._predict(track, now)
                if x2 > x1 and y2 > y1:
                    rows.append((x1, y1, x2, y2, track.confidence))
            for key in stale:
                current.pop(key, None)
            return rows


class SixCameraRFDETR(SixCameraCore):
    def __init__(self) -> None:
        self.det_cfg = _load_detector_config()
        self.capture_width = max(320, int(self.det_cfg.get("capture_width", 768)))
        self.capture_height = max(192, int(self.det_cfg.get("capture_height", 432)))
        self.micro_batch = max(1, min(2, int(self.det_cfg.get("micro_batch", 1))))
        self.capture_lock = threading.Lock()
        self.capture_requested: dict[str, bool] = {}
        self.mailbox = FreshFrameMailbox()
        self.camera_index: dict[str, int] = {}
        self.tee_request_pads: list[tuple[object, object]] = []
        self.det_stop = threading.Event()
        self.det_lock = threading.RLock()
        self.det_ready = False
        self.det_error = ""
        self.det_calls = 0
        self.det_inputs = 0
        self.det_batch_ms = 0.0
        self.det_counts: dict[str, int] = {}
        self.capture_timeouts = 0
        self.meta_boxes = 0
        self.det_duty = float(self.det_cfg.get("gpu_duty", 0.22))
        self.det_duty_min = float(self.det_cfg.get("gpu_duty_min", 0.12))
        self.det_duty_max = float(self.det_cfg.get("gpu_duty_max", 0.32))
        self.wall_intervals_ms: deque[float] = deque(maxlen=240)
        self.wall_last_mono: float | None = None
        self.worker = None
        self.scheduler_thread = None
        self.job_q = None
        self.result_q = None
        self._detector_cleanup_done = False

        super().__init__()
        self.boxes = ProtectiveBoxManager(self.working_width, self.working_height, self.det_cfg.get("box") or {})
        self.bridge = NativeBoxBridge()
        self._install_osd()

    def _preflight(self) -> None:
        super()._preflight()
        required = ("tee", "nvvideoconvert", "appsink", "capsfilter", "nvdsosd")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("Missing detector GStreamer/DeepStream plugins: " + ", ".join(missing))

    def _add_camera(self, index, camera) -> None:
        cid = camera.camera_id
        self.camera_index[cid] = index
        self.capture_requested[cid] = False

        source = self._make("nvurisrcbin", f"v3_source_{index}")
        tee = self._make("tee", f"v3_detect_tee_{index}")
        display_queue = self._make("queue", f"v3_source_queue_{index}")
        infer_queue = self._make("queue", f"v3_detect_queue_{index}")
        converter = self._make("nvvideoconvert", f"v3_detect_convert_{index}")
        capsfilter = self._make("capsfilter", f"v3_detect_caps_{index}")
        appsink = self._make("appsink", f"v3_detect_sink_{index}")

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 0)
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

        self._configure_latest_queue(display_queue)
        self._configure_latest_queue(infer_queue)
        self._set_if(converter, "gpu-id", self.gpu_id)
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={self.capture_width},height={self.capture_height},pixel-aspect-ratio=1/1"
            ),
        )
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        self._set_if(appsink, "enable-last-sample", False)
        self._set_if(appsink, "wait-on-eos", False)

        for element in (source, tee, display_queue, infer_queue, converter, capsfilter, appsink):
            self.pipeline.add(element)

        mux_pad = self._request_mux_pad(index)
        if display_queue.get_static_pad("src").link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display queue -> nvstreammux failed")

        tee_display = tee.request_pad_simple("src_%u")
        tee_infer = tee.request_pad_simple("src_%u")
        if tee_display is None or tee_infer is None:
            raise RuntimeError(f"{cid}: tee request pad failed")
        if tee_display.link(display_queue.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> display queue failed")
        if tee_infer.link(infer_queue.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> detector queue failed")
        self.tee_request_pads.extend([(tee, tee_display), (tee, tee_infer)])

        if not infer_queue.link(converter) or not converter.link(capsfilter) or not capsfilter.link(appsink):
            raise RuntimeError(f"{cid}: detector capture branch link failed")
        infer_queue.get_static_pad("src").add_probe(self.Gst.PadProbeType.BUFFER, self._infer_gate_probe, cid)
        appsink.connect("new-sample", self._on_infer_sample, cid)
        source.connect("pad-added", self._source_to_tee, tee, cid)
        display_queue.get_static_pad("src").add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)

        self.sources[cid] = source
        self.queues[cid] = display_queue

    def _source_to_tee(self, _source, pad, tee, camera_id: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        if not str(caps.get_structure(0).get_name()).startswith("video/"):
            return
        sink = tee.get_static_pad("sink")
        if sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"V3_RFDETR {camera_id} source -> tee failed: {result}", file=sys.stderr, flush=True)

    def _infer_gate_probe(self, _pad, _info, camera_id: str):
        with self.capture_lock:
            if not self.capture_requested.get(camera_id, False):
                return self.Gst.PadProbeReturn.DROP
            self.capture_requested[camera_id] = False
        return self.Gst.PadProbeReturn.OK

    def _on_infer_sample(self, sink, camera_id: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK
        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.OK
        try:
            needed = width * height * 4
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
            frame = raw.reshape((height, width, 4))[..., :3].copy()
        finally:
            buffer.unmap(mapped)
        self.mailbox.put(camera_id, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _install_osd(self) -> None:
        if not self.wall_queue.unlink(self.sink):
            raise RuntimeError("could not detach camera-core sink for RF-DETR OSD")
        convert = self._make("nvvideoconvert", "v3_detect_wall_convert")
        caps = self._make("capsfilter", "v3_detect_wall_caps")
        osd = self._make("nvdsosd", "v3_detect_osd")
        self._set_if(convert, "gpu-id", self.gpu_id)
        caps.set_property("caps", self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", self.gpu_id)
        for element in (convert, caps, osd):
            self.pipeline.add(element)
        if not self.wall_queue.link(convert) or not convert.link(caps) or not caps.link(osd) or not osd.link(self.sink):
            raise RuntimeError("failed RF-DETR OSD display chain")
        self.mux.get_static_pad("src").add_probe(self.Gst.PadProbeType.BUFFER, self._inject_boxes_probe)
        osd.get_static_pad("src").add_probe(self.Gst.PadProbeType.BUFFER, self._wall_probe)
        self.osd = osd

    def _inject_boxes_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        added = 0
        for camera_id, source_id in self.camera_index.items():
            rows = self.boxes.render(camera_id, now)
            if rows:
                result = self.bridge.add_person_boxes(buffer, source_id, rows)
                if result > 0:
                    added += result
        with self.det_lock:
            self.meta_boxes += added
        return self.Gst.PadProbeReturn.OK

    def _wall_probe(self, _pad, _info):
        now = time.monotonic()
        if self.wall_last_mono is not None:
            dt = (now - self.wall_last_mono) * 1000.0
            if 1.0 < dt < 1000.0:
                self.wall_intervals_ms.append(dt)
        self.wall_last_mono = now
        return self.Gst.PadProbeReturn.OK

    def _request_group(self, camera_ids: list[str]) -> None:
        with self.capture_lock:
            for camera_id in camera_ids:
                self.capture_requested[camera_id] = True

    def _clear_requests(self) -> None:
        with self.capture_lock:
            for camera_id in self.capture_requested:
                self.capture_requested[camera_id] = False

    def _scaled_detections(self, rows):
        sx = self.working_width / float(self.capture_width)
        sy = self.working_height / float(self.capture_height)
        output = []
        for coords, conf in rows:
            x1, y1, x2, y2 = coords
            output.append(((x1 * sx, y1 * sy, x2 * sx, y2 * sy), float(conf)))
        return output

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        startup_timeout = float(self.det_cfg.get("startup_timeout_sec", 90))
        try:
            ready = self.result_q.get(timeout=startup_timeout)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "RF-DETR-S startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "RF-DETR-S worker failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "V3_RFDETR ready "
            f"model={ready.get('model')} device={ready.get('device')} cuda={ready.get('cuda')} "
            f"micro_batch={self.micro_batch} capture={self.capture_width}x{self.capture_height} "
            "protective_full_body_envelope=1 raw_boxes_preserved=1",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        groups = [ids[i : i + self.micro_batch] for i in range(0, len(ids), self.micro_batch)]
        versions = {cid: 0 for cid in ids}
        group_index = 0
        result_timeout = float(self.det_cfg.get("result_timeout_sec", 12))

        while not self.det_stop.is_set():
            group = groups[group_index % len(groups)]
            group_index += 1
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=1.5)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.10)
                continue

            frames = []
            captured = []
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                frames.append(frame)
                captured.append(captured_t)
            self._clear_requests()

            try:
                self.job_q.put({"cameras": group, "frames": frames, "captured": captured}, timeout=0.5)
                result = self.result_q.get(timeout=result_timeout)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "RF-DETR-S result timeout"
                self.det_stop.wait(0.25)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "RF-DETR-S fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "RF-DETR-S batch error")
                self.det_stop.wait(0.50)
                continue
            if result.get("type") != "result":
                continue

            counts = {}
            for cid, captured_t in zip(result["cameras"], result["captured"]):
                detections = self._scaled_detections(result["boxes"].get(cid, []))
                counts[cid] = len(detections)
                self.boxes.update(cid, captured_t, detections)

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(group)
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                duty = max(self.det_duty_min, min(self.det_duty_max, self.det_duty))
                self.det_error = ""
            active = batch_ms / 1000.0
            idle = max(0.03, active * (1.0 / max(0.05, duty) - 1.0))
            self.det_stop.wait(min(2.0, idle))

    @staticmethod
    def _p95(values: deque[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        p95 = self._p95(self.wall_intervals_ms)
        slow_ms = float(self.det_cfg.get("wall_p95_slow_ms", 72))
        fast_ms = float(self.det_cfg.get("wall_p95_fast_ms", 60))
        with self.det_lock:
            if p95 is not None:
                if p95 > slow_ms:
                    self.det_duty = max(self.det_duty_min, self.det_duty - 0.025)
                elif p95 < fast_ms and self.det_ready:
                    self.det_duty = min(self.det_duty_max, self.det_duty + 0.010)
            calls = self.det_calls
            inputs = self.det_inputs
            batch_ms = self.det_batch_ms
            counts = dict(self.det_counts)
            meta = self.meta_boxes
            duty = self.det_duty
            ready = self.det_ready
            error = self.det_error
            timeouts = self.capture_timeouts
        count_text = " ".join(f"{cid}:{counts.get(cid, 0)}" for cid in self.camera_index)
        wall = "?" if p95 is None else f"{p95:.1f}ms"
        print(
            "V3_RFDETR_STATS "
            f"ready={int(ready)} calls={calls} inputs={inputs} micro_batch={self.micro_batch} "
            f"batch={batch_ms:.1f}ms duty={duty:.0%} wall_p95={wall} meta_boxes={meta} "
            f"timeouts={timeouts} persons=[{count_text}]" + (f" error={error}" if error else ""),
            flush=True,
        )
        return keep

    def run(self) -> int:
        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.worker = ctx.Process(target=_rfdetr_worker, args=(self.job_q, self.result_q, self.det_cfg), daemon=True)
        self.worker.start()
        self.scheduler_thread = threading.Thread(target=self._scheduler, name="vision-v3-rfdetr-scheduler", daemon=True)
        self.scheduler_thread.start()
        print(
            "V3_RFDETR starting: six-camera GPU-native display stays hot; RF-DETR-S runs as a ticketed async side path",
            flush=True,
        )
        return super().run()

    def stop(self) -> None:
        if not self._detector_cleanup_done:
            self._detector_cleanup_done = True
            self.det_stop.set()
            self._clear_requests()
            self.mailbox.close()
            if self.job_q is not None:
                try:
                    self.job_q.put_nowait(None)
                except Exception:
                    pass
            if self.scheduler_thread is not None:
                self.scheduler_thread.join(timeout=2.0)
            if self.worker is not None:
                self.worker.join(timeout=3.0)
                if self.worker.is_alive():
                    self.worker.terminate()
                    self.worker.join(timeout=1.0)
            for tee, pad in self.tee_request_pads:
                try:
                    tee.release_request_pad(pad)
                except Exception:
                    pass
        super().stop()


def main() -> int:
    try:
        return SixCameraRFDETR().run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"V3_RFDETR_FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
