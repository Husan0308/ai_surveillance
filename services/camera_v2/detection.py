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

from .native_bridge import NativeMetaBridge
from .secure import SecureCameraWallV2

ROOT = Path(__file__).resolve().parents[2]
MODEL_SPEC = os.environ.get("CAMERA_V2_YOLO_MODEL", "yolo26m.pt")
INFER_WIDTH = max(320, int(os.environ.get("CAMERA_V2_DETECT_WIDTH", "512")))
INFER_HEIGHT = max(192, int(os.environ.get("CAMERA_V2_DETECT_HEIGHT", "288")))
MICRO_BATCH = max(1, min(3, int(os.environ.get("CAMERA_V2_MICRO_BATCH", "2"))))
CONF = float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.20"))
IOU = float(os.environ.get("CAMERA_V2_DETECT_IOU", "0.55"))
MAX_DET = max(5, int(os.environ.get("CAMERA_V2_MAX_DET", "30")))


def _resolve_model(spec: str) -> str:
    p = Path(spec)
    if p.is_file():
        return str(p)
    p = ROOT / spec
    return str(p) if p.is_file() else spec


def _yolo_worker(job_q, result_q) -> None:
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

        # Let the camera wall settle before model loading/warmup touches the GPU.
        time.sleep(float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "3.0")))
        model_path = _resolve_model(MODEL_SPEC)
        model = YOLO(model_path)
        kwargs = {
            "imgsz": (INFER_HEIGHT, INFER_WIDTH),
            "rect": True,
            "classes": [0],
            "conf": CONF,
            "iou": IOU,
            "max_det": MAX_DET,
            "device": "cuda:0",
            "verbose": False,
            "stream": False,
        }
        warm = [np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8) for _ in range(MICRO_BATCH)]
        with torch.inference_mode():
            model.predict(source=warm, **kwargs)

        result_q.put(
            {
                "type": "ready",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": model_path,
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    predictions = model.predict(source=job["frames"], **kwargs)
                ended = time.monotonic()
                output = {}
                for cid, prediction in zip(job["cameras"], predictions):
                    boxes = getattr(prediction, "boxes", None)
                    rows = []
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().tolist()
                        confs = boxes.conf.detach().cpu().tolist()
                        for coords, score in zip(xyxy, confs):
                            rows.append(([float(v) for v in coords], float(score)))
                    output[cid] = rows
                result_q.put(
                    {
                        "type": "result",
                        "cameras": job["cameras"],
                        "captured": job["captured"],
                        "boxes": output,
                        "batch_ms": (ended - started) * 1000.0,
                    }
                )
            except BaseException as exc:
                result_q.put({"type": "batch_error", "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})


class FreshFrameMailbox:
    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.rows: dict[str, tuple[int, float, np.ndarray]] = {}
        self.versions: dict[str, int] = {}
        self.closed = False

    def put(self, cid: str, captured: float, frame: np.ndarray) -> None:
        with self.cv:
            version = self.versions.get(cid, 0) + 1
            self.versions[cid] = version
            self.rows[cid] = (version, captured, frame)
            self.cv.notify_all()

    def wait_group(self, cids: list[str], old: dict[str, int], timeout: float):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not self.closed:
                if all(cid in self.rows and self.rows[cid][0] > old.get(cid, 0) for cid in cids):
                    return [self.rows[cid] for cid in cids]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cv.wait(remaining)
        return None

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()


def _xyxy_to_state(box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5, max(2.0, x2 - x1), max(2.0, y2 - y1))


def _state_to_xyxy(cx: float, cy: float, w: float, h: float):
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MotionTrack:
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float
    vy: float
    vw: float
    vh: float
    last_det_t: float
    confidence: float


class SmoothBoxManager:
    """Local per-camera box stabilizer; no identity/ReID semantics.

    Detection timestamps are preserved. When YOLO returns later, the box state is
    extrapolated from the original capture time to the current display time. This
    removes most detector-latency trailing without touching camera pixels.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        self.lock = threading.RLock()
        self.tracks: dict[str, dict[int, MotionTrack]] = {}
        self.next_id = 1
        self.side_margin = float(os.environ.get("CAMERA_V2_BOX_SIDE_MARGIN", "0.08"))
        self.top_margin = float(os.environ.get("CAMERA_V2_BOX_TOP_MARGIN", "0.05"))
        self.bottom_margin = float(os.environ.get("CAMERA_V2_BOX_BOTTOM_MARGIN", "0.08"))
        self.max_age = float(os.environ.get("CAMERA_V2_BOX_MAX_AGE", "1.8"))
        self.max_predict = float(os.environ.get("CAMERA_V2_BOX_MAX_PREDICT", "0.75"))

    def _guard_box(self, box):
        x1, y1, x2, y2 = box
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        x1 -= w * self.side_margin
        x2 += w * self.side_margin
        y1 -= h * self.top_margin
        y2 += h * self.bottom_margin
        return (
            max(0.0, x1),
            max(0.0, y1),
            min(self.width - 1.0, x2),
            min(self.height - 1.0, y2),
        )

    def _predict(self, track: MotionTrack, when: float):
        dt = min(self.max_predict, max(0.0, when - track.last_det_t))
        # Mild velocity damping prevents the box from shooting ahead after a stop.
        damping = 1.0 / (1.0 + 0.75 * dt)
        cx = track.cx + track.vx * dt * damping
        cy = track.cy + track.vy * dt * damping
        w = max(8.0, track.w + track.vw * dt * 0.35)
        h = max(16.0, track.h + track.vh * dt * 0.35)
        x1, y1, x2, y2 = _state_to_xyxy(cx, cy, w, h)
        shift_x = 0.0
        shift_y = 0.0
        if x1 < 0:
            shift_x = -x1
        elif x2 > self.width - 1:
            shift_x = (self.width - 1) - x2
        if y1 < 0:
            shift_y = -y1
        elif y2 > self.height - 1:
            shift_y = (self.height - 1) - y2
        return (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)

    def update(self, cid: str, captured_t: float, detections: list[tuple[tuple[float, float, float, float], float]]) -> None:
        guarded = [(self._guard_box(box), conf) for box, conf in detections]
        with self.lock:
            current = self.tracks.setdefault(cid, {})
            track_ids = list(current)
            candidates = []
            for tid in track_ids:
                track = current[tid]
                pred = self._predict(track, captured_t)
                pcx, pcy, pw, ph = _xyxy_to_state(pred)
                for di, (box, _conf) in enumerate(guarded):
                    dcx, dcy, dw, dh = _xyxy_to_state(box)
                    dist = math.hypot(dcx - pcx, dcy - pcy) / max(30.0, math.hypot(pw, ph))
                    score = _iou(pred, box) * 0.75 + max(0.0, 1.0 - dist) * 0.25
                    if score >= 0.12:
                        candidates.append((score, tid, di))
            candidates.sort(reverse=True)
            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            matches = []
            for score, tid, di in candidates:
                if tid in used_tracks or di in used_dets:
                    continue
                used_tracks.add(tid)
                used_dets.add(di)
                matches.append((tid, di))

            for tid, di in matches:
                track = current[tid]
                box, conf = guarded[di]
                mcx, mcy, mw, mh = _xyxy_to_state(box)
                dt = max(0.05, captured_t - track.last_det_t)
                pred_box = self._predict(track, captured_t)
                pcx, pcy, pw, ph = _xyxy_to_state(pred_box)

                measured_vx = (mcx - track.cx) / dt
                measured_vy = (mcy - track.cy) / dt
                max_vx = self.width * 0.90
                max_vy = self.height * 0.90
                measured_vx = max(-max_vx, min(max_vx, measured_vx))
                measured_vy = max(-max_vy, min(max_vy, measured_vy))

                track.vx = track.vx * 0.55 + measured_vx * 0.45
                track.vy = track.vy * 0.55 + measured_vy * 0.45
                track.vw = track.vw * 0.70 + ((mw - track.w) / dt) * 0.30
                track.vh = track.vh * 0.70 + ((mh - track.h) / dt) * 0.30

                # Position follows quickly. Size expands quickly but shrinks slowly,
                # keeping head/hands/feet inside during transient detector shrinkage.
                pos_alpha = 0.82
                size_alpha_w = 0.75 if mw >= pw else 0.28
                size_alpha_h = 0.78 if mh >= ph else 0.25
                track.cx = pcx + (mcx - pcx) * pos_alpha
                track.cy = pcy + (mcy - pcy) * pos_alpha
                track.w = pw + (mw - pw) * size_alpha_w
                track.h = ph + (mh - ph) * size_alpha_h
                track.last_det_t = captured_t
                track.confidence = conf

            for di, (box, conf) in enumerate(guarded):
                if di in used_dets:
                    continue
                cx, cy, w, h = _xyxy_to_state(box)
                tid = self.next_id
                self.next_id += 1
                current[tid] = MotionTrack(tid, cx, cy, w, h, 0.0, 0.0, 0.0, 0.0, captured_t, conf)

            stale = [tid for tid, track in current.items() if captured_t - track.last_det_t > self.max_age]
            for tid in stale:
                current.pop(tid, None)

    def render(self, cid: str, now: float) -> list[tuple[float, float, float, float, float]]:
        with self.lock:
            current = self.tracks.get(cid, {})
            rows = []
            stale = []
            for tid, track in current.items():
                age = now - track.last_det_t
                if age > self.max_age:
                    stale.append(tid)
                    continue
                x1, y1, x2, y2 = self._predict(track, now)
                if x2 > x1 and y2 > y1:
                    rows.append((x1, y1, x2, y2, track.confidence))
            for tid in stale:
                current.pop(tid, None)
            return rows


class CameraDetectionV2(SecureCameraWallV2):
    def __init__(self) -> None:
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
        self.det_duty = float(os.environ.get("CAMERA_V2_DETECT_GPU_DUTY", "0.24"))
        self.det_duty_min = float(os.environ.get("CAMERA_V2_DETECT_GPU_DUTY_MIN", "0.12"))
        self.det_duty_max = float(os.environ.get("CAMERA_V2_DETECT_GPU_DUTY_MAX", "0.30"))
        self.wall_intervals_ms: deque[float] = deque(maxlen=240)
        self.wall_last_mono: float | None = None
        self.worker = None
        self.scheduler_thread = None
        self.job_q = None
        self.result_q = None

        super().__init__()
        self.boxes = SmoothBoxManager(self.frame_width, self.frame_height)
        self.bridge = NativeMetaBridge()
        self._install_osd_and_meta()

    def _add_camera(self, index, camera) -> None:
        cid = camera.camera_id
        self.camera_index[cid] = index
        self.capture_requested[cid] = False

        source = self._make("nvurisrcbin", f"camera_v2_source_{index}")
        tee = self._make("tee", f"detect_tee_{index}")
        display_queue = self._make("queue", f"camera_v2_queue_{index}")
        infer_queue = self._make("queue", f"detect_queue_{index}")
        converter = self._make("nvvideoconvert", f"detect_convert_{index}")
        capsfilter = self._make("capsfilter", f"detect_caps_{index}")
        appsink = self._make("appsink", f"detect_sink_{index}")

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

        for q in (display_queue, infer_queue):
            self._set_if(q, "max-size-buffers", 1)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)
            self._set_if(q, "silent", True)

        self._set_if(converter, "gpu-id", self.gpu_id)
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INFER_WIDTH},height={INFER_HEIGHT},pixel-aspect-ratio=1/1"
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
            raise RuntimeError(f"{cid}: display queue -> nvstreammux link failed")

        tee_display = tee.request_pad_simple("src_%u")
        tee_infer = tee.request_pad_simple("src_%u")
        if tee_display is None or tee_infer is None:
            raise RuntimeError(f"{cid}: tee could not allocate branch pads")
        if tee_display.link(display_queue.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> display queue failed")
        if tee_infer.link(infer_queue.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> inference queue failed")
        self.tee_request_pads.extend([(tee, tee_display), (tee, tee_infer)])

        if not infer_queue.link(converter) or not converter.link(capsfilter) or not capsfilter.link(appsink):
            raise RuntimeError(f"{cid}: inference branch link failed")
        infer_queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._infer_gate_probe, cid
        )
        appsink.connect("new-sample", self._on_infer_sample, cid)
        source.connect("pad-added", self._source_to_tee, tee, cid)
        display_queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._source_probe, cid
        )

        self.sources[cid] = source
        self.queues[cid] = display_queue

    def _source_to_tee(self, _source, pad, tee, cid: str) -> None:
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
            print(f"CAMERA_DETECT {cid} source -> tee failed: {result}", flush=True)

    def _infer_gate_probe(self, _pad, _info, cid: str):
        with self.capture_lock:
            if not self.capture_requested.get(cid, False):
                return self.Gst.PadProbeReturn.DROP
            self.capture_requested[cid] = False
        return self.Gst.PadProbeReturn.OK

    def _on_infer_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            with self.capture_lock:
                self.capture_requested[cid] = True
            return self.Gst.FlowReturn.OK
        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            with self.capture_lock:
                self.capture_requested[cid] = True
            return self.Gst.FlowReturn.OK
        try:
            needed = width * height * 4
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
            frame = raw.reshape((height, width, 4))[..., :3].copy()
        finally:
            buffer.unmap(mapped)
        self.mailbox.put(cid, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _install_osd_and_meta(self) -> None:
        # Preserve the exact camera baseline through tiler + latest-only wall queue.
        # Only the final wall is converted once for OSD; no per-camera CPU overlay.
        if not self.wall_queue.unlink(self.sink):
            raise RuntimeError("could not detach baseline sink for OSD")
        convert = self._make("nvvideoconvert", "detect_wall_convert")
        caps = self._make("capsfilter", "detect_wall_caps")
        osd = self._make("nvdsosd", "detect_osd")
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
            raise RuntimeError("failed wall queue -> convert -> OSD -> EGL link")

        self.mux.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._inject_boxes_probe
        )
        osd.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._wall_probe
        )
        self.osd = osd

    def _inject_boxes_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        added = 0
        for cid, source_id in self.camera_index.items():
            rows = self.boxes.render(cid, now)
            if rows:
                result = self.bridge.add_boxes(buffer, source_id, rows)
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

    def _request_group(self, cids: list[str]) -> None:
        with self.capture_lock:
            for cid in cids:
                self.capture_requested[cid] = True

    def _clear_requests(self) -> None:
        with self.capture_lock:
            for cid in self.capture_requested:
                self.capture_requested[cid] = False

    def _scaled_detections(self, rows):
        sx = self.frame_width / float(INFER_WIDTH)
        sy = self.frame_height / float(INFER_HEIGHT)
        output = []
        for coords, conf in rows:
            x1, y1, x2, y2 = coords
            output.append(((x1 * sx, y1 * sy, x2 * sx, y2 * sy), conf))
        return output

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO worker failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_DETECT ready: "
            f"YOLO26m micro_batch={MICRO_BATCH} input={INFER_WIDTH}x{INFER_HEIGHT} "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            "box_motion_stabilizer=1 camera_baseline_preserved=1",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        groups = [ids[i : i + MICRO_BATCH] for i in range(0, len(ids), MICRO_BATCH)]
        versions = {cid: 0 for cid in ids}
        group_index = 0

        while not self.det_stop.is_set():
            group = groups[group_index % len(groups)]
            group_index += 1
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=1.2)
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
                captured.append(captured_t)
                frames.append(frame)
            self._clear_requests()

            try:
                self.job_q.put(
                    {"cameras": group, "frames": frames, "captured": captured},
                    timeout=0.5,
                )
                result = self.result_q.get(timeout=8.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO result timeout"
                self.det_stop.wait(0.25)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO batch error")
                self.det_stop.wait(0.50)
                continue
            if result.get("type") != "result":
                continue

            counts = {}
            for cid, captured_t in zip(result["cameras"], result["captured"]):
                dets = self._scaled_detections(result["boxes"].get(cid, []))
                counts[cid] = len(dets)
                self.boxes.update(cid, captured_t, dets)

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(group)
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                duty = max(self.det_duty_min, min(self.det_duty_max, self.det_duty))
                self.det_error = ""

            # No backlog: after every CUDA burst, explicitly leave idle GPU time
            # for NVDEC/tiler/EGL. Short micro-batches are deliberate on Pascal.
            active = batch_ms / 1000.0
            idle = max(0.04, active * (1.0 / max(0.05, duty) - 1.0))
            self.det_stop.wait(min(1.5, idle))

    @staticmethod
    def _p95(values: deque[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        p95 = self._p95(self.wall_intervals_ms)
        with self.det_lock:
            # Adapt only detection duty; never alter the known-good camera path.
            if p95 is not None:
                if p95 > 72.0:
                    self.det_duty = max(self.det_duty_min, self.det_duty - 0.025)
                elif p95 < 60.0 and self.det_ready:
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
            "CAMERA_DETECT "
            f"ready={int(ready)} calls={calls} inputs={inputs} micro_batch={MICRO_BATCH} "
            f"batch={batch_ms:.1f}ms duty_cap={duty:.0%} wall_p95={wall} "
            f"meta_boxes={meta} timeouts={timeouts} persons=[{count_text}]"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        return keep

    def run(self) -> int:
        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.worker = ctx.Process(target=_yolo_worker, args=(self.job_q, self.result_q), daemon=True)
        self.worker.start()
        self.scheduler_thread = threading.Thread(target=self._scheduler, name="camera-v2-yolo-scheduler", daemon=True)
        self.scheduler_thread.start()
        print(
            "CAMERA_DETECT starting: known-good Camera V2 + ticketed YOLO26m sidecar; "
            f"micro_batch={MICRO_BATCH} input={INFER_WIDTH}x{INFER_HEIGHT}; "
            "no ReID, no face, no heatmap, no global tracking",
            flush=True,
        )
        try:
            return super().run()
        finally:
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


def main() -> int:
    return CameraDetectionV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
