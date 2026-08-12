from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    camera_id: str
    frame_id: int
    frame_captured_monotonic: float
    produced_monotonic: float
    boxes: tuple[PersonBox, ...]


class LatestDetectionStore:
    """One detector result per camera. Historical results are never queued."""

    def __init__(self):
        self._lock = threading.Lock()
        self._results: dict[str, DetectionResult] = {}

    def put(self, result: DetectionResult) -> None:
        with self._lock:
            previous = self._results.get(result.camera_id)
            if previous is None or result.frame_id >= previous.frame_id:
                self._results[result.camera_id] = result

    def get(self, camera_id: str) -> DetectionResult | None:
        with self._lock:
            return self._results.get(camera_id)

    def snapshot(self):
        with self._lock:
            return dict(self._results)


def _normalize_imgsz(value, default=(448, 704)):
    value = value if value is not None else default
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return int(value)


def _predict_kwargs(config: dict):
    return {
        "imgsz": _normalize_imgsz(config.get("imgsz", [448, 704])),
        "conf": float(config.get("conf", 0.06)),
        "iou": float(config.get("iou", 0.50)),
        "classes": [0],
        "max_det": max(1, int(config.get("max_det", 50))),
        "device": str(config.get("device", "cuda:0")),
        "quantize": config.get("quantize", 32),
        "verbose": False,
    }


def _roi_predict_kwargs(config: dict, base_kwargs: dict):
    roi_cfg = dict(config.get("roi_second_pass") or {})
    kwargs = dict(base_kwargs)
    kwargs.update({
        "imgsz": _normalize_imgsz(roi_cfg.get("imgsz", [512, 640]), (512, 640)),
        "conf": float(roi_cfg.get("conf", max(0.03, float(base_kwargs["conf"]) * 0.75))),
        "iou": float(roi_cfg.get("iou", base_kwargs["iou"])),
        "max_det": max(1, int(roi_cfg.get("max_det", 20))),
    })
    return kwargs


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b):
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b):
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


def _center_distance(a, b):
    acx = (a[0] + a[2]) * 0.5; acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5; bcy = (b[1] + b[3]) * 0.5
    scale = max(20.0, max(_area(a), _area(b)) ** 0.5)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale


def _deduplicate_boxes(boxes, iou_threshold: float, containment_threshold: float = 0.90, center_threshold: float = 0.30):
    """Confidence-first fusion guard for full-frame + ROI detections.

    The ROI second pass often returns a tighter box around a person already found
    by the full-frame pass. We collapse only very strong geometric duplicates so
    two nearby/overlapping real people remain separate.
    """
    ordered = sorted(boxes, key=lambda item: item[4], reverse=True)
    kept = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if _iou(candidate, existing) >= iou_threshold:
                duplicate = True
                break
            if _containment(candidate, existing) >= containment_threshold and _center_distance(candidate, existing) <= center_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _prediction_boxes(prediction, sx: float, sy: float, offset_x: float = 0.0, offset_y: float = 0.0):
    boxes = []
    pred_boxes = getattr(prediction, "boxes", None)
    if pred_boxes is None or not len(pred_boxes):
        return boxes
    xyxy = pred_boxes.xyxy.detach().cpu().tolist()
    confs = pred_boxes.conf.detach().cpu().tolist()
    for coords, confidence in zip(xyxy, confs):
        boxes.append((
            offset_x + float(coords[0]) * sx,
            offset_y + float(coords[1]) * sy,
            offset_x + float(coords[2]) * sx,
            offset_y + float(coords[3]) * sy,
            float(confidence),
        ))
    return boxes


def _box_center_inside(box, bounds):
    x1, y1, x2, y2 = bounds
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _detector_process_main(input_queue, output_queue, config: dict, model_path: str):
    """Own CUDA inference in a spawned child process."""
    import faulthandler
    faulthandler.enable(all_threads=True)

    try:
        import numpy as np
        import torch
        from ultralytics import YOLO

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        kwargs = _predict_kwargs(config)
        roi_kwargs = _roi_predict_kwargs(config, kwargs)
        device = str(kwargs["device"])
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

        model = YOLO(model_path)
        imgsz = kwargs["imgsz"]
        warm_h, warm_w = (int(imgsz[0]), int(imgsz[1])) if isinstance(imgsz, tuple) else (int(imgsz), int(imgsz))
        model.predict(source=[np.zeros((warm_h, warm_w, 3), dtype=np.uint8)], **kwargs)
        output_queue.put(("ready", {"device": device, "model": model_path}))
    except BaseException as exc:
        try:
            output_queue.put(("startup_error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        return

    duplicate_iou = float(config.get("duplicate_iou", 0.70))
    fusion_containment = float(config.get("fusion_containment", 0.90))
    fusion_center_distance = float(config.get("fusion_center_distance", 0.30))
    roi_cfg = dict(config.get("roi_second_pass") or {})
    roi_enabled = bool(roi_cfg.get("enabled", False))
    roi_trigger_max = max(0, int(roi_cfg.get("trigger_max_full_roi_persons", 1)))

    while True:
        try:
            payload = input_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if payload is None:
            return

        batch_id, entries = payload
        started = time.perf_counter()
        images = [entry["full_image"] for entry in entries]
        try:
            full_predictions = model.predict(source=images, **kwargs)
        except BaseException as exc:
            try:
                output_queue.put(("batch_error", {"batch_id": batch_id, "error": f"{type(exc).__name__}: {exc}"}))
            except Exception:
                pass
            continue

        boxes_by_camera = {}
        for entry, prediction in zip(entries, full_predictions):
            resized_h, resized_w = entry["full_shape"]
            sx = float(entry["source_w"]) / max(1.0, float(resized_w))
            sy = float(entry["source_h"]) / max(1.0, float(resized_h))
            boxes_by_camera[entry["camera_id"]] = _prediction_boxes(prediction, sx, sy)

        # Only difficult configured cameras can enter this path. The second pass
        # runs when the full-frame pass still sees at most N people inside the
        # difficult region. If it already sees two people there, we skip the extra
        # GPU work for that frame.
        roi_entries = []
        if roi_enabled:
            for entry in entries:
                roi = entry.get("roi")
                if not roi:
                    continue
                full_inside = sum(1 for box in boxes_by_camera.get(entry["camera_id"], ()) if _box_center_inside(box, roi["bounds"]))
                if full_inside <= roi_trigger_max:
                    roi_entries.append((entry, roi))

        roi_wall_ms = 0.0
        if roi_entries:
            roi_started = time.perf_counter()
            try:
                roi_predictions = model.predict(source=[roi["image"] for _entry, roi in roi_entries], **roi_kwargs)
                for (entry, roi), prediction in zip(roi_entries, roi_predictions):
                    rx1, ry1, rx2, ry2 = roi["bounds"]
                    roi_h = max(1.0, float(ry2 - ry1)); roi_w = max(1.0, float(rx2 - rx1))
                    resized_h, resized_w = roi["shape"]
                    sx = roi_w / max(1.0, float(resized_w))
                    sy = roi_h / max(1.0, float(resized_h))
                    boxes_by_camera.setdefault(entry["camera_id"], []).extend(
                        _prediction_boxes(prediction, sx, sy, float(rx1), float(ry1))
                    )
            except BaseException as exc:
                # Full-frame detections remain valid if the optional ROI pass fails.
                try:
                    output_queue.put(("roi_error", {"batch_id": batch_id, "error": f"{type(exc).__name__}: {exc}"}))
                except Exception:
                    pass
            roi_wall_ms = (time.perf_counter() - roi_started) * 1000.0

        produced = time.monotonic()
        result_entries = []
        total_boxes = 0
        for entry in entries:
            boxes = _deduplicate_boxes(
                boxes_by_camera.get(entry["camera_id"], []),
                duplicate_iou,
                fusion_containment,
                fusion_center_distance,
            )
            total_boxes += len(boxes)
            result_entries.append((
                entry["camera_id"], entry["frame_id"], entry["captured_mono"], produced, boxes
            ))

        wall_ms = (time.perf_counter() - started) * 1000.0
        try:
            output_queue.put(("result", {
                "batch_id": batch_id,
                "wall_ms": wall_ms,
                "roi_wall_ms": roi_wall_ms,
                "roi_inputs": len(roi_entries),
                "inputs": len(entries),
                "detections": total_boxes,
                "entries": result_entries,
            }))
        except (BrokenPipeError, EOFError, OSError):
            return


class YoloDetectorWorker:
    """Latest-only detector bridge with one in-flight batch."""

    def __init__(self, frame_stores, config: dict, project_root: Path):
        self.frame_stores = dict(frame_stores)
        self.config = dict(config)
        self.project_root = Path(project_root)
        self.camera_ids = sorted(self.frame_stores)
        self.batch_size = max(1, min(len(self.camera_ids) or 1, int(self.config.get("batch_size", 3))))
        self.poll_interval = max(0.002, float(self.config.get("poll_interval_ms", 5.0)) / 1000.0)

        raw_imgsz = _normalize_imgsz(self.config.get("imgsz", [448, 704]))
        if isinstance(raw_imgsz, tuple):
            self.input_h, self.input_w = int(raw_imgsz[0]), int(raw_imgsz[1])
        else:
            self.input_h = self.input_w = int(raw_imgsz)

        self.roi_cfg = dict(self.config.get("roi_second_pass") or {})
        raw_roi_imgsz = _normalize_imgsz(self.roi_cfg.get("imgsz", [512, 640]), (512, 640))
        if isinstance(raw_roi_imgsz, tuple):
            self.roi_h, self.roi_w = int(raw_roi_imgsz[0]), int(raw_roi_imgsz[1])
        else:
            self.roi_h = self.roi_w = int(raw_roi_imgsz)
        self.roi_cameras = dict(self.roi_cfg.get("cameras") or {})
        self.roi_every_n = max(1, int(self.roi_cfg.get("every_n", 2)))
        self._roi_seen = {cid: 0 for cid in self.camera_ids}

        model_value = str(self.config.get("model", "models/yolo26m.pt"))
        model_path = Path(model_value).expanduser()
        self.model_path = model_path if model_path.is_absolute() else (self.project_root / model_path)
        self.start_delay_sec = max(0.0, float(self.config.get("start_delay_sec", 2.0)))

        self.results = LatestDetectionStore()
        self._stop = threading.Event()
        self._thread = None
        self._process = None
        self._ctx = mp.get_context("spawn")
        self._input_queue = None
        self._output_queue = None
        self._last_versions = {cid: 0 for cid in self.camera_ids}
        self._cursor = 0
        self._batch_id = 0
        self._inflight_batch_id: int | None = None
        self._lock = threading.Lock()
        self._started_mono = time.monotonic()
        self._ready = False
        self._submitted = 0
        self._batches = 0
        self._inputs = 0
        self._detections = 0
        self._roi_inputs = 0
        self._last_batch_ms = 0.0
        self._last_roi_ms = 0.0
        self._last_prepare_ms = 0.0
        self._last_error = ""
        self._per_camera_inputs = {cid: 0 for cid in self.camera_ids}
        self._per_camera_last_frame_id = {cid: 0 for cid in self.camera_ids}
        self._per_camera_last_detection_mono = {cid: 0.0 for cid in self.camera_ids}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_bridge, name="core-v1-yolo-bridge", daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._input_queue is not None:
            try:
                self._input_queue.put_nowait(None)
            except Exception:
                pass

    def join(self, timeout=10):
        deadline = time.monotonic() + timeout
        if self._thread:
            self._thread.join(max(0.0, deadline - time.monotonic()))
        process = self._process
        if process is not None and process.is_alive():
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate(); process.join(2.0)
        return (not self._thread or not self._thread.is_alive()) and (process is None or not process.is_alive())

    def _spawn_process(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self._input_queue = self._ctx.Queue(maxsize=1)
        self._output_queue = self._ctx.Queue(maxsize=16)
        self._process = self._ctx.Process(
            target=_detector_process_main,
            name="core-v1-yolo-cuda",
            args=(self._input_queue, self._output_queue, self.config, str(self.model_path)),
            daemon=False,
        )
        self._process.start()
        log.info("CORE_V1_YOLO_PROCESS_STARTED pid=%s start_method=spawn", self._process.pid)

    def _select_latest_batch(self):
        if not self.camera_ids:
            return []
        selected = []
        n = len(self.camera_ids)
        scanned = 0
        while scanned < n and len(selected) < self.batch_size:
            cid = self.camera_ids[(self._cursor + scanned) % n]
            frame, version = self.frame_stores[cid].get()
            if frame is not None and version > self._last_versions[cid]:
                newest, newest_version = self.frame_stores[cid].get()
                if newest is not None and newest_version > self._last_versions[cid]:
                    selected.append((cid, newest, newest_version))
            scanned += 1
        self._cursor = (self._cursor + max(1, scanned)) % n
        return selected

    def _prepare_roi(self, cid, frame):
        camera_cfg = dict(self.roi_cameras.get(cid) or {})
        if not camera_cfg or not bool(self.roi_cfg.get("enabled", False)):
            return None
        self._roi_seen[cid] += 1
        if self._roi_seen[cid] % self.roi_every_n:
            return None

        import cv2
        values = camera_cfg.get("box") or camera_cfg.get("roi")
        if not isinstance(values, (list, tuple)) or len(values) < 4:
            return None
        nx1, ny1, nx2, ny2 = [max(0.0, min(1.0, float(v))) for v in values[:4]]
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        x1 = max(0, min(frame.width - 1, int(round(nx1 * frame.width))))
        y1 = max(0, min(frame.height - 1, int(round(ny1 * frame.height))))
        x2 = max(x1 + 1, min(frame.width, int(round(nx2 * frame.width))))
        y2 = max(y1 + 1, min(frame.height, int(round(ny2 * frame.height))))
        crop = frame.image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (self.roi_w, self.roi_h), interpolation=cv2.INTER_LINEAR)
        return {"image": resized, "bounds": (x1, y1, x2, y2), "shape": (self.roi_h, self.roi_w)}

    def _prepare_payload(self, selected):
        import cv2
        started = time.perf_counter()
        entries = []
        for cid, frame, _version in selected:
            full = cv2.resize(frame.image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            entries.append({
                "camera_id": cid,
                "frame_id": int(frame.frame_id),
                "captured_mono": float(frame.captured_monotonic),
                "source_w": int(frame.width),
                "source_h": int(frame.height),
                "full_shape": (self.input_h, self.input_w),
                "full_image": full,
                "roi": self._prepare_roi(cid, frame),
            })
        with self._lock:
            self._last_prepare_ms = (time.perf_counter() - started) * 1000.0
        return entries

    def _submit_if_idle(self):
        if self._inflight_batch_id is not None:
            return
        selected = self._select_latest_batch()
        if not selected:
            return
        entries = self._prepare_payload(selected)
        self._batch_id += 1
        batch_id = self._batch_id
        try:
            self._input_queue.put_nowait((batch_id, entries))
        except queue.Full:
            return
        for cid, _frame, version in selected:
            self._last_versions[cid] = version
        self._inflight_batch_id = batch_id
        with self._lock:
            self._submitted += len(entries)

    def _drain_outputs(self):
        if self._output_queue is None:
            return
        while True:
            try:
                kind, payload = self._output_queue.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return

            if kind == "ready":
                with self._lock:
                    self._ready = True; self._last_error = ""
                log.info("CORE_V1_YOLO_READY process_pid=%s device=%s model=%s", self._process.pid if self._process else None, payload.get("device"), payload.get("model"))
                continue

            if kind in {"startup_error", "batch_error", "roi_error"}:
                error = payload if isinstance(payload, str) else payload.get("error", str(payload))
                if kind != "roi_error" and isinstance(payload, dict) and payload.get("batch_id") == self._inflight_batch_id:
                    self._inflight_batch_id = None
                with self._lock:
                    self._last_error = error
                log.error("CORE_V1_YOLO_%s %s", kind.upper(), error)
                continue

            if kind != "result":
                continue
            if payload.get("batch_id") == self._inflight_batch_id:
                self._inflight_batch_id = None

            for cid, frame_id, captured_mono, produced_mono, raw_boxes in payload["entries"]:
                boxes = tuple(PersonBox(*map(float, box)) for box in raw_boxes)
                self.results.put(DetectionResult(
                    camera_id=str(cid), frame_id=int(frame_id),
                    frame_captured_monotonic=float(captured_mono),
                    produced_monotonic=float(produced_mono), boxes=boxes,
                ))
                with self._lock:
                    self._per_camera_inputs[str(cid)] += 1
                    self._per_camera_last_frame_id[str(cid)] = int(frame_id)
                    self._per_camera_last_detection_mono[str(cid)] = float(produced_mono)

            with self._lock:
                self._batches += 1
                self._inputs += int(payload["inputs"])
                self._detections += int(payload["detections"])
                self._roi_inputs += int(payload.get("roi_inputs", 0))
                self._last_batch_ms = float(payload["wall_ms"])
                self._last_roi_ms = float(payload.get("roi_wall_ms", 0.0))
                self._last_error = ""

    def _run_bridge(self):
        if self.start_delay_sec and self._stop.wait(self.start_delay_sec):
            return
        try:
            self._spawn_process()
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            log.exception("CORE_V1_YOLO_PROCESS_START_FAILED")
            return

        while not self._stop.is_set():
            self._drain_outputs()
            process = self._process
            if process is not None and not process.is_alive():
                exitcode = process.exitcode
                with self._lock:
                    self._ready = False
                    if not self._last_error:
                        self._last_error = f"detector process exited unexpectedly with exitcode={exitcode}"
                log.error("CORE_V1_YOLO_PROCESS_EXITED exitcode=%s camera_core_continues=true", exitcode)
                break
            if self._ready:
                self._submit_if_idle()
            self._stop.wait(self.poll_interval)
        self._drain_outputs()

    def metrics(self):
        now = time.monotonic(); process = self._process
        with self._lock:
            elapsed = max(0.001, now - self._started_mono)
            cameras = {}
            for cid in self.camera_ids:
                last = self._per_camera_last_detection_mono[cid]
                cameras[cid] = {
                    "inputs": self._per_camera_inputs[cid],
                    "input_rate": self._per_camera_inputs[cid] / elapsed,
                    "last_frame_id": self._per_camera_last_frame_id[cid],
                    "observation_age_ms": ((now - last) * 1000.0) if last else None,
                }
            return {
                "ready": self._ready,
                "process_alive": bool(process and process.is_alive()),
                "process_pid": process.pid if process else None,
                "process_exitcode": process.exitcode if process and not process.is_alive() else None,
                "start_method": "spawn",
                "model": str(self.model_path),
                "device": str(self.config.get("device", "cuda:0")),
                "quantize": self.config.get("quantize", 32),
                "batch_size": self.batch_size,
                "inflight_batch_id": self._inflight_batch_id,
                "submitted_camera_inputs": self._submitted,
                "batches": self._batches,
                "batch_rate": self._batches / elapsed,
                "camera_inputs": self._inputs,
                "camera_input_rate": self._inputs / elapsed,
                "detections": self._detections,
                "roi_inputs": self._roi_inputs,
                "last_prepare_ms": self._last_prepare_ms,
                "last_batch_ms": self._last_batch_ms,
                "last_roi_ms": self._last_roi_ms,
                "last_error": self._last_error,
                "cameras": cameras,
            }
