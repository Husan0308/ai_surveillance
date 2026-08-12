from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, asdict
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
    """One detection result per camera. No queue and no tracker state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._results: dict[str, DetectionResult] = {}

    def put(self, result: DetectionResult) -> None:
        with self._lock:
            self._results[result.camera_id] = result

    def get(self, camera_id: str) -> DetectionResult | None:
        with self._lock:
            return self._results.get(camera_id)

    def snapshot(self):
        with self._lock:
            return dict(self._results)


class YoloDetectorWorker:
    """Minimal core-v1 person detector.

    Design rules for this stage:
    - one model and one inference worker only;
    - read only the newest frame from each camera store;
    - never create a frame backlog;
    - batch at most ``batch_size`` cameras;
    - person class only;
    - no tracker, ReID, identity, ROI, prediction or heatmap.
    """

    def __init__(self, frame_stores, config: dict, project_root: Path):
        self.frame_stores = dict(frame_stores)
        self.config = dict(config)
        self.project_root = Path(project_root)
        self.camera_ids = sorted(self.frame_stores)
        self.batch_size = max(1, min(len(self.camera_ids) or 1, int(self.config.get("batch_size", 3))))
        self.batch_interval = max(0.0, float(self.config.get("batch_interval_ms", 20.0)) / 1000.0)
        self.conf = float(self.config.get("conf", 0.15))
        self.iou = float(self.config.get("iou", 0.45))
        self.max_det = max(1, int(self.config.get("max_det", 40)))
        raw_imgsz = self.config.get("imgsz", [384, 640])
        self.imgsz = tuple(int(v) for v in raw_imgsz) if isinstance(raw_imgsz, (list, tuple)) else int(raw_imgsz)
        self.device = str(self.config.get("device", "cuda:0"))
        self.half = bool(self.config.get("half", False))
        model_value = str(self.config.get("model", "models/yolo26m.pt"))
        model_path = Path(model_value).expanduser()
        self.model_path = model_path if model_path.is_absolute() else (self.project_root / model_path)

        self.results = LatestDetectionStore()
        self._stop = threading.Event()
        self._thread = None
        self._model = None
        self._last_versions = {cid: 0 for cid in self.camera_ids}
        self._cursor = 0
        self._lock = threading.Lock()
        self._started_mono = time.monotonic()
        self._batches = 0
        self._inputs = 0
        self._detections = 0
        self._last_batch_ms = 0.0
        self._last_error = ""
        self._per_camera_inputs = {cid: 0 for cid in self.camera_ids}
        self._per_camera_last_frame_id = {cid: 0 for cid in self.camera_ids}
        self._per_camera_last_detection_mono = {cid: 0.0 for cid in self.camera_ids}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="core-v1-yolo", daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=10):
        if self._thread:
            self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        from ultralytics import YOLO
        model = YOLO(str(self.model_path))
        # Warmup with a tiny synthetic image before the realtime loop. This cost
        # is paid once and never blocks camera capture/display threads.
        import numpy as np
        warm = np.zeros((384, 640, 3), dtype=np.uint8)
        model.predict(
            source=[warm],
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        self._model = model
        log.info("CORE_V1_YOLO_READY model=%s device=%s batch=%d imgsz=%s", self.model_path, self.device, self.batch_size, self.imgsz)

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
                # Re-fetch at selection time: the detector always consumes the
                # newest snapshot and never an accumulated queue entry.
                newest, newest_version = self.frame_stores[cid].get()
                if newest is not None and newest_version > self._last_versions[cid]:
                    selected.append((cid, newest, newest_version))
            scanned += 1
        self._cursor = (self._cursor + max(1, scanned)) % n
        return selected

    def _run(self):
        try:
            self._load_model()
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            log.exception("CORE_V1_YOLO_START_FAILED")
            return

        while not self._stop.is_set():
            batch = self._select_latest_batch()
            if not batch:
                self._stop.wait(0.005)
                continue

            images = [item[1].image for item in batch]
            started = time.perf_counter()
            try:
                predictions = self._model.predict(
                    source=images,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    classes=[0],
                    max_det=self.max_det,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                log.exception("CORE_V1_YOLO_BATCH_FAILED cameras=%s", [item[0] for item in batch])
                self._stop.wait(0.25)
                continue

            wall_ms = (time.perf_counter() - started) * 1000.0
            total_boxes = 0
            produced = time.monotonic()
            for (cid, frame, version), prediction in zip(batch, predictions):
                boxes = []
                pred_boxes = getattr(prediction, "boxes", None)
                if pred_boxes is not None and len(pred_boxes):
                    xyxy = pred_boxes.xyxy.detach().cpu().tolist()
                    confs = pred_boxes.conf.detach().cpu().tolist()
                    for coords, confidence in zip(xyxy, confs):
                        boxes.append(PersonBox(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), float(confidence)))
                result = DetectionResult(
                    camera_id=cid,
                    frame_id=frame.frame_id,
                    frame_captured_monotonic=frame.captured_monotonic,
                    produced_monotonic=produced,
                    boxes=tuple(boxes),
                )
                self.results.put(result)
                self._last_versions[cid] = version
                total_boxes += len(boxes)
                with self._lock:
                    self._per_camera_inputs[cid] += 1
                    self._per_camera_last_frame_id[cid] = frame.frame_id
                    self._per_camera_last_detection_mono[cid] = produced

            with self._lock:
                self._batches += 1
                self._inputs += len(batch)
                self._detections += total_boxes
                self._last_batch_ms = wall_ms
                self._last_error = ""

            if self.batch_interval:
                self._stop.wait(self.batch_interval)

    def metrics(self):
        now = time.monotonic()
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
                "ready": self._model is not None,
                "model": str(self.model_path),
                "device": self.device,
                "batch_size": self.batch_size,
                "batches": self._batches,
                "batch_rate": self._batches / elapsed,
                "camera_inputs": self._inputs,
                "camera_input_rate": self._inputs / elapsed,
                "detections": self._detections,
                "last_batch_ms": self._last_batch_ms,
                "last_error": self._last_error,
                "cameras": cameras,
            }
