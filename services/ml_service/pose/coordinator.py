from __future__ import annotations

from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PosePerson:
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[PoseKeypoint, ...]


@dataclass(frozen=True, slots=True)
class PoseResult:
    camera_id: str
    frame_id: int
    frame_captured_monotonic: float
    produced_monotonic: float
    people: tuple[PosePerson, ...]


class PoseCoordinator:
    """Optional latest-only top-down pose side path.

    The primary detector/tracker remains authoritative. Pose only consumes exact
    detector frames from the bounded frame history, runs at a configurable
    cadence, and never queues old work.
    """

    def __init__(self, frame_stores, detections, config: dict | None = None):
        self.frame_stores = dict(frame_stores)
        self.detections = detections
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.model_name = str(self.config.get("model", "yolo11n-pose.pt"))
        self.device = str(self.config.get("device", "cuda:0"))
        self.imgsz = int(self.config.get("imgsz", 320))
        self.conf = float(self.config.get("conf", 0.20))
        self.every_n = max(1, int(self.config.get("every_n", 3)))
        self.max_people = max(1, int(self.config.get("max_people", 6)))
        self.max_frame_age_ms = max(0.0, float(self.config.get("max_frame_age_ms", 700)))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._model = None
        self._last_frame = {cid: -1 for cid in self.frame_stores}
        self._seen = {cid: 0 for cid in self.frame_stores}
        self._results: dict[str, PoseResult] = {}
        self._processed = 0
        self._frame_misses = 0
        self._stale_skips = 0
        self._errors = 0
        self._last_inference_ms = 0.0
        self._last_error = ""

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pose-sidepath", daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=6):
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self):
        with self._lock:
            return dict(self._results)

    def metrics(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "ready": self._model is not None,
                "processed": self._processed,
                "frame_misses": self._frame_misses,
                "stale_skips": self._stale_skips,
                "errors": self._errors,
                "last_inference_ms": self._last_inference_ms,
                "last_error": self._last_error,
                "model": self.model_name,
                "device": self.device,
            }

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO
        self._model = YOLO(self.model_name)
        return self._model

    @staticmethod
    def _crop(frame, box):
        h, w = frame.image.shape[:2]
        x1 = max(0, min(w - 1, int(box.x1)))
        y1 = max(0, min(h - 1, int(box.y1)))
        x2 = max(x1 + 1, min(w, int(box.x2)))
        y2 = max(y1 + 1, min(h, int(box.y2)))
        return frame.image[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _infer(self, frame, boxes):
        model = self._ensure_model()
        crops = []
        offsets = []
        source_boxes = []
        for box in sorted(boxes, key=lambda b: b.confidence, reverse=True)[: self.max_people]:
            crop, bounds = self._crop(frame, box)
            if crop.size == 0:
                continue
            crops.append(crop)
            offsets.append((bounds[0], bounds[1]))
            source_boxes.append(box)
        if not crops:
            return ()
        started = time.perf_counter()
        predictions = model.predict(
            source=crops,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        self._last_inference_ms = (time.perf_counter() - started) * 1000.0
        people = []
        for source_box, offset, pred in zip(source_boxes, offsets, predictions):
            keypoints = getattr(pred, "keypoints", None)
            if keypoints is None or getattr(keypoints, "xy", None) is None or len(keypoints.xy) == 0:
                continue
            xy = keypoints.xy[0].detach().cpu().tolist()
            conf_tensor = getattr(keypoints, "conf", None)
            confs = conf_tensor[0].detach().cpu().tolist() if conf_tensor is not None else [1.0] * len(xy)
            ox, oy = offset
            points = tuple(
                PoseKeypoint(float(p[0]) + ox, float(p[1]) + oy, float(c))
                for p, c in zip(xy, confs)
            )
            people.append(PosePerson(
                (float(source_box.x1), float(source_box.y1), float(source_box.x2), float(source_box.y2)),
                float(source_box.confidence),
                points,
            ))
        return tuple(people)

    def _run(self):
        while not self._stop.is_set():
            snapshot = self.detections.snapshot() if self.detections is not None else {}
            did_work = False
            for camera_id, detection in snapshot.items():
                frame_id = int(detection.frame_id)
                if frame_id <= self._last_frame.get(camera_id, -1):
                    continue
                self._last_frame[camera_id] = frame_id
                self._seen[camera_id] = self._seen.get(camera_id, 0) + 1
                if self._seen[camera_id] % self.every_n:
                    continue
                age_ms = max(0.0, (time.monotonic() - float(detection.frame_captured_monotonic)) * 1000.0)
                if self.max_frame_age_ms and age_ms > self.max_frame_age_ms:
                    self._stale_skips += 1
                    continue
                store = self.frame_stores.get(camera_id)
                frame = store.get_frame(frame_id) if store and hasattr(store, "get_frame") else None
                if frame is None:
                    self._frame_misses += 1
                    continue
                try:
                    people = self._infer(frame, detection.boxes)
                    result = PoseResult(
                        camera_id,
                        frame_id,
                        float(detection.frame_captured_monotonic),
                        time.monotonic(),
                        people,
                    )
                    with self._lock:
                        self._results[camera_id] = result
                        self._processed += 1
                        self._last_error = ""
                    did_work = True
                except Exception as exc:
                    with self._lock:
                        self._errors += 1
                        self._last_error = f"{type(exc).__name__}: {exc}"
            if not did_work:
                self._stop.wait(0.02)
