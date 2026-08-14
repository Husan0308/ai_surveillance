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
    detector frames from bounded frame history, runs at a sparse configurable
    cadence, and never queues old work. In the full-safe profile this worker is
    intentionally CPU-only so it cannot create a second CUDA context beside the
    detector process.
    """

    def __init__(self, frame_stores, detections, config: dict | None = None):
        self.frame_stores = dict(frame_stores)
        self.detections = detections
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.model_name = str(self.config.get("model", "yolo26m-pose.pt"))
        self.device = str(self.config.get("device", "cpu"))
        self.imgsz = int(self.config.get("imgsz", 256))
        self.conf = float(self.config.get("conf", 0.25))
        self.half = bool(self.config.get("half", False)) and self.device.startswith("cuda")
        self.every_n = max(1, int(self.config.get("every_n", 12)))
        self.max_people = max(1, int(self.config.get("max_people", 1)))
        self.max_cameras_per_cycle = max(1, int(self.config.get("max_cameras_per_cycle", 1)))
        self.max_frame_age_ms = max(0.0, float(self.config.get("max_frame_age_ms", 1200)))
        self.poll_sec = max(0.01, float(self.config.get("poll_interval_ms", 25)) / 1000.0)
        self.retry_after_sec = max(1.0, float(self.config.get("retry_after_sec", 30.0)))
        self.torch_cpu_threads = max(1, int(self.config.get("torch_cpu_threads", 1)))

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._model = None
        self._last_model_error_at = 0.0
        self._last_frame = {cid: -1 for cid in self.frame_stores}
        self._seen = {cid: 0 for cid in self.frame_stores}
        self._results: dict[str, PoseResult] = {}
        self._processed = 0
        self._frame_misses = 0
        self._stale_skips = 0
        self._errors = 0
        self._budget_skips = 0
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
                "budget_skips": self._budget_skips,
                "errors": self._errors,
                "last_inference_ms": self._last_inference_ms,
                "last_error": self._last_error,
                "model": self.model_name,
                "device": self.device,
                "half": self.half,
                "every_n": self.every_n,
                "max_people": self.max_people,
                "max_cameras_per_cycle": self.max_cameras_per_cycle,
                "isolation": "detector_independent_sidepath",
                "cuda_context": "none" if not self.device.startswith("cuda") else "separate",
            }

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._last_model_error_at and time.monotonic() - self._last_model_error_at < self.retry_after_sec:
            raise RuntimeError(self._last_error or "pose model retry backoff")

        try:
            if not self.device.startswith("cuda"):
                import torch

                try:
                    torch.set_num_threads(self.torch_cpu_threads)
                except Exception:
                    pass
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            return self._model
        except Exception:
            self._last_model_error_at = time.monotonic()
            raise

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
            half=self.half,
            max_det=1,
            verbose=False,
        )
        self._last_inference_ms = (time.perf_counter() - started) * 1000.0

        people = []
        for source_box, offset, pred in zip(source_boxes, offsets, predictions):
            keypoints = getattr(pred, "keypoints", None)
            if keypoints is None or getattr(keypoints, "xy", None) is None or len(keypoints.xy) == 0:
                continue

            index = 0
            pred_boxes = getattr(pred, "boxes", None)
            if pred_boxes is not None and getattr(pred_boxes, "conf", None) is not None and len(pred_boxes.conf):
                try:
                    index = int(pred_boxes.conf.argmax().item())
                except Exception:
                    index = 0
            if index >= len(keypoints.xy):
                index = 0

            xy = keypoints.xy[index].detach().cpu().tolist()
            conf_tensor = getattr(keypoints, "conf", None)
            confs = (
                conf_tensor[index].detach().cpu().tolist()
                if conf_tensor is not None and len(conf_tensor) > index
                else [1.0] * len(xy)
            )
            ox, oy = offset
            points = tuple(
                PoseKeypoint(float(point[0]) + ox, float(point[1]) + oy, float(confidence))
                for point, confidence in zip(xy, confs)
            )
            pose_confidence = float(source_box.confidence)
            if pred_boxes is not None and getattr(pred_boxes, "conf", None) is not None and len(pred_boxes.conf) > index:
                try:
                    pose_confidence = float(pred_boxes.conf[index].detach().cpu().item())
                except Exception:
                    pass
            people.append(
                PosePerson(
                    (
                        float(source_box.x1),
                        float(source_box.y1),
                        float(source_box.x2),
                        float(source_box.y2),
                    ),
                    pose_confidence,
                    points,
                )
            )
        return tuple(people)

    def _run(self):
        while not self._stop.is_set():
            snapshot = self.detections.snapshot() if self.detections is not None else {}
            work_count = 0

            for camera_id in sorted(snapshot):
                detection = snapshot[camera_id]
                frame_id = int(detection.frame_id)
                if frame_id <= self._last_frame.get(camera_id, -1):
                    continue
                self._last_frame[camera_id] = frame_id
                self._seen[camera_id] = self._seen.get(camera_id, 0) + 1
                if self._seen[camera_id] % self.every_n:
                    continue
                if work_count >= self.max_cameras_per_cycle:
                    self._budget_skips += 1
                    continue

                age_ms = max(
                    0.0,
                    (time.monotonic() - float(detection.frame_captured_monotonic)) * 1000.0,
                )
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
                    work_count += 1
                except Exception as exc:
                    with self._lock:
                        self._errors += 1
                        self._last_error = f"{type(exc).__name__}: {exc}"
                        self._last_model_error_at = time.monotonic()

            self._stop.wait(self.poll_sec)
