from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
from typing import Any

from services.ml_service.app.latest_frame import LatestFrameStore


@dataclass(frozen=True, slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    camera_id: str
    frame_id: int
    captured_monotonic: float
    inferred_monotonic: float
    batch_ms: float
    detections: tuple[Detection, ...]


class DetectionStore:
    """Thread-safe latest detection result per camera."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, DetectionSnapshot] = {}

    def put(self, snapshot: DetectionSnapshot) -> None:
        with self._lock:
            self._rows[snapshot.camera_id] = snapshot

    def get(self, camera_id: str) -> DetectionSnapshot | None:
        with self._lock:
            return self._rows.get(camera_id)

    def payload(self, camera_id: str) -> dict | None:
        snapshot = self.get(camera_id)
        if snapshot is None:
            return None
        now = time.monotonic()
        return {
            "camera_id": snapshot.camera_id,
            "frame_id": snapshot.frame_id,
            "people": len(snapshot.detections),
            "age_ms": max(0.0, (now - snapshot.inferred_monotonic) * 1000.0),
            "source_age_ms": max(0.0, (now - snapshot.captured_monotonic) * 1000.0),
            "batch_ms": snapshot.batch_ms,
            "detections": [asdict(row) for row in snapshot.detections],
        }


@dataclass
class DetectorMetrics:
    state: str = "stopped"
    model: str = ""
    device: str = ""
    batches: int = 0
    images: int = 0
    last_batch_ms: float = 0.0
    average_batch_ms: float = 0.0
    last_error: str = ""


class PersonDetector:
    """One GPU model for all cameras using only their newest frame.

    There is intentionally no tracker, ReID or face-recognition logic here.
    If inference is slower than camera capture, intermediate frames are skipped
    rather than queued so camera ingest and MJPEG output cannot build latency.
    """

    def __init__(self, config, stores: dict[str, LatestFrameStore]) -> None:
        self.config = config
        self.stores = dict(stores)
        self.results = DetectionStore()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._metrics = DetectorMetrics(state="disabled" if not config.enabled else "stopped")
        self._camera_updates: dict[str, int] = {camera_id: 0 for camera_id in self.stores}
        self._camera_started: dict[str, float] = {camera_id: 0.0 for camera_id in self.stores}
        self._camera_last_infer: dict[str, float] = {camera_id: 0.0 for camera_id in self.stores}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="person-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    def metrics(self) -> dict:
        with self._lock:
            payload = asdict(self._metrics)
        payload["enabled"] = self.enabled
        payload["batch_size"] = int(self.config.batch_size)
        payload["target_fps_per_camera"] = float(self.config.target_fps_per_camera)
        payload["imgsz"] = [int(self.config.height), int(self.config.width)]
        return payload

    def camera_metrics(self, camera_id: str) -> dict:
        now = time.monotonic()
        snapshot = self.results.get(camera_id)
        with self._lock:
            updates = int(self._camera_updates.get(camera_id, 0))
            started = float(self._camera_started.get(camera_id, 0.0))
            last_infer = float(self._camera_last_infer.get(camera_id, 0.0))
            state = self._metrics.state
            error = self._metrics.last_error
        elapsed = max(0.001, now - started) if started else 0.0
        return {
            "state": state,
            "people": len(snapshot.detections) if snapshot else 0,
            "fps": updates / elapsed if elapsed else 0.0,
            "age_ms": max(0.0, (now - last_infer) * 1000.0) if last_infer else None,
            "frame_id": snapshot.frame_id if snapshot else 0,
            "last_error": error if state == "error" else "",
        }

    def snapshot_payload(self, camera_id: str) -> dict:
        return {
            "detector": self.metrics(),
            "result": self.results.payload(camera_id),
        }

    def _set_state(self, state: str, error: str = "") -> None:
        with self._lock:
            self._metrics.state = state
            self._metrics.last_error = error

    def _run(self) -> None:
        self._set_state("starting")
        try:
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is unavailable")

            model = YOLO(self.config.model)
            device = str(self.config.device)
            predict_kwargs: dict[str, Any] = {
                "imgsz": (int(self.config.height), int(self.config.width)),
                "classes": [0],
                "conf": float(self.config.confidence),
                "iou": float(self.config.iou),
                "max_det": int(self.config.max_detections),
                "device": device,
                "verbose": False,
                "stream": False,
                "rect": True,
                "half": bool(self.config.half),
            }

            # Warm once with the configured batch shape before touching live frames.
            import numpy as np

            warm = [
                np.zeros((int(self.config.height), int(self.config.width), 3), dtype=np.uint8)
                for _ in range(int(self.config.batch_size))
            ]
            with torch.inference_mode():
                model.predict(source=warm, **predict_kwargs)

            with self._lock:
                self._metrics.model = str(self.config.model)
                self._metrics.device = device
                self._metrics.state = "ready"
                self._metrics.last_error = ""

            print(
                f"[DETECT] ready model={self.config.model} device={device} "
                f"batch={self.config.batch_size} imgsz={self.config.width}x{self.config.height} "
                f"target_fps_per_camera={self.config.target_fps_per_camera}",
                flush=True,
            )

            camera_ids = list(self.stores)
            last_versions = {camera_id: 0 for camera_id in camera_ids}
            next_due = {camera_id: 0.0 for camera_id in camera_ids}
            cursor = 0
            interval = 1.0 / max(0.1, float(self.config.target_fps_per_camera))

            while not self._stop.is_set():
                now = time.monotonic()
                selected: list[tuple[str, object, int]] = []
                checked = 0

                while checked < len(camera_ids) and len(selected) < int(self.config.batch_size):
                    index = (cursor + checked) % len(camera_ids)
                    checked += 1
                    camera_id = camera_ids[index]
                    if now < next_due[camera_id]:
                        continue
                    frame, version = self.stores[camera_id].get()
                    if frame is None or version <= last_versions[camera_id]:
                        continue
                    selected.append((camera_id, frame, version))

                cursor = (cursor + max(1, checked)) % max(1, len(camera_ids))
                if not selected:
                    self._stop.wait(0.005)
                    continue

                frames = [frame.image for _, frame, _ in selected]
                started = time.monotonic()
                with torch.inference_mode():
                    predictions = model.predict(source=frames, **predict_kwargs)
                finished = time.monotonic()
                batch_ms = (finished - started) * 1000.0

                for (camera_id, frame, version), prediction in zip(selected, predictions):
                    rows: list[Detection] = []
                    boxes = getattr(prediction, "boxes", None)
                    if boxes is not None and len(boxes):
                        coords = boxes.xyxy.detach().cpu().tolist()
                        scores = boxes.conf.detach().cpu().tolist()
                        for xyxy, confidence in zip(coords, scores):
                            rows.append(
                                Detection(
                                    xyxy=tuple(float(value) for value in xyxy),
                                    confidence=float(confidence),
                                )
                            )

                    snapshot = DetectionSnapshot(
                        camera_id=camera_id,
                        frame_id=int(frame.frame_id),
                        captured_monotonic=float(frame.captured_monotonic),
                        inferred_monotonic=finished,
                        batch_ms=batch_ms,
                        detections=tuple(rows),
                    )
                    self.results.put(snapshot)
                    last_versions[camera_id] = version
                    next_due[camera_id] = finished + interval

                    with self._lock:
                        if not self._camera_started[camera_id]:
                            self._camera_started[camera_id] = finished
                        self._camera_updates[camera_id] += 1
                        self._camera_last_infer[camera_id] = finished

                with self._lock:
                    self._metrics.batches += 1
                    self._metrics.images += len(selected)
                    self._metrics.last_batch_ms = batch_ms
                    n = self._metrics.batches
                    self._metrics.average_batch_ms += (batch_ms - self._metrics.average_batch_ms) / n

        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._set_state("error", message)
            print(f"[DETECT] ERROR {message}", flush=True)
        finally:
            if self._stop.is_set():
                with self._lock:
                    if self._metrics.state != "error":
                        self._metrics.state = "stopped"
