from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracker_profile import resolve_reid_model


@dataclass
class ReIDTask:
    source_id: int
    object_id: int
    crop_bgr: np.ndarray
    confidence: float
    tracker_confidence: float
    bbox: tuple[float, float, float, float]
    submitted_at: float


class ExternalReIDWorker:
    """Sparse CPU ONNX ReID for the Pascal Camera V2 host.

    Geometry/local identity remain owned by YOLO + NvDCF. This worker extracts a
    TAO ReIdentificationNet embedding plus a cheap clothing-colour signature from
    the same person crop. The colour cue is intentionally secondary: it stabilizes
    difficult cross-view matches but must never be trusted by itself.
    """

    MEAN_RGB = np.asarray([123.6750, 116.2800, 103.5300], dtype=np.float32)
    SCALE = np.float32(0.01735207)
    FEATURE_SIZE = 256
    INPUT_WIDTH = 128
    INPUT_HEIGHT = 256
    COLOR_FEATURE_SIZE = 96

    def __init__(self) -> None:
        self.max_queue = max(4, int(os.environ.get("CAMERA_V2_REID_QUEUE", "16")))
        self.input_q: queue.Queue[ReIDTask | None] = queue.Queue(maxsize=self.max_queue)
        self.output_q: queue.Queue[dict] = queue.Queue(maxsize=self.max_queue * 2)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.fatal_error = ""
        self.last_error = ""
        self.backend = "opencv-cpu"
        self.features = 0
        self.submitted = 0
        self.failed = 0
        self.dropped = 0
        self.infer_ms = 0.0
        self.warmup_ms = 0.0
        self.model_path: Path | None = None
        self.thread = threading.Thread(target=self._run, name="camera-v2-reid", daemon=True)
        self.thread.start()

    @property
    def error(self) -> str:
        return self.fatal_error

    @classmethod
    def _resize_rgb(cls, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("empty ReID crop")
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] < 3:
            raise ValueError(f"invalid ReID crop shape: {getattr(crop_bgr, 'shape', None)}")
        rgb = cv2.cvtColor(crop_bgr[..., :3], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (cls.INPUT_WIDTH, cls.INPUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized.astype(np.float32, copy=False)

    @classmethod
    def _blob(cls, crop_bgr: np.ndarray) -> np.ndarray:
        image = cls._resize_rgb(crop_bgr)
        image = (image - cls.MEAN_RGB) * cls.SCALE
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    @classmethod
    def _normalize_feature(cls, raw: np.ndarray) -> tuple[float, ...]:
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vector.size != cls.FEATURE_SIZE:
            raise ValueError(
                f"unexpected ReID feature size: got={vector.size} expected={cls.FEATURE_SIZE}"
            )
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("invalid ReID feature norm")
        vector = vector / norm
        return tuple(float(v) for v in vector)

    @classmethod
    def _color_signature(cls, crop_bgr: np.ndarray) -> tuple[float, ...]:
        """Return upper/lower-body HSV histograms with a Hellinger transform."""
        import cv2

        if crop_bgr is None or crop_bgr.size == 0:
            return ()
        h, w = crop_bgr.shape[:2]
        if h < 24 or w < 12:
            return ()

        # Ignore border/background and most head/feet pixels. Two body bands make
        # shirt/trouser colour useful without trying to turn colour into an ID.
        x1, x2 = int(w * 0.12), max(int(w * 0.88), int(w * 0.12) + 1)
        bands = ((0.18, 0.55), (0.52, 0.90))
        parts: list[np.ndarray] = []
        for ylo, yhi in bands:
            y1 = int(h * ylo)
            y2 = max(int(h * yhi), y1 + 1)
            roi = crop_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                return ()
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
            hist = np.asarray(hist, dtype=np.float32).reshape(-1)
            total = float(hist.sum())
            if total <= 1e-8:
                return ()
            hist /= total
            parts.append(np.sqrt(hist))

        vector = np.concatenate(parts).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if vector.size != cls.COLOR_FEATURE_SIZE or not np.isfinite(norm) or norm <= 1e-8:
            return ()
        vector /= norm
        return tuple(float(v) for v in vector)

    def _run(self) -> None:
        try:
            import cv2

            try:
                cv2.setNumThreads(max(1, int(os.environ.get("CAMERA_V2_REID_CPU_THREADS", "2"))))
            except Exception:
                pass

            self.model_path = resolve_reid_model()
            net = cv2.dnn.readNetFromONNX(str(self.model_path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            dummy = np.full((320, 120, 3), 127, dtype=np.uint8)
            started = time.monotonic()
            blob = self._blob(dummy)
            if blob.shape != (1, 3, self.INPUT_HEIGHT, self.INPUT_WIDTH):
                raise RuntimeError(f"unexpected ReID blob shape: {blob.shape}")
            net.setInput(blob)
            self._normalize_feature(net.forward())
            color = self._color_signature(dummy)
            if len(color) != self.COLOR_FEATURE_SIZE:
                raise RuntimeError(f"unexpected colour feature size: {len(color)}")
            self.warmup_ms = (time.monotonic() - started) * 1000.0
            self.ready_event.set()
            print(
                "CAMERA_REID external worker ready: "
                f"backend=opencv-cpu input={self.INPUT_WIDTH}x{self.INPUT_HEIGHT} "
                f"feature={self.FEATURE_SIZE} color={self.COLOR_FEATURE_SIZE} "
                f"preprocess=tao-direct-resize warmup={self.warmup_ms:.1f}ms "
                f"model={self.model_path.name}",
                flush=True,
            )

            while not self.stop_event.is_set():
                try:
                    task = self.input_q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if task is None:
                    return

                started = time.monotonic()
                try:
                    net.setInput(self._blob(task.crop_bgr))
                    feature = self._normalize_feature(net.forward())
                    color_feature = self._color_signature(task.crop_bgr)
                    self.infer_ms = (time.monotonic() - started) * 1000.0
                    self.features += 1
                    self.last_error = ""
                    row = {
                        "source_id": int(task.source_id),
                        "object_id": int(task.object_id),
                        "confidence": float(task.confidence),
                        "tracker_confidence": float(task.tracker_confidence),
                        "feature": feature,
                        "color_feature": color_feature,
                        "bbox": tuple(float(v) for v in task.bbox),
                        "captured_at": float(task.submitted_at),
                    }
                    try:
                        self.output_q.put_nowait(row)
                    except queue.Full:
                        try:
                            self.output_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.output_q.put_nowait(row)
                        except queue.Full:
                            self.dropped += 1
                except Exception as exc:
                    self.failed += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.ready_event.set()
            print(f"CAMERA_REID external worker unavailable: {self.fatal_error}", flush=True)

    def submit(
        self,
        *,
        source_id: int,
        object_id: int,
        crop_bgr: np.ndarray,
        confidence: float,
        tracker_confidence: float,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> bool:
        if self.fatal_error or not self.ready_event.is_set() or crop_bgr is None or crop_bgr.size == 0:
            return False
        if bbox is None:
            bbox = (0.0, 0.0, 0.0, 0.0)
        task = ReIDTask(
            source_id=int(source_id),
            object_id=int(object_id),
            crop_bgr=np.ascontiguousarray(crop_bgr),
            confidence=float(confidence),
            tracker_confidence=float(tracker_confidence),
            bbox=tuple(float(v) for v in bbox),
            submitted_at=time.monotonic(),
        )
        try:
            self.input_q.put_nowait(task)
            self.submitted += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def drain(self, limit: int = 16) -> list[dict]:
        rows: list[dict] = []
        for _ in range(max(1, int(limit))):
            try:
                rows.append(self.output_q.get_nowait())
            except queue.Empty:
                break
        return rows

    def snapshot(self) -> dict:
        return {
            "backend": self.backend,
            "ready": self.ready_event.is_set() and not self.fatal_error,
            "features": self.features,
            "submitted": self.submitted,
            "failed": self.failed,
            "queued": self.input_q.qsize(),
            "dropped": self.dropped,
            "infer_ms": self.infer_ms,
            "warmup_ms": self.warmup_ms,
            "error": self.fatal_error,
            "last_error": self.last_error,
            "model": str(self.model_path) if self.model_path else "",
        }

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.input_q.put_nowait(None)
        except queue.Full:
            pass
