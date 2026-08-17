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
    submitted_at: float


class ExternalReIDWorker:
    """Sparse CPU ONNX ReID for GPUs unsupported by TensorRT 10.x.

    DeepStream 7.1 uses TensorRT 10.3. TensorRT 10.x does not support Pascal/SM6.1,
    so enabling NvDCF's TensorRT ReID on a GTX 1050 Ti can abort the process while
    the tracker tries to create an engine. This worker keeps NvDCF geometry/local
    IDs on the GPU, but extracts cross-camera appearance embeddings from the already
    available 704x416 detector frames with OpenCV DNN on CPU.

    ReID is intentionally sparse and asynchronous. It never blocks the GStreamer
    tracker probe or YOLO scheduler, and it never feeds geometry back into NvDCF.
    """

    MEAN_RGB = np.asarray([123.6750, 116.2800, 103.5300], dtype=np.float32)
    SCALE = np.float32(0.01735207)

    def __init__(self) -> None:
        self.max_queue = max(4, int(os.environ.get("CAMERA_V2_REID_QUEUE", "16")))
        self.input_q: queue.Queue[ReIDTask | None] = queue.Queue(maxsize=self.max_queue)
        self.output_q: queue.Queue[dict] = queue.Queue(maxsize=self.max_queue * 2)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error = ""
        self.backend = "opencv-cpu"
        self.features = 0
        self.dropped = 0
        self.infer_ms = 0.0
        self.model_path: Path | None = None
        self.thread = threading.Thread(target=self._run, name="camera-v2-reid", daemon=True)
        self.thread.start()

    @staticmethod
    def _letterbox_rgb(crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("empty ReID crop")
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        target_w, target_h = 128, 256
        h, w = rgb.shape[:2]
        scale = min(target_w / max(1, w), target_h / max(1, h))
        new_w = max(1, min(target_w, int(round(w * scale))))
        new_h = max(1, min(target_h, int(round(h * scale))))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Fill with ImageNet mean so padded pixels become approximately zero after
        # y = scale * (x - mean), matching the DeepStream ReID preprocessing model.
        canvas = np.empty((target_h, target_w, 3), dtype=np.float32)
        canvas[...] = ExternalReIDWorker.MEAN_RGB
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized.astype(np.float32)
        return canvas

    @classmethod
    def _blob(cls, crop_bgr: np.ndarray) -> np.ndarray:
        image = cls._letterbox_rgb(crop_bgr)
        image = (image - cls.MEAN_RGB) * cls.SCALE
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    @staticmethod
    def _normalize_feature(raw: np.ndarray) -> tuple[float, ...]:
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("invalid ReID feature norm")
        vector = vector / norm
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
            self.ready_event.set()

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
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    self.infer_ms = elapsed_ms
                    self.features += 1
                    row = {
                        "source_id": int(task.source_id),
                        "object_id": int(task.object_id),
                        "confidence": float(task.confidence),
                        "tracker_confidence": float(task.tracker_confidence),
                        "feature": feature,
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
                    self.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready_event.set()

    def submit(
        self,
        *,
        source_id: int,
        object_id: int,
        crop_bgr: np.ndarray,
        confidence: float,
        tracker_confidence: float,
    ) -> bool:
        if self.error or crop_bgr is None or crop_bgr.size == 0:
            return False
        task = ReIDTask(
            source_id=int(source_id),
            object_id=int(object_id),
            crop_bgr=np.ascontiguousarray(crop_bgr),
            confidence=float(confidence),
            tracker_confidence=float(tracker_confidence),
            submitted_at=time.monotonic(),
        )
        try:
            self.input_q.put_nowait(task)
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
            "ready": self.ready_event.is_set() and not self.error,
            "features": self.features,
            "queued": self.input_q.qsize(),
            "dropped": self.dropped,
            "infer_ms": self.infer_ms,
            "error": self.error,
            "model": str(self.model_path) if self.model_path else "",
        }

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.input_q.put_nowait(None)
        except queue.Full:
            pass
