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
    """Sparse CPU ONNX ReID for Pascal hosts running DeepStream/TensorRT 10.x.

    Local detection/tracking stay exactly where they are: YOLO on PyTorch CUDA and
    NvDCF on the DeepStream GPU path. Only appearance embeddings move out of the
    unsupported TensorRT-10 ReID path. They are computed asynchronously from the
    detector branch's existing 704x416 CPU frames and never change tracker geometry.
    """

    MEAN_RGB = np.asarray([123.6750, 116.2800, 103.5300], dtype=np.float32)
    SCALE = np.float32(0.01735207)
    FEATURE_SIZE = 256

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
        self.warmup_ms = 0.0
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

        # Mean-colored padding maps near zero after normalization, which is safer
        # than black padding for tall/narrow CCTV person crops.
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

            # Parse + execute one real-shaped input before declaring the worker
            # healthy. This catches unsupported ONNX operators immediately while
            # keeping failure isolated from GStreamer/NvDCF.
            dummy = np.full((256, 128, 3), 127, dtype=np.uint8)
            started = time.monotonic()
            net.setInput(self._blob(dummy))
            self._normalize_feature(net.forward())
            self.warmup_ms = (time.monotonic() - started) * 1000.0
            self.ready_event.set()
            print(
                "CAMERA_REID external worker ready: "
                f"backend=opencv-cpu feature={self.FEATURE_SIZE} "
                f"warmup={self.warmup_ms:.1f}ms model={self.model_path.name}",
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
                    self.infer_ms = (time.monotonic() - started) * 1000.0
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
            print(f"CAMERA_REID external worker unavailable: {self.error}", flush=True)

    def submit(
        self,
        *,
        source_id: int,
        object_id: int,
        crop_bgr: np.ndarray,
        confidence: float,
        tracker_confidence: float,
    ) -> bool:
        if (
            self.error
            or not self.ready_event.is_set()
            or crop_bgr is None
            or crop_bgr.size == 0
        ):
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
            "warmup_ms": self.warmup_ms,
            "error": self.error,
            "model": str(self.model_path) if self.model_path else "",
        }

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.input_q.put_nowait(None)
        except queue.Full:
            pass