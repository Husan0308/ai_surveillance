from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .global_identity import GlobalIdentityCore
from .qwen_reid import QwenReIdVerifier
from .reid_embedder import AutoReIdEmbedder
from .reid_quality import crop_signature, evaluate_crop_quality, hamming64


@dataclass
class CropJob:
    camera_id: str
    local_id: int
    room_id: str
    crop: np.ndarray
    source_bbox: tuple[float, float, float, float]
    source_width: int
    source_height: int
    detector_confidence: float
    tracker_confidence: float
    max_other_iou: float
    captured_at: float


class ReIdIdentityEngine:
    """Bounded asynchronous ReID + Qwen identity side-path."""

    def __init__(
        self,
        camera_rooms: dict[str, str],
        config: dict | None = None,
        *,
        root: Path | None = None,
        embedder=None,
        qwen=None,
    ) -> None:
        cfg = dict(config or {})
        cfg["camera_rooms"] = dict(camera_rooms)
        self.core = GlobalIdentityCore(cfg)
        self.embedder = embedder or AutoReIdEmbedder(cfg, root)
        self.qwen = qwen or QwenReIdVerifier(cfg)
        self.sample_interval = max(0.12, float(cfg.get("sample_interval_sec", 0.28)))
        self.min_quality = max(0.1, min(0.9, float(cfg.get("min_crop_quality", 0.34))))
        self.max_batch = max(1, min(8, int(cfg.get("embed_batch", 4))))
        self.duplicate_hamming = max(0, min(16, int(cfg.get("duplicate_hamming", 5))))
        self.duplicate_window = max(0.2, float(cfg.get("duplicate_window_sec", 1.2)))
        self.jpeg_quality = max(55, min(92, int(cfg.get("qwen_jpeg_quality", 82))))

        self._jobs: queue.Queue[CropJob | None] = queue.Queue(maxsize=max(16, int(cfg.get("crop_queue", 48))))
        self._qwen_jobs: queue.Queue[dict | None] = queue.Queue(maxsize=max(8, int(cfg.get("qwen_queue", 24))))
        self._stop = threading.Event()
        self._embed_thread: threading.Thread | None = None
        self._qwen_thread: threading.Thread | None = None
        self._last_sample: dict[tuple[str, int], float] = {}
        self._signatures: dict[tuple[str, int], list[tuple[float, int]]] = {}
        self._qwen_scheduled: set[tuple[str, int, int, int]] = set()
        self._lock = threading.RLock()
        self._metrics = {
            "submitted": 0,
            "queue_dropped": 0,
            "quality_rejects": 0,
            "duplicate_rejects": 0,
            "embedded": 0,
            "embed_errors": 0,
            "qwen_queued": 0,
            "qwen_dropped": 0,
            "qwen_completed": 0,
        }
        self._last_error = ""

    def start(self) -> None:
        if self._embed_thread is not None:
            return
        self._stop.clear()
        self._embed_thread = threading.Thread(target=self._embed_loop, name="camera-v2-reid", daemon=False)
        self._qwen_thread = threading.Thread(target=self._qwen_loop, name="camera-v2-qwen-reid", daemon=False)
        self._embed_thread.start()
        self._qwen_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for q in (self._jobs, self._qwen_jobs):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for thread in (self._embed_thread, self._qwen_thread):
            if thread is not None:
                thread.join(timeout=4.0)
        self._embed_thread = None
        self._qwen_thread = None

    def observe_tracks(self, camera_id: str, room_id: str, tracks: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        visible: list[int] = []
        for row in tracks:
            local_id = int(row.get("object_id", -1))
            if local_id < 0:
                continue
            visible.append(local_id)
            bbox = (
                float(row.get("left", 0.0)),
                float(row.get("top", 0.0)),
                float(row.get("left", 0.0)) + float(row.get("width", 0.0)),
                float(row.get("top", 0.0)) + float(row.get("height", 0.0)),
            )
            self.core.observe_track_activity(
                camera_id, local_id, room_id=room_id, bbox=bbox, seen_at=now
            )
        self.core.observe_camera_snapshot(camera_id, visible, seen_at=now)
        self.core.maintenance(now)

    def submit_crop(self, job: CropJob) -> bool:
        key = (str(job.camera_id), int(job.local_id))
        now = float(job.captured_at)
        with self._lock:
            last = self._last_sample.get(key, -1e9)
            if now - last < self.sample_interval:
                return False
            signature = crop_signature(job.crop)
            history = [row for row in self._signatures.get(key, []) if now - row[0] <= self.duplicate_window]
            if signature and any(hamming64(signature, old) <= self.duplicate_hamming for _t, old in history):
                self._signatures[key] = history
                self._metrics["duplicate_rejects"] += 1
                return False
            history.append((now, signature))
            self._signatures[key] = history[-6:]
            self._last_sample[key] = now
        try:
            self._jobs.put_nowait(job)
            self._metrics["submitted"] += 1
            return True
        except queue.Full:
            self._metrics["queue_dropped"] += 1
            return False

    def _prepare_job(self, job: CropJob):
        quality = evaluate_crop_quality(
            job.crop,
            source_bbox=job.source_bbox,
            source_width=job.source_width,
            source_height=job.source_height,
            detector_confidence=job.detector_confidence,
            tracker_confidence=job.tracker_confidence,
            max_other_iou=job.max_other_iou,
            min_score=self.min_quality,
        )
        if not quality.accepted:
            self._metrics["quality_rejects"] += 1
            return None
        ok, encoded = cv2.imencode(
            ".jpg", job.crop, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        jpeg = encoded.tobytes() if ok else None
        return quality, jpeg

    def _embed_loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            if first is None:
                break
            batch = [first]
            deadline = time.monotonic() + 0.018
            while len(batch) < self.max_batch and time.monotonic() < deadline:
                try:
                    row = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if row is None:
                    self._stop.set()
                    break
                batch.append(row)

            prepared = []
            for job in batch:
                result = self._prepare_job(job)
                if result is not None:
                    prepared.append((job, *result))
            if not prepared:
                continue

            try:
                embeddings = self.embedder.embed_batch([row[0].crop for row in prepared])
                if len(embeddings) != len(prepared):
                    raise RuntimeError("ReID backend returned wrong batch size")
                for (job, quality, jpeg), embedding in zip(prepared, embeddings):
                    action = self.core.observe_embedding(
                        camera_id=job.camera_id,
                        local_id=job.local_id,
                        embedding=embedding,
                        quality=quality.score,
                        captured_at=job.captured_at,
                        room_id=job.room_id,
                        bbox=job.source_bbox,
                        jpeg=jpeg,
                    )
                    self._metrics["embedded"] += 1
                    self._schedule_qwen(job.camera_id, job.local_id, action)
                self._last_error = ""
            except Exception as exc:
                self._metrics["embed_errors"] += len(prepared)
                self._last_error = f"embed:{type(exc).__name__}:{exc}"

    def _schedule_qwen(self, camera_id: str, local_id: int, action: dict) -> None:
        if not self.qwen.enabled or not action.get("needs_qwen"):
            return
        candidate = action.get("candidate") or action.get("global_id")
        if candidate is None:
            return
        payload = self.core.comparison_payload(camera_id, local_id, int(candidate))
        if not payload:
            return
        key = (
            str(camera_id), int(local_id), int(candidate), int(payload.get("evidence_version", 0))
        )
        with self._lock:
            if key in self._qwen_scheduled:
                return
            if len(self._qwen_scheduled) > 256:
                self._qwen_scheduled.clear()
            self._qwen_scheduled.add(key)
        try:
            self._qwen_jobs.put_nowait(payload)
            self._metrics["qwen_queued"] += 1
        except queue.Full:
            self._metrics["qwen_dropped"] += 1

    def _qwen_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._qwen_jobs.get(timeout=0.30)
            except queue.Empty:
                continue
            if payload is None:
                break
            verdict = self.qwen.verify(payload)
            try:
                self.core.apply_qwen_result(
                    payload["camera_id"], int(payload["local_id"]), int(payload["global_id"]),
                    verdict.verdict, verdict.confidence,
                    evidence_version=int(payload.get("evidence_version", 0)),
                )
                self._metrics["qwen_completed"] += 1
            except Exception as exc:
                self._last_error = f"qwen-result:{type(exc).__name__}:{exc}"

    def binding_for_track(self, camera_id: str, local_id: int) -> dict | None:
        return self.core.binding_for_track(camera_id, local_id)

    def bindings(self) -> dict[tuple[str, int], dict]:
        return self.core.bindings()

    def metrics(self) -> dict:
        with self._lock:
            own = dict(self._metrics)
        return {
            **own,
            "crop_queue": self._jobs.qsize(),
            "qwen_queue": self._qwen_jobs.qsize(),
            "last_error": self._last_error,
            "identity": self.core.metrics(),
            "embedder": self.embedder.metrics(),
            "qwen": self.qwen.metrics(),
        }
