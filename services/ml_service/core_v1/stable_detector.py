from __future__ import annotations

import logging
from pathlib import Path
import time

from .detector import YoloDetectorWorker, _detector_process_main

log = logging.getLogger(__name__)


class StableYoloDetectorWorker(YoloDetectorWorker):
    """Detector-only worker with safe checkpoint fallback and zero redundant resize."""

    def __init__(self, frame_stores, config: dict, project_root: Path):
        super().__init__(frame_stores, config, project_root)

        configured = str(config.get("model", "yolo26m.pt")).strip() or "yolo26m.pt"
        configured_path = Path(configured).expanduser()
        local_path = configured_path if configured_path.is_absolute() else self.project_root / configured_path

        self.model_local_exists = local_path.exists()
        if self.model_local_exists:
            self.model_source = str(local_path)
        else:
            fallback = str(config.get("model_fallback", "")).strip()
            self.model_source = fallback or configured_path.name or "yolo26m.pt"
            log.warning(
                "CORE_V1_YOLO_LOCAL_MODEL_MISSING configured=%s fallback=%s",
                local_path,
                self.model_source,
            )
        self._same_size_prepare_skips = 0
        self._min_submit_interval = max(0.0, float(config.get("min_submit_interval_ms", 0.0)) / 1000.0)
        self._next_submit_monotonic = 0.0
        self._submit_throttle_skips = 0

    def _spawn_process(self):
        self._input_queue = self._ctx.Queue(maxsize=1)
        self._output_queue = self._ctx.Queue(maxsize=16)
        self._process = self._ctx.Process(
            target=_detector_process_main,
            name="core-v1-yolo-cuda",
            args=(
                self._input_queue,
                self._output_queue,
                self.config,
                self.model_source,
            ),
            daemon=False,
        )
        self._process.start()
        log.info(
            "CORE_V1_YOLO_PROCESS_STARTED pid=%s start_method=spawn model_source=%s",
            self._process.pid,
            self.model_source,
        )

    def _prepare_payload(self, selected):
        """Do not resize again when capture already matches the network canvas."""
        import cv2

        started = time.perf_counter()
        entries = []
        for cid, frame, _version, age_ms in selected:
            if int(frame.width) == self.input_w and int(frame.height) == self.input_h:
                full = frame.image
                self._same_size_prepare_skips += 1
            else:
                full = cv2.resize(
                    frame.image,
                    (self.input_w, self.input_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            entries.append(
                {
                    "camera_id": cid,
                    "frame_id": int(frame.frame_id),
                    "captured_mono": float(frame.captured_monotonic),
                    "source_w": int(frame.width),
                    "source_h": int(frame.height),
                    "full_shape": (self.input_h, self.input_w),
                    "full_image": full,
                    "roi": self._prepare_roi(cid, frame),
                }
            )
            self._last_submit_age_ms = age_ms
            self._submit_age_ms.append(age_ms)
        with self._lock:
            self._last_prepare_ms = (time.perf_counter() - started) * 1000.0
        return entries

    def _submit_if_idle(self):
        if self._min_submit_interval > 0.0:
            now = time.monotonic()
            if now < self._next_submit_monotonic:
                self._submit_throttle_skips += 1
                return
        before = self._inflight_batch_id
        super()._submit_if_idle()
        if before is None and self._inflight_batch_id is not None and self._min_submit_interval > 0.0:
            self._next_submit_monotonic = time.monotonic() + self._min_submit_interval

    def metrics(self):
        payload = super().metrics()
        payload.update(
            {
                "model": self.model_source,
                "configured_model": str(self.config.get("model", "yolo26m.pt")),
                "model_local_exists": self.model_local_exists,
                "cuda_topology": "detector_only_spawned_process",
                "pose_in_hot_path": False,
                "same_size_prepare_skips": self._same_size_prepare_skips,
                "min_submit_interval_ms": self._min_submit_interval * 1000.0,
                "submit_throttle_skips": self._submit_throttle_skips,
            }
        )
        return payload
