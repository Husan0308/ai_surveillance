from __future__ import annotations

import logging
from pathlib import Path

from .detector import YoloDetectorWorker, _detector_process_main

log = logging.getLogger(__name__)


class StableYoloDetectorWorker(YoloDetectorWorker):
    """Detector-only worker with safe Ultralytics checkpoint fallback.

    A project-local model path is used when it exists. If it does not exist,
    the worker passes a plain Ultralytics checkpoint name (for example
    ``yolo26m.pt``) to YOLO so Ultralytics can resolve/download it. Pose is not
    part of this hot path.
    """

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

    def metrics(self):
        payload = super().metrics()
        payload.update(
            {
                "model": self.model_source,
                "configured_model": str(self.config.get("model", "yolo26m.pt")),
                "model_local_exists": self.model_local_exists,
                "cuda_topology": "detector_only_spawned_process",
                "pose_in_hot_path": False,
            }
        )
        return payload
