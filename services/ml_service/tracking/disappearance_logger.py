"""Structured Disappearance Logger for AI Surveillance System.

Instruments every visual track disappearance with a structured reason:
- DETECTOR_MISS
- SCHEDULER_OMISSION
- LOW_CONFIDENCE
- ASSOCIATION_FAIL
- RETENTION_EXPIRED
- BOUNDARY_EXIT
- FRONTEND_STALE_METADATA
- GENERATION_MISMATCH
"""
from dataclasses import dataclass, asdict
from collections import deque
import threading
import logging
import json
import time

log = logging.getLogger(__name__)

REASONS = (
    "DETECTOR_MISS",
    "SCHEDULER_OMISSION",
    "LOW_CONFIDENCE",
    "ASSOCIATION_FAIL",
    "RETENTION_EXPIRED",
    "BOUNDARY_EXIT",
    "FRONTEND_STALE_METADATA",
    "GENERATION_MISMATCH",
)

@dataclass
class DisappearanceRecord:
    camera: str
    track_id: str
    last_real_bbox: tuple[float, float, float, float]
    predicted_bbox: tuple[float, float, float, float]
    last_confidence: float
    observation_age_ms: float
    lost_for_ms: float
    reason: str
    next_detection_bbox: tuple[float, float, float, float] | None = None
    timestamp: float = 0.0

class DisappearanceAuditor:
    def __init__(self, maxlen=500):
        self._lock = threading.Lock()
        self._records = deque(maxlen=maxlen)
        self._pending_next_detection = {}  # (camera, track_id) -> record

    def record_disappearance(self, camera, track_id, last_real_bbox, predicted_bbox,
                             last_confidence, observation_age_ms, lost_for_ms, reason,
                             next_detection_bbox=None):
        if reason not in REASONS:
            reason = "DETECTOR_MISS"
        record = DisappearanceRecord(
            camera=camera,
            track_id=track_id,
            last_real_bbox=tuple(float(v) for v in last_real_bbox),
            predicted_bbox=tuple(float(v) for v in predicted_bbox),
            last_confidence=float(last_confidence),
            observation_age_ms=float(observation_age_ms),
            lost_for_ms=float(lost_for_ms),
            reason=reason,
            next_detection_bbox=tuple(float(v) for v in next_detection_bbox) if next_detection_bbox else None,
            timestamp=time.time()
        )
        with self._lock:
            self._records.append(record)
            key = (camera, track_id)
            self._pending_next_detection[key] = record

        log.warning("VISUAL_DISAPPEARANCE_EVENT %s", json.dumps(asdict(record)))
        return record

    def record_reacquisition(self, camera, track_id, next_detection_bbox):
        key = (camera, track_id)
        with self._lock:
            record = self._pending_next_detection.pop(key, None)
            if record:
                record.next_detection_bbox = tuple(float(v) for v in next_detection_bbox)
                log.info("VISUAL_REACQUISITION_CORRELATED camera=%s track_id=%s next_bbox=%s",
                         camera, track_id, record.next_detection_bbox)

    def snapshot(self):
        with self._lock:
            return [asdict(r) for r in self._records]

    def summary(self):
        with self._lock:
            counts = {reason: 0 for reason in REASONS}
            by_camera = {}
            for r in self._records:
                counts[r.reason] = counts.get(r.reason, 0) + 1
                by_cam = by_camera.setdefault(r.camera, {reason: 0 for reason in REASONS})
                by_cam[r.reason] = by_cam.get(r.reason, 0) + 1
            return {
                "total_disappearances": len(self._records),
                "counts_by_reason": counts,
                "counts_by_camera": by_camera,
            }

disappearance_auditor = DisappearanceAuditor()
