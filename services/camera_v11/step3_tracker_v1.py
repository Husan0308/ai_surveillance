from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from services.ml_service.app.local_tracker import Detection, LocalPersonTracker, TrackerUpdate


class V11PerCameraTrackerV1:
    """CPU-only, per-camera, time-aware ByteTrack-style association for Step3.

    Step2 already emits person detections at roughly 2 Hz per camera.  This adapter
    deliberately disables appearance features: Step3 is local geometric tracking only,
    not ReID.  Lifecycle and prediction use monotonic capture seconds rather than frame
    counts so bursty detector substreams do not shorten or extend tracks accidentally.
    """

    def __init__(
        self,
        camera_ids: Iterable[str],
        *,
        width: int = 672,
        height: int = 384,
        low_thresh: float = 0.18,
        high_thresh: float = 0.30,
        new_track_thresh: float = 0.30,
        match_thresh: float = 0.22,
        low_match_thresh: float = 0.18,
        confirm_hits: int = 2,
        tentative_ttl_sec: float = 0.9,
        shadow_sec: float = 0.9,
        max_lost_sec: float = 2.5,
    ) -> None:
        ids = tuple(str(cid) for cid in camera_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("camera IDs must be non-empty and unique")
        self.width = int(width)
        self.height = int(height)
        self.trackers = {
            cid: LocalPersonTracker(
                cid,
                self.width,
                self.height,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
                new_track_thresh=new_track_thresh,
                match_thresh=match_thresh,
                low_match_thresh=low_match_thresh,
                confirm_hits=confirm_hits,
                tentative_ttl_sec=tentative_ttl_sec,
                shadow_sec=shadow_sec,
                max_lost_sec=max_lost_sec,
                appearance_weight=0.0,
            )
            for cid in ids
        }

    def update(
        self,
        camera_id: str,
        boxes: Iterable[Iterable[float]],
        captured_ns: int,
    ) -> TrackerUpdate:
        tracker = self.trackers.get(camera_id)
        if tracker is None:
            raise KeyError(f"unknown camera {camera_id!r}")
        detections: list[Detection] = []
        for row in boxes:
            values = list(row)
            if len(values) != 5:
                continue
            x1, y1, x2, y2, score = (float(value) for value in values)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
                continue
            x1 = min(float(self.width - 1), max(0.0, x1))
            x2 = min(float(self.width - 1), max(0.0, x2))
            y1 = min(float(self.height - 1), max(0.0, y1))
            y2 = min(float(self.height - 1), max(0.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=np.array((x1, y1, x2, y2), dtype=np.float64),
                    score=score,
                    appearance=None,
                )
            )
        timestamp = int(captured_ns) / 1_000_000_000.0
        return tracker.update(detections, timestamp)
