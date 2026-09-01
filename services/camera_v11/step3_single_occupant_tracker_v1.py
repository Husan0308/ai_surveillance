from __future__ import annotations

from typing import Iterable

import numpy as np

from services.ml_service.app.local_tracker import Detection, TrackerUpdate
from services.ml_service.app.local_tracker_sparse_v3 import ObservationRecoveryPersonTracker


class SingleOccupantPersonTrackerV1(ObservationRecoveryPersonTracker):
    """One-person validation tracker that blocks duplicate local-ID births.

    This class is intentionally scoped to the current bbox-only single-person test.
    Existing low-confidence association and lost-track recovery remain unchanged.
    The only extra rule is: while a confirmed track was seen recently, an unmatched
    detection is not allowed to mint a second local ID. If association cannot recover
    the existing target, the display may temporarily hold/blank instead of showing a
    second false-positive box.
    """

    def __init__(self, *args, single_occupant_block_sec: float = 4.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.single_occupant_block_sec = max(0.0, float(single_occupant_block_sec))
        self._admission_timestamp = 0.0
        self.blocked_new_total = 0

    def update(self, detections, timestamp: float) -> TrackerUpdate:
        self._admission_timestamp = float(timestamp)
        return super().update(detections, timestamp)

    def _is_duplicate_of_matched(self, det: Detection, matched_tracks: list[object]) -> bool:
        if super()._is_duplicate_of_matched(det, matched_tracks):
            return True

        now = self._admission_timestamp
        for track in self._tracks:
            if track.status == "removed" or track.hits < self.confirm_hits:
                continue
            since = max(0.0, now - float(track.last_detection))
            if since <= self.single_occupant_block_sec:
                self.blocked_new_total += 1
                return True
        return False


class V11SingleOccupantTrackerV1:
    def __init__(
        self,
        camera_ids: Iterable[str],
        *,
        width: int = 672,
        height: int = 384,
        low_thresh: float = 0.18,
        high_thresh: float = 0.30,
        new_track_thresh: float = 0.50,
        match_thresh: float = 0.22,
        low_match_thresh: float = 0.18,
        reacquire_thresh: float = 0.12,
        low_recovery_thresh: float = 0.10,
        low_recovery_sec: float = 2.5,
        confirm_hits: int = 3,
        tentative_ttl_sec: float = 1.6,
        shadow_sec: float = 1.2,
        max_lost_sec: float = 4.5,
        lost_velocity_half_life_sec: float = 0.9,
        live_duplicate_iou: float = 0.60,
        single_occupant_block_sec: float = 4.5,
    ) -> None:
        ids = tuple(str(cid) for cid in camera_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("camera IDs must be non-empty and unique")
        self.width = int(width)
        self.height = int(height)
        self.trackers = {
            cid: SingleOccupantPersonTrackerV1(
                cid,
                self.width,
                self.height,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
                new_track_thresh=new_track_thresh,
                match_thresh=match_thresh,
                low_match_thresh=low_match_thresh,
                reacquire_thresh=reacquire_thresh,
                low_recovery_thresh=low_recovery_thresh,
                low_recovery_sec=low_recovery_sec,
                confirm_hits=confirm_hits,
                tentative_ttl_sec=tentative_ttl_sec,
                shadow_sec=shadow_sec,
                max_lost_sec=max_lost_sec,
                appearance_weight=0.0,
                low_appearance_weight=0.0,
                live_duplicate_iou=live_duplicate_iou,
                lost_velocity_half_life_sec=lost_velocity_half_life_sec,
                single_occupant_block_sec=single_occupant_block_sec,
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
            if not all(np.isfinite(value) for value in (x1, y1, x2, y2, score)):
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

    def blocked_new_total(self, camera_id: str) -> int:
        tracker = self.trackers.get(camera_id)
        if tracker is None:
            return 0
        return int(tracker.blocked_new_total)
