from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from services.ml_service.app.local_tracker import Detection, TrackerUpdate, appearance_descriptor
from services.ml_service.app.local_tracker_sparse_v2 import SparseRecoveryPersonTracker


class ObservationRecoveryPersonTracker(SparseRecoveryPersonTracker):
    """Step 4 v3: sparse-detector tracker with bounded observation-centric recovery.

    V2 fixed immediate lost->low recovery, but live 2 Hz tests still showed two failure
    modes: a lost track could expire after a few consecutive misses, and a low-score
    duplicate could pull another track away from its last observation. V3 keeps V14 and
    the CPU-only tracker boundary frozen while strengthening only local association:

    * recently-lost tracks keep a longer non-rendered recovery window;
    * lost-motion velocity decays instead of extrapolating indefinitely;
    * lost association considers both predicted state and the last real observation;
    * low-score association uses appearance only as a veto/tie hint, never to teleport;
    * a strongly overlapping recent confirmed track can veto minting a duplicate ID.

    This is inspired by observation-centric tracking: long gaps should not blindly trust
    accumulated motion prediction. No neural ReID, GPU tracker, NvDCF or global identity
    is introduced here.
    """

    def __init__(
        self,
        *args,
        low_appearance_weight: float = 0.16,
        low_appearance_floor: float = 0.45,
        live_duplicate_iou: float = 0.72,
        lost_velocity_half_life_sec: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.low_appearance_weight = min(0.30, max(0.0, float(low_appearance_weight)))
        self.low_appearance_floor = min(0.90, max(0.0, float(low_appearance_floor)))
        self.live_duplicate_iou = min(0.95, max(0.55, float(live_duplicate_iou)))
        self.lost_velocity_half_life_sec = max(0.25, float(lost_velocity_half_life_sec))

    def _predict_to(self, track, timestamp: float) -> None:
        if track.status != "lost":
            super()._predict_to(track, timestamp)
            return

        dt = max(0.0, min(2.0, timestamp - track.state_time))
        if dt <= 0.0:
            return

        # Sparse detections can disappear for multiple 500 ms periods. Constant-velocity
        # extrapolation becomes increasingly unsafe over those gaps, so decay motion and
        # let the last real observation remain a strong recovery anchor.
        decay = math.pow(0.5, dt / self.lost_velocity_half_life_sec)
        track.state_vec = track.state_vec + track.velocity * dt * decay
        track.velocity = track.velocity * decay
        track.state_vec[2] = min(float(self.frame_width), max(4.0, track.state_vec[2]))
        track.state_vec[3] = min(float(self.frame_height), max(4.0, track.state_vec[3]))
        track.state_time = timestamp

    def _score_at_last_observation(
        self,
        track,
        det: Detection,
        timestamp: float,
        *,
        low_stage: bool,
    ) -> float:
        current = track.state_vec.copy()
        try:
            track.state_vec = track.last_measurement.copy()
            return super()._pair_score(track, det, timestamp, low_stage=low_stage)
        finally:
            track.state_vec = current

    def _pair_score(self, track, det: Detection, timestamp: float, *, low_stage: bool) -> float:
        score = super()._pair_score(track, det, timestamp, low_stage=low_stage)

        # After a miss, do not rely only on the extrapolated state. The last detector
        # observation is a stable second hypothesis and prevents long-gap drift from
        # forcing a new ID when the person reappears close to where they were observed.
        if track.status == "lost":
            anchor_score = self._score_at_last_observation(
                track, det, timestamp, low_stage=low_stage
            )
            score = max(score, anchor_score)

        if score < 0.0 or not low_stage:
            return score

        app = self._appearance_similarity(track.appearance, det.appearance)
        if app is None:
            return score

        anchor_box = self._state_to_xyxy(track.last_measurement)
        anchor_iou = self._iou(anchor_box, det.bbox)
        tw = max(1.0, float(anchor_box[2] - anchor_box[0]))
        th = max(1.0, float(anchor_box[3] - anchor_box[1]))
        box_diag = math.hypot(tw, th)
        tcx = 0.5 * float(anchor_box[0] + anchor_box[2])
        tcy = 0.5 * float(anchor_box[1] + anchor_box[3])
        dcx = 0.5 * float(det.bbox[0] + det.bbox[2])
        dcy = 0.5 * float(det.bbox[1] + det.bbox[3])
        dist = math.hypot(dcx - tcx, dcy - tcy)

        # A low-confidence box with weak appearance and essentially no overlap must not
        # hijack a nearby person's track. Strong geometry can still recover a changed
        # appearance, and appearance alone can never jump across the frame.
        if (
            app < self.low_appearance_floor
            and anchor_iou < 0.05
            and dist > 0.30 * box_diag
        ):
            return -1.0

        return float(
            (1.0 - self.low_appearance_weight) * score
            + self.low_appearance_weight * app
        )

    def _is_duplicate_of_matched(self, det: Detection, matched_tracks: list[object]) -> bool:
        if super()._is_duplicate_of_matched(det, matched_tracks):
            return True

        # V2 only checked tracks matched in the current update. A detector box can still
        # mint a duplicate ID beside a confirmed live/lost track if association just
        # missed its threshold. Strong overlap plus compatible appearance vetoes that.
        for track in self._tracks:
            if track.status == "removed" or track.hits < self.confirm_hits:
                continue
            predicted = self._state_to_xyxy(track.state_vec)
            anchor = self._state_to_xyxy(track.last_measurement)
            overlap = max(self._iou(predicted, det.bbox), self._iou(anchor, det.bbox))
            if overlap < self.live_duplicate_iou:
                continue
            app = self._appearance_similarity(track.appearance, det.appearance)
            if app is None:
                if overlap >= 0.88:
                    return True
            elif app >= 0.55:
                return True
        return False


class MultiCameraObservationRecoveryTracker:
    def __init__(self, camera_ids: Iterable[str], width: int, height: int, **kwargs) -> None:
        self.trackers = {
            cid: ObservationRecoveryPersonTracker(cid, width, height, **kwargs)
            for cid in camera_ids
        }

    def update(
        self,
        camera_id: str,
        boxes: Iterable[Iterable[float]],
        frame_bgr: np.ndarray,
        captured_ns: int,
    ) -> TrackerUpdate:
        detections: list[Detection] = []
        for row in boxes:
            values = list(row)
            if len(values) != 5:
                continue
            x1, y1, x2, y2, score = (float(v) for v in values)
            bbox = np.array((x1, y1, x2, y2), dtype=np.float64)
            detections.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    appearance=appearance_descriptor(frame_bgr, bbox),
                )
            )
        timestamp = captured_ns / 1_000_000_000.0
        return self.trackers[camera_id].update(detections, timestamp)
