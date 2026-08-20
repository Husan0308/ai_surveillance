from __future__ import annotations

from .visual_tracker import VisualTracker, _BirthCandidate, _center_distance, _iou


class SparseCadenceVisualTracker(VisualTracker):
    """Core-v1 visual tracker with a birth window sized for sparse RF-DETR cadence.

    The mature tracker used a hard-coded 1.25 s birth-candidate lifetime because
    YOLO revisited each camera much faster.  On the six-camera GTX 1050 Ti RF-DETR
    schedule a camera is commonly revisited only every ~1.5-2.0 s, so a candidate
    expired before the second confirming observation and no visual track could
    ever be born.  Only this temporal window is widened; Kalman/Byte association
    remains unchanged.
    """

    def __init__(self, *, birth_candidate_ttl_ms=5000, **kwargs):
        super().__init__(**kwargs)
        self.birth_candidate_ttl_sec = max(
            1.25,
            float(birth_candidate_ttl_ms) / 1000.0,
        )

    def _confirm_birth(
        self,
        det,
        observation: float,
        now: float,
        frame_id: int,
        required_hits: int,
        used_candidates: set[int],
    ):
        self._birth_candidates = [
            candidate
            for candidate in self._birth_candidates
            if now - candidate.last_seen_wall <= self.birth_candidate_ttl_sec
        ]

        best = None
        best_score = float("inf")
        for candidate in self._birth_candidates:
            if id(candidate) in used_candidates or candidate.last_frame_id == frame_id:
                continue
            iou = _iou(candidate.box, det)
            distance = _center_distance(candidate.box, det)
            elapsed = max(0.0, observation - candidate.observation_time)
            max_center = min(
                1.35,
                max(0.35, 0.35 + 2.5 * min(elapsed, 0.40)),
            )
            if iou < 0.12 and distance > max_center:
                continue
            score = (1.0 - iou) + 0.25 * distance
            if score < best_score:
                best = candidate
                best_score = score

        if best is None:
            if required_hits <= 1:
                return det, observation, 1
            self._birth_candidates.append(
                _BirthCandidate(det, observation, now, frame_id, 1)
            )
            return None

        used_candidates.add(id(best))
        previous_box = best.box
        previous_observation = best.observation_time
        best.box = det
        best.observation_time = observation
        best.last_seen_wall = now
        best.last_frame_id = frame_id
        best.hits += 1
        if best.hits >= required_hits:
            hits = best.hits
            self._birth_candidates.remove(best)
            return previous_box, previous_observation, hits
        return None

    def metrics(self):
        payload = super().metrics()
        payload["birth_candidate_ttl_ms"] = self.birth_candidate_ttl_sec * 1000.0
        return payload
