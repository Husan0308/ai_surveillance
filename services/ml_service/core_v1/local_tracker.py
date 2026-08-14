from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .visual_tracker import (
    VisualTracker,
    _area_ratio,
    _center_distance,
    _center_size,
    _iou,
)


@dataclass(slots=True)
class TrackedVisualBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int


_INVALID_COST = 1_000_000.0


def linear_sum_assignment(cost_matrix: np.ndarray):
    """Small dependency-free Hungarian solver for dense rectangular costs.

    CCTV scenes normally contain only a handful of people, so O(n^3) assignment
    is negligible compared with detector/JPEG work. Invalid pairs should be
    represented by a very large finite cost and filtered after assignment.
    """
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2-D")
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cost_matrix must contain finite costs")

    transposed = rows > cols
    work = matrix.T.copy() if transposed else matrix.copy()
    n, m = work.shape

    # Classic shortest augmenting-path Hungarian implementation for n <= m.
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = work[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not math.isfinite(float(delta)):
                break
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = []
    for j in range(1, m + 1):
        if p[j] != 0:
            pairs.append((int(p[j]) - 1, j - 1))
    pairs.sort()
    row_ind = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    col_ind = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    if transposed:
        return col_ind, row_ind
    return row_ind, col_ind


class LocalByteTracker(VisualTracker):
    """Camera-local ByteTrack-style tracker with global min-cost association.

    The existing adaptive Kalman model, two-stage high/low confidence recovery,
    birth confirmation and bounded presentation prediction are preserved. The
    only association change is replacing greedy pair picking with a Hungarian
    minimum-cost assignment, which prevents one locally cheapest pair from
    forcing a much worse assignment for another nearby person.
    """

    def __init__(self, *args, fuse_score: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_score = bool(fuse_score)
        self._last_visible: list[TrackedVisualBox] = []
        self._assignment_calls = 0
        self._assignment_pairs = 0
        self._assignment_invalid = 0

    @staticmethod
    def _box_from_track(track, mean=None) -> TrackedVisualBox:
        state = track.mean if mean is None else mean
        w = max(2.0, float(state[2]))
        h = max(2.0, float(state[3]))
        cx = float(state[0])
        cy = float(state[1])
        return TrackedVisualBox(
            cx - w * 0.5,
            cy - h * 0.5,
            cx + w * 0.5,
            cy + h * 0.5,
            float(track.confidence),
            int(track.track_id),
        )

    def _associate(
        self,
        track_ids: list[int],
        detections,
        *,
        observation: float,
        decisive_high: bool,
    ):
        if not track_ids or not detections:
            return [], set(), set()

        ordered_tracks = sorted(int(tid) for tid in track_ids)
        matrix = np.full(
            (len(ordered_tracks), len(detections)),
            _INVALID_COST,
            dtype=np.float64,
        )

        for row, tid in enumerate(ordered_tracks):
            track = self._tracks[tid]
            predicted, predicted_covariance, _horizon = self._observation_prediction(
                track, observation
            )
            ref = self._box_from_track(track, predicted)
            age = max(0.0, observation - track.last_observation)

            for di, det in enumerate(detections):
                if det.confidence < self.byte_high_conf and age > self.low_match_max_age_sec:
                    continue

                iou = _iou(ref, det)
                rcx, rcy, rw, rh = _center_size(ref)
                dcx, dcy, dw, dh = _center_size(det)
                sx = max(20.0, 0.5 * (rw + dw))
                sy = max(20.0, 0.5 * (rh + dh))
                distance = math.hypot((dcx - rcx) / sx, (dcy - rcy) / sy)
                isotropic_distance = _center_distance(ref, det)
                area_similarity = _area_ratio(ref, det)
                if area_similarity < 0.18:
                    continue

                reliability = self._detection_reliability(det.confidence)
                reliability2 = reliability * reliability
                center_cap = self.byte_second_match_center + reliability * (
                    self.byte_match_center - self.byte_second_match_center
                )
                center_gate = 0.20 + reliability2 * (center_cap - 0.20)
                iou_gate = self.byte_second_match_iou + reliability * (
                    self.match_iou - self.byte_second_match_iou
                )

                if age <= self.hold_sec:
                    sigma_norm = math.hypot(
                        math.sqrt(max(0.0, float(predicted_covariance[0, 0]))) / sx,
                        math.sqrt(max(0.0, float(predicted_covariance[1, 1]))) / sy,
                    )
                    center_gate += min(0.20, 2.0 * sigma_norm)
                else:
                    center_gate = max(
                        center_gate,
                        0.20 + reliability2 * (self.reacquire_distance - 0.20),
                    )

                far_quality = min(
                    1.0,
                    max(
                        0.0,
                        (det.confidence - self.start_conf)
                        / max(1e-6, 1.0 - self.start_conf),
                    ),
                )
                far_gate = reliability * (
                    self.snap_distance_boxes
                    + far_quality
                    * max(0.0, self.reacquire_distance - self.snap_distance_boxes)
                )
                strong_far_recovery = (
                    area_similarity >= 0.45 and isotropic_distance <= far_gate
                )

                if decisive_high:
                    trust = min(
                        1.0,
                        max(
                            0.0,
                            (det.confidence - self.byte_high_conf)
                            / max(1e-6, self.start_conf - self.byte_high_conf),
                        ),
                    )
                    decisive_center = 0.12 + trust * (center_gate - 0.12)
                    decisive_iou = 0.55 + trust * (iou_gate - 0.55)
                    accepted = (
                        iou >= decisive_iou
                        or distance <= decisive_center
                        or strong_far_recovery
                    )
                else:
                    accepted = (
                        iou >= iou_gate
                        or distance <= center_gate
                        or strong_far_recovery
                    )
                    if det.confidence < self.start_conf and distance > center_gate:
                        accepted = False
                if not accepted:
                    continue

                size_cost = -math.log(max(1e-6, area_similarity))
                iou_cost = 1.0 - iou
                if self.fuse_score and decisive_high:
                    # Ultralytics ByteTrack defaults to fuse_score=True. Keep
                    # low-confidence rescue geometry-led, but confidence-weight
                    # first-stage IoU so a clean box wins close spatial ties.
                    iou_cost = 1.0 - iou * max(0.05, float(det.confidence))

                cost = (
                    0.55 * min(3.0, distance)
                    + 0.35 * iou_cost
                    + 0.08 * min(3.0, size_cost)
                    + 0.02 * (1.0 - det.confidence)
                )
                # Deterministic microscopic tie-break keeps assignments stable
                # when two costs are numerically identical.
                cost += 1e-9 * (row * max(1, len(detections)) + di)
                matrix[row, di] = cost

        self._assignment_calls += 1
        rows, cols = linear_sum_assignment(matrix)
        matches = []
        used_tracks = set()
        used_detections = set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if matrix[row, col] >= _INVALID_COST * 0.5:
                self._assignment_invalid += 1
                continue
            tid = ordered_tracks[row]
            matches.append((tid, col))
            used_tracks.add(tid)
            used_detections.add(col)
        self._assignment_pairs += len(matches)
        return matches, used_tracks, used_detections

    def visible(self, *args, **kwargs):
        boxes = super().visible(*args, **kwargs)
        with self._lock:
            self._last_visible = [
                TrackedVisualBox(
                    float(box.x1),
                    float(box.y1),
                    float(box.x2),
                    float(box.y2),
                    float(box.confidence),
                    int(getattr(box, "track_id", 0)),
                )
                for box in boxes
            ]
            return list(self._last_visible)

    def snapshot(self):
        with self._lock:
            return [
                {
                    "track_id": int(box.track_id),
                    "bbox": [float(box.x1), float(box.y1), float(box.x2), float(box.y2)],
                    "confidence": float(box.confidence),
                }
                for box in self._last_visible
            ]

    def metrics(self):
        payload = super().metrics()
        with self._lock:
            payload.update(
                {
                    "algorithm": "adaptive-kalman-bytetrack-hungarian-v3",
                    "assignment_solver": "hungarian",
                    "fuse_score": self.fuse_score,
                    "assignment_calls": self._assignment_calls,
                    "assignment_pairs": self._assignment_pairs,
                    "assignment_invalid": self._assignment_invalid,
                    "visible_track_ids": [int(box.track_id) for box in self._last_visible],
                }
            )
        return payload
