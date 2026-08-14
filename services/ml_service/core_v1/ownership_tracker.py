from __future__ import annotations

import math
import re

from .local_tracker import LocalByteTracker
from .visual_tracker import _area_ratio, _center_distance, _iou


class OwnershipLockedTracker(LocalByteTracker):
    """Camera-local tracker that prefers a short hold over an ID hijack.

    ByteTrack is intentionally appearance-free. When two mature tracks become
    spatially indistinguishable, a geometry-only assignment can be numerically
    valid but semantically unsafe. This wrapper quarantines detections that are
    almost equally plausible for two nearby mature tracks. Existing tracks keep
    their predicted identity for the short hold window and can re-associate once
    the people separate. A quarantined observation is also forbidden from
    birthing a duplicate track in the same update.

    Track IDs use a camera namespace so the same visible ID is never reused by
    two different cameras while cross-camera ReID is disabled.
    """

    def __init__(
        self,
        *args,
        camera_id: str,
        ownership_lock: bool = True,
        ownership_min_hits: int = 3,
        ownership_margin: float = 0.09,
        ownership_low_margin_multiplier: float = 1.5,
        ownership_proximity: float = 0.70,
        ownership_overlap_iou: float = 0.08,
        ownership_iou_advantage: float = 0.14,
        ownership_distance_advantage: float = 0.14,
        ownership_max_competitor_age_ms: float = 700.0,
        id_namespace_stride: int = 10000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.camera_id = str(camera_id)
        self.ownership_lock = bool(ownership_lock)
        self.ownership_min_hits = max(1, int(ownership_min_hits))
        self.ownership_margin = max(0.0, float(ownership_margin))
        self.ownership_low_margin_multiplier = max(
            1.0, float(ownership_low_margin_multiplier)
        )
        self.ownership_proximity = max(0.05, float(ownership_proximity))
        self.ownership_overlap_iou = max(0.0, min(1.0, float(ownership_overlap_iou)))
        self.ownership_iou_advantage = max(0.0, float(ownership_iou_advantage))
        self.ownership_distance_advantage = max(
            0.0, float(ownership_distance_advantage)
        )
        self.ownership_max_competitor_age_sec = max(
            0.05, float(ownership_max_competitor_age_ms) / 1000.0
        )

        stride = max(1000, int(id_namespace_stride))
        digits = re.findall(r"\d+", self.camera_id)
        camera_index = int(digits[-1]) if digits else 0
        self.id_namespace_stride = stride
        self.id_namespace_base = camera_index * stride
        self._next_id = self.id_namespace_base + 1

        self._ownership_quarantine: set[int] = set()
        self._ownership_rejects = 0
        self._ownership_birth_blocks = 0
        self._ownership_clear_matches = 0

    def display_label(self, track_id: int) -> str:
        local = max(0, int(track_id) - self.id_namespace_base)
        digits = re.findall(r"\d+", self.camera_id)
        camera_index = int(digits[-1]) if digits else 0
        return f"C{camera_index:02d}-{local:03d}"

    @staticmethod
    def _identity_cost(ref, det) -> float:
        area_similarity = _area_ratio(ref, det)
        if area_similarity < 0.18:
            return float("inf")
        distance = _center_distance(ref, det)
        iou = _iou(ref, det)
        size_cost = -math.log(max(1e-6, area_similarity))
        return (
            0.62 * min(3.0, distance)
            + 0.30 * (1.0 - iou)
            + 0.08 * min(3.0, size_cost)
        )

    def _ownership_is_ambiguous(
        self,
        tid: int,
        det,
        track_ids: list[int],
        observation: float,
        decisive_high: bool,
    ) -> bool:
        if not self.ownership_lock or len(track_ids) < 2:
            return False
        assigned = self._tracks.get(int(tid))
        if assigned is None or assigned.hits < self.ownership_min_hits:
            return False

        assigned_mean, _cov, _horizon = self._observation_prediction(
            assigned, observation
        )
        assigned_ref = self._box_from_track(assigned, assigned_mean)
        assigned_cost = self._identity_cost(assigned_ref, det)
        if not math.isfinite(assigned_cost):
            return False
        assigned_iou = _iou(assigned_ref, det)
        assigned_distance = _center_distance(assigned_ref, det)
        margin = self.ownership_margin * (
            1.0 if decisive_high else self.ownership_low_margin_multiplier
        )

        for other_tid in track_ids:
            other_tid = int(other_tid)
            if other_tid == int(tid):
                continue
            other = self._tracks.get(other_tid)
            if other is None or other.hits < self.ownership_min_hits:
                continue
            if observation - other.last_observation > self.ownership_max_competitor_age_sec:
                continue

            other_mean, _cov, _horizon = self._observation_prediction(
                other, observation
            )
            other_ref = self._box_from_track(other, other_mean)
            tracks_close = (
                _center_distance(assigned_ref, other_ref) <= self.ownership_proximity
                or _iou(assigned_ref, other_ref) >= self.ownership_overlap_iou
            )
            if not tracks_close:
                continue

            other_cost = self._identity_cost(other_ref, det)
            if not math.isfinite(other_cost):
                continue
            if other_cost > assigned_cost + margin:
                continue

            other_iou = _iou(other_ref, det)
            other_distance = _center_distance(other_ref, det)
            clear_iou_owner = (
                assigned_iou >= other_iou + self.ownership_iou_advantage
            )
            clear_distance_owner = (
                assigned_distance + self.ownership_distance_advantage
                <= other_distance
            )
            if clear_iou_owner or clear_distance_owner:
                continue
            return True
        return False

    def _associate(
        self,
        track_ids: list[int],
        detections,
        *,
        observation: float,
        decisive_high: bool,
    ):
        if not detections:
            return [], set(), set()

        # A detection quarantined in stage 1 must not steal another ID in stage 2.
        eligible_indices = [
            index
            for index, det in enumerate(detections)
            if id(det) not in self._ownership_quarantine
        ]
        if not eligible_indices:
            return [], set(), set()
        eligible = [detections[index] for index in eligible_indices]

        matches, _used_tracks, _used_detections = super()._associate(
            track_ids,
            eligible,
            observation=observation,
            decisive_high=decisive_high,
        )

        accepted = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for tid, eligible_index in matches:
            original_index = eligible_indices[eligible_index]
            det = detections[original_index]
            if self._ownership_is_ambiguous(
                tid,
                det,
                track_ids,
                observation,
                decisive_high,
            ):
                self._ownership_quarantine.add(id(det))
                self._ownership_rejects += 1
                continue
            accepted.append((int(tid), int(original_index)))
            used_tracks.add(int(tid))
            used_detections.add(int(original_index))
            self._ownership_clear_matches += 1
        return accepted, used_tracks, used_detections

    def _confirm_birth(
        self,
        det,
        observation: float,
        now: float,
        frame_id: int,
        required_hits: int,
        used_candidates: set[int],
    ):
        if id(det) in self._ownership_quarantine:
            self._ownership_birth_blocks += 1
            return None
        return super()._confirm_birth(
            det,
            observation,
            now,
            frame_id,
            required_hits,
            used_candidates,
        )

    def update(self, result, now: float, source_width=None, source_height=None) -> None:
        self._ownership_quarantine = set()
        super().update(result, now, source_width, source_height)

    def snapshot(self):
        rows = super().snapshot()
        for row in rows:
            track_id = int(row.get("track_id") or 0)
            row["camera_id"] = self.camera_id
            row["display_id"] = self.display_label(track_id)
            row["local_sequence"] = max(0, track_id - self.id_namespace_base)
        return rows

    def metrics(self):
        payload = super().metrics()
        payload.update(
            {
                "algorithm": "adaptive-kalman-bytetrack-hungarian-ownership-v4",
                "ownership_lock": self.ownership_lock,
                "ownership_rejects": self._ownership_rejects,
                "ownership_birth_blocks": self._ownership_birth_blocks,
                "ownership_clear_matches": self._ownership_clear_matches,
                "ownership_quarantined_now": len(self._ownership_quarantine),
                "id_namespace_base": self.id_namespace_base,
                "id_namespace_stride": self.id_namespace_stride,
            }
        )
        return payload
