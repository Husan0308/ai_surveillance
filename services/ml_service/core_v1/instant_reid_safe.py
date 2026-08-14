from __future__ import annotations

import numpy as np

from .instant_reid import InstantGlobalReIdCoordinator
from .local_tracker import linear_sum_assignment


class SafeInstantGlobalReIdCoordinator(InstantGlobalReIdCoordinator):
    """Instant ReID policy with a hard no-merge result on camera conflicts."""

    def _merge_global_ids(self, gid_a: str, gid_b: str, now: float, similarity: float):
        a = self._canonical_gid(gid_a) or gid_a
        b = self._canonical_gid(gid_b) or gid_b
        if a == b:
            return a
        ia = self._globals.get(a)
        ib = self._globals.get(b)
        if ia is None or ib is None:
            return None
        target, source = (ia, ib) if ia.created_at <= ib.created_at else (ib, ia)
        target_cameras = {
            camera_id for camera_id, _ in self._active_bindings(target.global_id, now)
        }
        source_cameras = {
            camera_id for camera_id, _ in self._active_bindings(source.global_id, now)
        }
        if target_cameras & source_cameras:
            return None
        return super()._merge_global_ids(gid_a, gid_b, now, similarity)

    def _reconcile_overlap_pairs(self, now: float):
        for group in self.overlap_groups:
            cameras = sorted(group)
            if len(cameras) != 2:
                continue
            left = [
                state
                for state in self._tracks.values()
                if state.camera_id == cameras[0]
                and state.embeddings
                and state.global_id
                and now - state.last_seen <= self.active_timeout
            ]
            right = [
                state
                for state in self._tracks.values()
                if state.camera_id == cameras[1]
                and state.embeddings
                and state.global_id
                and now - state.last_seen <= self.active_timeout
            ]
            if not left or not right:
                continue

            similarities = np.zeros((len(left), len(right)), dtype=np.float32)
            for i, a in enumerate(left):
                pa = self._tracklet_prototype(a)
                for j, b in enumerate(right):
                    similarities[i, j] = self._cosine_pair(pa, self._tracklet_prototype(b))

            rows, cols = linear_sum_assignment(1.0 - similarities)
            single = len(left) == 1 and len(right) == 1
            for row, col in zip(rows.tolist(), cols.tolist()):
                sim = float(similarities[row, col])
                a, b = left[row], right[col]
                gid_a = self._canonical_gid(a.global_id)
                gid_b = self._canonical_gid(b.global_id)
                if not gid_a or not gid_b or gid_a == gid_b:
                    continue
                both_confirmed = (
                    gid_a not in self._provisional_globals
                    and gid_b not in self._provisional_globals
                )
                threshold = (
                    self.confirmed_pair_merge_similarity
                    if both_confirmed
                    else (self.fast_single_similarity if single else self.fast_pair_similarity)
                )
                if sim < threshold:
                    continue
                row_scores = np.sort(similarities[row])[::-1]
                col_scores = np.sort(similarities[:, col])[::-1]
                row_margin = sim - float(row_scores[1]) if len(row_scores) > 1 else 1.0
                col_margin = sim - float(col_scores[1]) if len(col_scores) > 1 else 1.0
                if min(row_margin, col_margin) < self.fast_pair_margin:
                    continue
                merged = self._merge_global_ids(gid_a, gid_b, now, sim)
                if merged is None:
                    continue
                a.global_id = merged
                b.global_id = merged
                a.assignment_similarity = sim
                b.assignment_similarity = sim
                a.assignment_reason = "pair_global_assignment"
                b.assignment_reason = "pair_global_assignment"
                self._pair_reconciles += 1

    @staticmethod
    def _cosine_pair(a, b) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an <= 1e-9 or bn <= 1e-9:
            return -1.0
        return float(np.dot(a / an, b / bn))
