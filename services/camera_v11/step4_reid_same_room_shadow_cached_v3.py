from __future__ import annotations

import time
from dataclasses import replace

from .step4_reid_pair_score_cache_v1 import lookup_gallery_pair_step3_cached_v1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1
from .step4_reid_same_room_matcher_v1 import match_same_room_camera_pair_v1
from .step4_reid_same_room_shadow_v1 import V11SameRoomMatcherShadowWorkerV1


class V11SameRoomMatcherShadowWorkerCachedV3(V11SameRoomMatcherShadowWorkerV1):
    """Step4 matcher that only consumes already-published Step3 evidence.

    A Step4 cache miss must never run the Step3 scorer again.  If the exact
    current-gallery pair score is not published yet, the edge is temporarily
    reported as INSUFFICIENT/evidence-pending and retried on the next dirty cycle.
    This preserves the layer boundary and removes cache-miss scoring from matcher
    tail latency. Whole-matrix results are cached only when every edge had exact
    Step3 evidence, so a pending result can never get stuck behind matrix caching.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evidence_cache_hits = 0
        self.evidence_cache_misses = 0

    def _lookup_only_pair_score(self, first, second) -> GalleryPairScoreV1:
        cached = lookup_gallery_pair_step3_cached_v1(first, second)
        if cached is not None:
            self.evidence_cache_hits += 1
            return cached
        self.evidence_cache_misses += 1
        return GalleryPairScoreV1(
            status="INSUFFICIENT",
            pair_count=len(first.samples) * len(second.samples),
            reason="exact Step3 pair evidence pending",
        )

    def _matrix_result(self, camera_a, camera_b, left, right, now_ns):
        started_ns = time.perf_counter_ns()
        recent_ns = int(self.config.recent_age_sec * 1e9)
        fingerprint = tuple(
            (
                view.camera_id,
                view.local_track_id,
                tuple(sample.sample_sequence for sample in view.samples),
                now_ns - int(view.last_seen_ns) > recent_ns,
            )
            for view in sorted(
                (*left, *right),
                key=lambda item: (item.camera_id, item.local_track_id),
            )
        )
        key = (camera_a, camera_b)
        cached_matrix = self._matrix_cache.get(key)
        if cached_matrix is not None and cached_matrix[0] == fingerprint:
            return replace(
                cached_matrix[1],
                elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
            )

        misses_before = self.evidence_cache_misses
        result = match_same_room_camera_pair_v1(
            left,
            right,
            self.camera_rooms,
            now_ns=now_ns,
            config=self.config,
            pair_scorer=self._lookup_only_pair_score,
        )
        if self.evidence_cache_misses == misses_before:
            self._matrix_cache[key] = (fingerprint, result)
        else:
            # Evidence may be published by Step3 shortly after this cycle. Never
            # cache a matrix containing evidence-pending placeholders.
            self._matrix_cache.pop(key, None)
        return result

    def snapshot(self) -> dict[str, int | float]:
        row = super().snapshot()
        row["evidence_cache_hits"] = self.evidence_cache_hits
        row["evidence_cache_misses"] = self.evidence_cache_misses
        return row
