from __future__ import annotations

import time
from dataclasses import replace

from .step4_reid_pair_score_cache_v1 import score_gallery_pair_step3_cached_v1
from .step4_reid_same_room_matcher_v1 import match_same_room_camera_pair_v1
from .step4_reid_same_room_shadow_v1 import V11SameRoomMatcherShadowWorkerV1


class V11SameRoomMatcherShadowWorkerCachedV2(V11SameRoomMatcherShadowWorkerV1):
    """Step4 matcher consuming the exact Step3 pair evidence cache.

    Step3's pair shadow worker runs first after each ReID update.  This worker is
    phase-delayed, so the normal path reads already-computed GalleryPairScoreV1
    objects instead of recomputing the Step3 embedding statistics.  Cache misses
    still call the authoritative Step3 scorer, preserving exact semantics.
    """

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
        cached = self._matrix_cache.get(key)
        if cached is not None and cached[0] == fingerprint:
            return replace(
                cached[1],
                elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
            )

        result = match_same_room_camera_pair_v1(
            left,
            right,
            self.camera_rooms,
            now_ns=now_ns,
            config=self.config,
            pair_scorer=score_gallery_pair_step3_cached_v1,
        )
        self._matrix_cache[key] = (fingerprint, result)
        return result
