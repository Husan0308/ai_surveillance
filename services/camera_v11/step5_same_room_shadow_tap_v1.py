from __future__ import annotations

import time

from .step4_reid_camera_tracklet_v1 import CameraTrackletContinuityV1
from .step4_reid_same_room_matcher_v1 import MATCH_PROPOSED
from .step4_reid_same_room_shadow_cached_v3 import V11SameRoomMatcherShadowWorkerCachedV3
from .step5_global_shadow_worker_v1 import V11GlobalShadowWorkerV1


class V11SameRoomMatcherShadowWorkerStep5TapV1(
    V11SameRoomMatcherShadowWorkerCachedV3
):
    """Step4 matcher with bounded Step4.5 continuity + Step5 proposal tap.

    Step4 still scores raw local T-IDs. Before a MATCH_PROPOSED row reaches Step5,
    CAM-01/CAM-04 raw fragments are translated to stable camera-tracklet IDs by
    the same-camera continuity layer. A not-yet-proven fragment is suppressed
    instead of being emitted as a conflicting new Global Shadow pair.
    """

    def __init__(
        self,
        *args,
        global_shadow_worker: V11GlobalShadowWorkerV1,
        camera_tracklet_continuity: CameraTrackletContinuityV1 | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.global_shadow_worker = global_shadow_worker
        self.camera_tracklet_continuity = camera_tracklet_continuity

    def _match_cycle(self) -> None:
        next_cycle = int(self.cycles) + 1
        if self.camera_tracklet_continuity is not None:
            self.camera_tracklet_continuity.refresh(next_cycle, time.monotonic_ns())
        self.global_shadow_worker.enqueue_cycle_start(next_cycle, time.time_ns())
        try:
            super()._match_cycle()
        finally:
            self.global_shadow_worker.enqueue_cycle_end(next_cycle, time.time_ns())

    def _tsv_row(self, timestamp_ns, cycle, row, stability):
        result = super()._tsv_row(timestamp_ns, cycle, row, stability)
        if row.status == MATCH_PROPOSED and row.reciprocal and row.assigned:
            proposal = row
            if self.camera_tracklet_continuity is not None:
                proposal = self.camera_tracklet_continuity.canonicalize_proposal(row)
            if proposal is not None:
                self.global_shadow_worker.enqueue_proposal(cycle, timestamp_ns, proposal)
        return result
