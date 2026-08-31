from __future__ import annotations

import time

from .step4_reid_same_room_matcher_v1 import MATCH_PROPOSED
from .step4_reid_same_room_shadow_cached_v3 import V11SameRoomMatcherShadowWorkerCachedV3
from .step5_global_shadow_worker_v1 import V11GlobalShadowWorkerV1


class V11SameRoomMatcherShadowWorkerStep5TapV1(
    V11SameRoomMatcherShadowWorkerCachedV3
):
    """Step4 matcher with an O(1), bounded Step5 proposal tap.

    Step4 matching semantics and timing remain unchanged. The tap only enqueues
    compact cycle/proposal messages to a separate Step5 worker; state handling and
    TSV I/O happen off the matcher thread.
    """

    def __init__(
        self,
        *args,
        global_shadow_worker: V11GlobalShadowWorkerV1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.global_shadow_worker = global_shadow_worker

    def _match_cycle(self) -> None:
        next_cycle = int(self.cycles) + 1
        self.global_shadow_worker.enqueue_cycle_start(next_cycle, time.time_ns())
        try:
            super()._match_cycle()
        finally:
            self.global_shadow_worker.enqueue_cycle_end(next_cycle, time.time_ns())

    def _tsv_row(self, timestamp_ns, cycle, row, stability):
        result = super()._tsv_row(timestamp_ns, cycle, row, stability)
        if row.status == MATCH_PROPOSED and row.reciprocal and row.assigned:
            self.global_shadow_worker.enqueue_proposal(cycle, timestamp_ns, row)
        return result
