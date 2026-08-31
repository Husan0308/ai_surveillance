from __future__ import annotations

import csv
import time

from .step4_reid_pair_score_cache_v1 import score_gallery_pair_step3_cached_v1
from .step4_reid_pair_shadow_v1 import V11GalleryPairShadowWorkerV1


class V11GalleryPairShadowWorkerCachedV2(V11GalleryPairShadowWorkerV1):
    """Step-3 shadow scorer that publishes exact reusable pair evidence.

    The authoritative Step-3 scorer still runs on every cache miss.  Results are
    keyed by immutable gallery sample fingerprints so the later Step-4 matcher can
    consume the exact same GalleryPairScoreV1 instead of recomputing it.
    """

    def _score_cycle(self) -> None:
        views = self.snapshot_provider()
        tsv_rows: list[tuple[str, ...]] = []
        for _priority, first, second, context, fingerprint in self._candidates(views):
            pair_key = self._pair_key(first, second)
            self._fingerprints[pair_key] = fingerprint
            with self._cv:
                self.pairs_considered += 1
                if context == "same_room":
                    self.same_room_pairs += 1
                else:
                    self.different_room_pairs += 1

            started_ns = time.perf_counter_ns()
            score = score_gallery_pair_step3_cached_v1(first, second)
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0

            with self._cv:
                self.score_ms.append(elapsed_ms)
                if score.status == "OK":
                    self.pairs_scored += 1
                elif score.status == "INSUFFICIENT":
                    self.pairs_insufficient += 1
                else:
                    self.pairs_invalid += 1

            tsv_row = self._tsv_row(first, second, context, score)
            if tsv_row is not None:
                tsv_rows.append(tsv_row)

        if self.tsv_path is not None and tsv_rows:
            with self.tsv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(
                    tsv_rows
                )
