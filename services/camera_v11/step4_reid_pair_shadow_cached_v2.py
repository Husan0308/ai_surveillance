from __future__ import annotations

import csv
import time

from .step4_reid_pair_shadow_v1 import V11GalleryPairShadowWorkerV1
from .step4_reid_same_room_evidence_padded_v1 import (
    score_gallery_matrix_step3_padded_exact_v1,
)


class V11GalleryPairShadowWorkerCachedV2(V11GalleryPairShadowWorkerV1):
    """Step-3 shadow scorer reusing the validated Step-4 gallery-matrix cache.

    This preserves the authoritative Step-3 score formula exactly while avoiding
    repeated embedding conversion, finite/norm checks, copies and stacking for
    every candidate pair.  Using the same cache as the Step-4 same-room matcher
    also primes gallery matrices before the matcher runs, reducing duplicate CPU
    work without changing thresholds or identity semantics.
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
            score = score_gallery_matrix_step3_padded_exact_v1(
                (first,),
                (second,),
            )[(0, 0)]
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
