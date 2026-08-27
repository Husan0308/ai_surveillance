from __future__ import annotations

import threading
import time
from collections import Counter, deque

from .runtime_v100_tracker_timeout40 import PascalTrackerTimeout40Runtime


class PascalTrackerBatchAuditRuntime(PascalTrackerTimeout40Runtime):
    """V10.1: diagnostic-only tracker mux batch formation audit.

    V10.0 reduced tracker_mux batched-push-timeout to 40 ms.  Gate->mux latency
    improved but remained too high and NvDCF p95 increased.  Keep every V10.0
    behavior unchanged and inspect exactly what tracker_mux is emitting:

      * frames filled per output batch,
      * unique source count per batch,
      * PTS spread inside a batch,
      * mux output cadence,
      * per-source inclusion ratio.

    This tells us whether the remaining delay is caused by waiting for full
    six-source batches, persistently partial/imbalanced batches, or downstream
    backpressure even when batches are already full.
    """

    def __init__(self) -> None:
        self._v101_lock = threading.RLock()
        self._v101_batch_sizes: deque[int] = deque(maxlen=4096)
        self._v101_unique_sizes: deque[int] = deque(maxlen=4096)
        self._v101_pts_spread_ms: deque[float] = deque(maxlen=4096)
        self._v101_output_dt_ms: deque[float] = deque(maxlen=4096)
        self._v101_source_hits: Counter[int] = Counter()
        self._v101_batches = 0
        self._v101_last_output_mono: float | None = None
        super().__init__()
        print(
            "CAMERA_V101_ARCH behavior_change=0 only_change=tracker-batch-audit "
            "tracker_timeout_us=40000 batch_target=6 exact_pts=1",
            flush=True,
        )

    def _inject_detector_probe(self, pad, info):
        buffer = info.get_buffer()
        rows = self._copy_pts(buffer)
        now = time.monotonic()
        valid = [row for row in rows if self._valid_pts(int(row["buf_pts"]))]
        source_ids = [int(row["source_id"]) for row in valid]
        pts_values = [int(row["buf_pts"]) for row in valid]

        with self._v101_lock:
            self._v101_batches += 1
            self._v101_batch_sizes.append(len(valid))
            unique = set(source_ids)
            self._v101_unique_sizes.append(len(unique))
            for sid in unique:
                self._v101_source_hits[sid] += 1
            if pts_values:
                spread = (max(pts_values) - min(pts_values)) / 1_000_000.0
                if 0.0 <= spread <= 5000.0:
                    self._v101_pts_spread_ms.append(float(spread))
            if self._v101_last_output_mono is not None:
                dt = (now - self._v101_last_output_mono) * 1000.0
                if 0.0 <= dt <= 5000.0:
                    self._v101_output_dt_ms.append(float(dt))
            self._v101_last_output_mono = now

        return super()._inject_detector_probe(pad, info)

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._v101_lock:
            sizes = list(self._v101_batch_sizes)
            unique_sizes = list(self._v101_unique_sizes)
            spreads = list(self._v101_pts_spread_ms)
            output_dt = list(self._v101_output_dt_ms)
            batches = int(self._v101_batches)
            hits = dict(self._v101_source_hits)

        if not sizes or batches <= 0:
            print("CAMERA_V101_BATCH samples=0", flush=True)
            return keep

        full = sum(1 for x in sizes if x >= len(self.cameras))
        partial = sum(1 for x in sizes if x < len(self.cameras))
        full_pct = 100.0 * full / len(sizes)
        partial_pct = 100.0 * partial / len(sizes)
        source_parts = []
        for source_id in sorted(self.index_camera):
            cid = self.index_camera[source_id]
            ratio = 100.0 * hits.get(source_id, 0) / max(1, len(sizes))
            source_parts.append(f"{cid}:{ratio:.0f}%")

        print(
            "CAMERA_V101_BATCH "
            f"samples={len(sizes)} target={len(self.cameras)} "
            f"size_p50={self._pct(sizes, 0.50):.0f} size_p95={self._pct(sizes, 0.95):.0f} "
            f"unique_p50={self._pct(unique_sizes, 0.50):.0f} unique_p95={self._pct(unique_sizes, 0.95):.0f} "
            f"full_pct={full_pct:.1f} partial_pct={partial_pct:.1f} "
            f"pts_spread_p50={self._pct(spreads, 0.50):.0f}ms pts_spread_p95={self._pct(spreads, 0.95):.0f}ms "
            f"output_dt_p50={self._pct(output_dt, 0.50):.0f}ms output_dt_p95={self._pct(output_dt, 0.95):.0f}ms "
            f"source_hit={'/'.join(source_parts)}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalTrackerBatchAuditRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
