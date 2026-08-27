from __future__ import annotations

from .runtime_v101_tracker_batch_audit import PascalTrackerBatchAuditRuntime


class PascalTrackerBatch4Runtime(PascalTrackerBatchAuditRuntime):
    """V10.2: make the tracker mux target match the batch it actually forms.

    V10.1 showed that with six configured sources and a 40 ms timeout the
    tracker mux emits a median of four frames per batch, 82.8% of batches are
    partial, and only ~17% reach six frames.  V10.2 changes one property only:

        tracker_mux batch-size: 6 -> 4

    The 40 ms timeout, tracker cadence/resolution, NvDCF, detector, display,
    bbox policy, and all PTS instrumentation remain unchanged.  This is an A/B
    test: if the mux is already naturally forming four-frame batches, making
    four the completion target should release those batches without waiting for
    the six-frame target/timeout tail.
    """

    TARGET_BATCH = 4

    def _configure_tracker_mux(self) -> None:
        super()._configure_tracker_mux()
        self._set_if(self.tracker_mux, "batch-size", self.TARGET_BATCH)

    def __init__(self) -> None:
        super().__init__()
        batch_size = None
        timeout_us = None
        try:
            batch_size = int(self.tracker_mux.get_property("batch-size"))
        except Exception:
            pass
        try:
            timeout_us = int(self.tracker_mux.get_property("batched-push-timeout"))
        except Exception:
            pass
        print(
            "CAMERA_V102_ARCH only_change=tracker-mux-batch-size-6-to-4 "
            f"tracker_batch_size={batch_size} target=4 tracker_timeout_us={timeout_us} "
            "display_pool=4 tracker_pool=4 tracker=512x288@10Hz pts_audit=v99 batch_audit=v101",
            flush=True,
        )
        if batch_size is not None and batch_size != self.TARGET_BATCH:
            raise RuntimeError(
                f"V10.2 expected tracker_mux batch-size=4, got {batch_size}"
            )
        if timeout_us is not None and timeout_us != 40_000:
            raise RuntimeError(
                f"V10.2 expected tracker_mux batched-push-timeout=40000, got {timeout_us}"
            )

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._v101_lock:
            sizes = list(self._v101_batch_sizes)
            unique_sizes = list(self._v101_unique_sizes)
            spreads = list(self._v101_pts_spread_ms)
            output_dt = list(self._v101_output_dt_ms)
            hits = dict(self._v101_source_hits)

        if not sizes:
            print("CAMERA_V102_BATCH samples=0", flush=True)
            return keep

        full = sum(1 for x in sizes if x >= self.TARGET_BATCH)
        partial = sum(1 for x in sizes if x < self.TARGET_BATCH)
        source_parts = []
        for source_id in sorted(self.index_camera):
            cid = self.index_camera[source_id]
            ratio = 100.0 * hits.get(source_id, 0) / max(1, len(sizes))
            source_parts.append(f"{cid}:{ratio:.0f}%")

        print(
            "CAMERA_V102_BATCH "
            f"samples={len(sizes)} target={self.TARGET_BATCH} "
            f"size_p50={self._pct(sizes, 0.50):.0f} size_p95={self._pct(sizes, 0.95):.0f} "
            f"unique_p50={self._pct(unique_sizes, 0.50):.0f} unique_p95={self._pct(unique_sizes, 0.95):.0f} "
            f"full_pct={100.0 * full / len(sizes):.1f} partial_pct={100.0 * partial / len(sizes):.1f} "
            f"pts_spread_p50={self._pct(spreads, 0.50):.0f}ms pts_spread_p95={self._pct(spreads, 0.95):.0f}ms "
            f"output_dt_p50={self._pct(output_dt, 0.50):.0f}ms output_dt_p95={self._pct(output_dt, 0.95):.0f}ms "
            f"source_hit={'/'.join(source_parts)}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalTrackerBatch4Runtime().run()


if __name__ == "__main__":
    raise SystemExit(main())
