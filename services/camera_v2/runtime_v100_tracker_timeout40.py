from __future__ import annotations

from .runtime_v99_tracker_latency_audit import PascalTrackerLatencyAuditRuntime


class PascalTrackerTimeout40Runtime(PascalTrackerLatencyAuditRuntime):
    """V10.0: fix only tracker_mux batch formation wait.

    V9.9 proved source->gate is fresh (p95 0 ms), while gate->tracker_mux is the
    dominant stale stage (median ~172 ms, max ~251 ms).  Keep V9.8/V9.9 display,
    tracker, NvDCF, detector, bbox and PTS behavior unchanged; only reduce the
    legacy nvstreammux batched-push-timeout from the 10 Hz period (~100 ms) to
    40 ms so incomplete live batches are released sooner.
    """

    def _configure_tracker_mux(self) -> None:
        super()._configure_tracker_mux()
        self._set_if(self.tracker_mux, "batched-push-timeout", 40_000)

    def __init__(self) -> None:
        super().__init__()
        timeout_us = None
        try:
            timeout_us = int(self.tracker_mux.get_property("batched-push-timeout"))
        except Exception:
            pass
        print(
            "CAMERA_V100_ARCH only_change=tracker-mux-batched-push-timeout "
            f"tracker_timeout_us={timeout_us} target_us=40000 "
            "display_pool=4 tracker_pool=4 tracker=512x288@10Hz pts_audit=v99",
            flush=True,
        )
        if timeout_us is not None and timeout_us != 40_000:
            raise RuntimeError(
                f"V10.0 expected tracker_mux batched-push-timeout=40000, got {timeout_us}"
            )


def main() -> int:
    return PascalTrackerTimeout40Runtime().run()


if __name__ == "__main__":
    raise SystemExit(main())
