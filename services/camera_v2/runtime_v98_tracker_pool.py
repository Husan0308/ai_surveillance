from __future__ import annotations

from .runtime_v97_display_pool import PascalDisplayPoolRuntime


class PascalTrackerPoolRuntime(PascalDisplayPoolRuntime):
    """V9.8: reduce only tracker_mux in-flight output buffers.

    V9.7 materially reduced source->display P95 by restoring display_mux's
    output pool from 8 to 4.  The PTS audit still shows source->tracker P95 at
    roughly 200-400 ms while tracker cadence itself remains ~10 Hz.

    V9.8 applies the same one-variable test to tracker_mux only:
        tracker_mux buffer-pool-size: 8 -> 4

    Display V9.7 behavior, NvDCF configuration, tracker rate/resolution,
    detector/TRT, bbox policy, compensation, and PTS instrumentation remain
    unchanged.
    """

    def _configure_tracker_mux(self) -> None:
        super()._configure_tracker_mux()
        self._set_if(self.tracker_mux, "buffer-pool-size", 4)

    def __init__(self) -> None:
        super().__init__()
        display_pool = None
        tracker_pool = None
        try:
            display_pool = int(self.display_mux.get_property("buffer-pool-size"))
        except Exception:
            pass
        try:
            tracker_pool = int(self.tracker_mux.get_property("buffer-pool-size"))
        except Exception:
            pass
        print(
            "CAMERA_V98_ARCH only_change=tracker-mux-buffer-pool "
            f"display_pool={display_pool} tracker_pool={tracker_pool} "
            "tracker=512x288@10Hz display=v97 pts_audit=v95",
            flush=True,
        )
        if tracker_pool is not None and tracker_pool != 4:
            raise RuntimeError(
                f"V9.8 expected tracker_mux buffer-pool-size=4, got {tracker_pool}"
            )


def main() -> int:
    return PascalTrackerPoolRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
