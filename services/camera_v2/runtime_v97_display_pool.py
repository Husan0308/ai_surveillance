from __future__ import annotations

from .runtime_v96_display_freshness import PascalDisplayFreshnessRuntime


class PascalDisplayPoolRuntime(PascalDisplayFreshnessRuntime):
    """V9.7: bound display_mux in-flight output buffers only.

    V9.6 moved the one-buffer downstream-leaky queue directly after display_mux,
    but source->display P95 still remained about 200-500 ms.  The display mux was
    still configured with buffer-pool-size=8.  For the legacy nvstreammux used by
    this runtime, buffer-pool-size is the number of mux output buffers that may be
    in flight.  V9.7 restores only the display mux pool to NVIDIA's documented
    default of 4 while leaving the tracker mux at 8 and every V9.6 behavior intact.
    """

    def _configure_display_mux(self) -> None:
        super()._configure_display_mux()
        # One-variable A/B: display mux output pool 8 -> 4.
        self._set_if(self.display_mux, "buffer-pool-size", 4)

    def __init__(self) -> None:
        super().__init__()
        pool = None
        try:
            pool = int(self.display_mux.get_property("buffer-pool-size"))
        except Exception:
            pass
        tracker_pool = None
        try:
            tracker_pool = int(self.tracker_mux.get_property("buffer-pool-size"))
        except Exception:
            pass
        print(
            "CAMERA_V97_ARCH only_change=display-mux-buffer-pool "
            f"display_pool={pool} tracker_pool={tracker_pool} "
            "display_queue=after-mux,max1,leaky-downstream pts_audit=v95",
            flush=True,
        )
        if pool is not None and pool != 4:
            raise RuntimeError(f"V9.7 expected display_mux buffer-pool-size=4, got {pool}")


def main() -> int:
    return PascalDisplayPoolRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
