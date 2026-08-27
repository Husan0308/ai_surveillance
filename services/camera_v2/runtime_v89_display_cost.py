from __future__ import annotations

from .runtime_v85_nvdcf_relief import PascalNvDCFReliefRuntime


class PascalDisplayCostProbeRuntime(PascalNvDCFReliefRuntime):
    """V8.9 diagnostic: keep analytics/detector, remove visual compositor GPU work.

    This is intentionally NOT a production UI configuration.  It isolates the
    cost of nvmultistreamtiler + nvvideoconvert + nvdsosd + EGL rendering while
    keeping the display mux/source branch alive.  Detector, NvDCF cadence,
    detector budget, confidence and bbox policy remain the V8.5 baseline.
    """

    def _make(self, factory: str, name: str):
        if name == "display_sink":
            return super()._make("fakesink", name)
        return super()._make(factory, name)

    def _configure_display_path(self) -> None:
        # V8.9 A/B: no tiler, wall convert, OSD or EGL rendering.  The display
        # mux still consumes source frames so camera/source behavior is comparable.
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "async", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "enable-last-sample", False)

    def _link_display_path(self) -> None:
        self._require_link(self.display_mux, self.sink, "display mux -> V89 fakesink")

    def _audit_graph(self) -> None:
        display_peer = self.display_mux.get_static_pad("src").get_peer()
        tracker_peer = self.tracker_mux.get_static_pad("src").get_peer()
        display_name = (
            display_peer.get_parent_element().get_name() if display_peer is not None else None
        )
        tracker_name = (
            tracker_peer.get_parent_element().get_name() if tracker_peer is not None else None
        )
        if display_name != self.sink.get_name():
            raise RuntimeError(
                f"CAMERA_V89_AUDIT display mux peer={display_name}, expected={self.sink.get_name()}"
            )
        if tracker_name != self.tracker.get_name():
            raise RuntimeError(
                f"CAMERA_V89_AUDIT tracker mux peer={tracker_name}, expected={self.tracker.get_name()}"
            )
        print(
            "CAMERA_V89_AUDIT display=display_mux->fakesink "
            "tiler=0 osd=0 egl=0 tracker=NvDCF detector=TRT8.6-sidecar",
            flush=True,
        )

    def __init__(self) -> None:
        super().__init__()
        print(
            "CAMERA_V89_ARCH mode=display-cost-ab display_mux=1 tiler=0 wall_convert=0 "
            "osd=0 egl=0 fakesink=1 tracker_quality=v85 detector=batch1-v84 "
            "bbox_policy=unchanged production_ui=0",
            flush=True,
        )

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_V89_STATS "
            f"headless_compositor=1 gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
            f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
            f"tracked_now={self.tracked_now} tracker_batches={self.tracker_batches} "
            "display_mux=1 tiler=0 osd=0 egl=0",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalDisplayCostProbeRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
