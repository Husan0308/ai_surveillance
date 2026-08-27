from __future__ import annotations

from .runtime_v85_nvdcf_relief import PascalNvDCFReliefRuntime


class PascalTracker480Runtime(PascalNvDCFReliefRuntime):
    """V8.7: single-change NvDCF tracker-resolution A/B.

    Baseline is V8.5: TRT8.6 batch-1 detector, NvDCF featureImgSizeLevel=1,
    ColorNames on, HOG off, 8 Hz tracker cadence, original bbox/confidence/hold
    policy and default DeepStream CUDA scheduling. The launcher changes only the
    tracker surface from 512x288 to 480x288.
    """

    def __init__(self) -> None:
        super().__init__()
        if self.track_width != 480 or self.track_height != 288:
            raise RuntimeError(
                f"V87 requires tracker=480x288, got {self.track_width}x{self.track_height}"
            )
        print(
            "CAMERA_V87_ARCH "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            "baseline=v85 feature_level=1 detector=batch1-v84 "
            "blocking_sync_override=0 bbox_unchanged=1 confidence_unchanged=1 "
            "cadence_unchanged=1 only_change=tracker-resolution-512x288-to-480x288",
            flush=True,
        )

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_V87_STATS "
            f"tracker={self.track_width}x{self.track_height} "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
            f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
            f"tracked_now={self.tracked_now} tracker_batches={self.tracker_batches}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalTracker480Runtime().run()


if __name__ == "__main__":
    raise SystemExit(main())
