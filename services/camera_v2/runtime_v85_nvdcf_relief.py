from __future__ import annotations

import os

from .runtime_v84_batch1 import PascalBatch1LowLatencyRuntime


class PascalNvDCFReliefRuntime(PascalBatch1LowLatencyRuntime):
    """V8.5 Step 3: reduce only NvDCF visual-feature GPU cost.

    V8.4 proved that the TRT8.6 batch-1 engine is ~60 ms in clean-room but rises
    to ~170-185 ms while DeepStream/NvDCF is active.  This A/B keeps the V8.4
    detector scheduler, RTSP/display path, tracker cadence/resolution, bbox policy,
    confidence thresholds and hold policy unchanged.  The only tracking-compute
    change is NvDCF featureImgSizeLevel 2 -> 1, reducing each ColorNames feature
    image from 18x18 to 12x12.
    """

    def __init__(self) -> None:
        self.v85_feature_level = max(
            1,
            min(5, int(os.environ.get("CAMERA_V85_NVDCF_FEATURE_LEVEL", "1"))),
        )
        super().__init__()
        print(
            "CAMERA_V85_ARCH "
            f"nvdcf_feature_level={self.v85_feature_level} "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            "color_names=1 hog=0 tracker_resolution_unchanged=1 tracker_rate_unchanged=1 "
            "detector=batch1-v84-unchanged bbox_policy=unchanged confidence=unchanged "
            "cuda_blocking_sync=unchanged",
            flush=True,
        )

    def _prepare_tracker_files(self):
        lib, generated = super()._prepare_tracker_files()
        lines = generated.read_text(encoding="utf-8").splitlines()
        self._replace_yaml_key(
            lines,
            "featureImgSizeLevel",
            str(self.v85_feature_level),
            required=False,
        )
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_V85_NVDCF "
            f"featureImgSizeLevel={self.v85_feature_level} "
            "feature_pixels=12x12 color_names=1 hog=0 high_precision=0 "
            "change_scope=feature-size-only",
            flush=True,
        )
        return lib, generated

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_V85_STATS "
            f"feature_level={self.v85_feature_level} "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
            f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
            f"tracked_now={self.tracked_now} tracker_batches={self.tracker_batches}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalNvDCFReliefRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
