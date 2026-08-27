from __future__ import annotations

import os

from .runtime_bbox_v72 import NvDCFStableBBoxRuntime


class PascalBalancedBBoxRuntime(NvDCFStableBBoxRuntime):
    """V7.4 production profile for GTX 1050 Ti / Pascal-class GPUs.

    Keep the V7.2 no-flicker bbox semantics and V7 serialized detector scheduling,
    but reduce NvDCF visual-feature cost. NVIDIA documents that HOG uses more feature
    channels than ColorNames and that larger featureImgSizeLevel values trade GPU
    performance for robustness. On a 4 GB GP107/SM 6.1 GPU, the previous HOG=1,
    feature-level=3, 20 Hz target overloaded the device and starved both TRT and NvDCF.
    """

    def _prepare_tracker_files(self):
        lib, generated = super()._prepare_tracker_files()
        lines = generated.read_text(encoding="utf-8").splitlines()

        # DeepStream's current NvDCF defaults are the right starting point for this
        # low-end Pascal GPU: ColorNames only and level-2 feature images. This cuts
        # feature channels and feature pixels substantially versus V7's HOG/level-3.
        self._replace_yaml_key(lines, "useColorNames", "1", required=False)
        self._replace_yaml_key(lines, "useHog", "0", required=False)
        self._replace_yaml_key(lines, "featureImgSizeLevel", "2", required=False)
        self._replace_yaml_key(lines, "useHighPrecisionFeature", "0", required=False)
        self._replace_yaml_key(lines, "searchRegionPaddingScale", "1", required=False)

        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_BBOX_V74_NVDCF "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            "features=ColorNames hog=0 feature_level=2 high_precision=0 "
            "search_padding=1 bbox_policy=v7.2-no-flicker",
            flush=True,
        )
        return lib, generated

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_BBOX_V74_STATS "
            f"trt_baseline_ms={os.environ.get('CAMERA_V2_TRT_BASELINE_MS', 'unknown')} "
            f"detector_target_hz={self.detect_hz:.2f} "
            f"tracker_target_hz={self.track_fps:.1f} "
            "nvdcf_features=colornames/level2/hog0",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalBalancedBBoxRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
