from __future__ import annotations

import os

from .runtime_bbox_v7 import NvDCFStickyBBoxRuntime


class NvDCFProductionBBoxRuntime(NvDCFStickyBBoxRuntime):
    """Final V7 production profile after official DeepStream parameter review.

    The V7 temporal/localization logic stays unchanged. This thin profile override fixes
    target-creation semantics according to NVIDIA's current NvDCF documentation:
    lower minIouDiff4NewTarget suppresses duplicate targets, while a conservative
    minDetectorConfidence prevents weak detector noise from minting tracker objects.
    """

    def _prepare_tracker_files(self):
        lib, generated = super()._prepare_tracker_files()
        lines = generated.read_text(encoding="utf-8").splitlines()
        min_detector_conf = os.environ.get("CAMERA_V2_NVDCF_MIN_DETECTOR_CONF", "0.18")
        min_iou_diff = os.environ.get("CAMERA_V2_NVDCF_MIN_IOU_DIFF_NEW_TARGET", "0.22")
        self._replace_yaml_key(lines, "minDetectorConfidence", min_detector_conf)
        self._replace_yaml_key(lines, "minIouDiff4NewTarget", min_iou_diff)
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_BBOX_V7_TARGET_POLICY "
            f"min_detector_conf={min_detector_conf} "
            f"min_iou_diff_new_target={min_iou_diff} duplicate_creation=conservative",
            flush=True,
        )
        return lib, generated


def main() -> int:
    return NvDCFProductionBBoxRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
