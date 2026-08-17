from __future__ import annotations

from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .person_tracking_qwen import CameraPersonTrackingQwen


class CameraPersonTrackingQwenHeatmap(CameraPersonTrackingHeatmap, CameraPersonTrackingQwen):
    """Full Camera V2 runtime: detection + NvDCF + ReID + Qwen + heatmap.

    Cooperative MRO intentionally layers Heatmap -> Qwen -> Final. Heatmap paints
    from raw NvDCF metadata first, Qwen audits global identity second, and the base
    runtime keeps detector/tracker scheduling unchanged.
    """


def main() -> int:
    return CameraPersonTrackingQwenHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
