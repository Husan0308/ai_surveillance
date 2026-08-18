from __future__ import annotations

from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .person_tracking_reid import CameraPersonTrackingReID


class CameraPersonTrackingReIDHeatmap(
    CameraPersonTrackingReID,
    CameraPersonTrackingHeatmap,
):
    """ReID runtime with the existing native camera-space heatmap mixed in.

    Python MRO is intentional:
    ReID tracker probe -> Heatmap tracker probe -> stable NvDCF tracker probe.
    This preserves the known live camera/tracking path while adding heatmap
    accumulation and per-camera render visibility as a side concern.
    """

    pass


def main() -> int:
    return CameraPersonTrackingReIDHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
