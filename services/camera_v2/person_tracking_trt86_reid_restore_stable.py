from __future__ import annotations

# Import the TRT86 stable runtime first so detection._yolo_worker is replaced
# before CameraPersonTrackingReID reaches the shared CameraPersonTrackingFinal base.
from .person_tracking_trt86_restore_stable import CameraPersonTrackingTRT86RestoreStable
from .person_tracking_reid import CameraPersonTrackingReID


class CameraPersonTrackingTRT86ReIDRestoreStable(
    CameraPersonTrackingReID,
    CameraPersonTrackingTRT86RestoreStable,
):
    """Stable local TRT86/NvDCF tracking plus label-only async Global ID.

    MRO intentionally preserves CameraPersonTrackingReID's side-path sampler while
    CameraPersonTrackingTRT86RestoreStable owns detector scheduling and NvDCF bbox
    motion. Global identity can style/rename an existing local track but cannot
    create, move or delete its bounding box.
    """

    pass


def main() -> int:
    return CameraPersonTrackingTRT86ReIDRestoreStable().run()


if __name__ == "__main__":
    raise SystemExit(main())
