from __future__ import annotations

import os

from services.ml_service.app.detector_substream_tracking import DetectorSubstreamTrackingService
from services.ml_service.app.local_tracker_sparse_v2 import MultiCameraSparseRecoveryTracker
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W


class DetectorSubstreamTrackingV2Service(DetectorSubstreamTrackingService):
    """Frozen V14 detector + Step 4 v2 sparse-recovery CPU tracker."""

    def __init__(self) -> None:
        super().__init__()
        self.track_reacquire_thresh = float(os.environ.get("ML_TRACK_REACQUIRE_THRESH", "0.12"))
        self.track_low_recovery_thresh = float(
            os.environ.get("ML_TRACK_LOW_RECOVERY_THRESH", "0.10")
        )
        self.track_low_recovery_sec = float(os.environ.get("ML_TRACK_LOW_RECOVERY_SEC", "1.6"))
        self.track_duplicate_iou = float(os.environ.get("ML_TRACK_DUPLICATE_IOU", "0.60"))
        confirm_hits = max(1, int(os.environ.get("ML_TRACK_CONFIRM_HITS", "2")))
        tentative_ttl_sec = max(
            0.3, float(os.environ.get("ML_TRACK_TENTATIVE_TTL_SEC", "0.9"))
        )
        appearance_weight = min(
            0.30, max(0.0, float(os.environ.get("ML_TRACK_APPEARANCE_WEIGHT", "0.22")))
        )

        self.trackers = MultiCameraSparseRecoveryTracker(
            (camera.camera_id for camera in self.cameras),
            INPUT_W,
            CONTENT_H,
            low_thresh=self.track_low_thresh,
            high_thresh=self.track_high_thresh,
            new_track_thresh=self.track_new_thresh,
            confirm_hits=confirm_hits,
            tentative_ttl_sec=tentative_ttl_sec,
            shadow_sec=self.track_shadow_sec,
            max_lost_sec=self.track_max_lost_sec,
            appearance_weight=appearance_weight,
            reacquire_thresh=self.track_reacquire_thresh,
            low_recovery_thresh=self.track_low_recovery_thresh,
            low_recovery_sec=self.track_low_recovery_sec,
            duplicate_iou=self.track_duplicate_iou,
        )

    def run(self) -> int:
        print(
            "ML_STEP4_V2_PROFILE "
            "algorithm=sparse-recovery-bytetrack cpu_only=1 "
            f"reacquire_thresh={self.track_reacquire_thresh:.2f} "
            f"low_recovery_thresh={self.track_low_recovery_thresh:.2f} "
            f"low_recovery_sec={self.track_low_recovery_sec:.1f}s "
            f"duplicate_iou={self.track_duplicate_iou:.2f} "
            "low_can_create_id=0 gpu_tracker=0 nvdcf=0 reid=0 global_id=0",
            flush=True,
        )
        return super().run()


def main() -> int:
    service = DetectorSubstreamTrackingV2Service()
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
