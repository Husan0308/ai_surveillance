from __future__ import annotations

import os

from services.ml_service.app.detector_substream_tracking_v2 import DetectorSubstreamTrackingV2Service
from services.ml_service.app.local_tracker_sparse_v3 import MultiCameraObservationRecoveryTracker
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W


class DetectorSubstreamTrackingV3Service(DetectorSubstreamTrackingV2Service):
    """Frozen V14 detector + Step 4 v3 observation-centric CPU local tracker."""

    def __init__(self) -> None:
        # With a ~2 Hz detector, 2.5 s retains only about five measurement opportunities.
        # Keep non-rendered identities longer while the visible shadow remains short.
        os.environ.setdefault("ML_TRACK_MAX_LOST_SEC", "5.0")
        os.environ.setdefault("ML_TRACK_LOW_RECOVERY_SEC", "3.0")
        os.environ.setdefault("ML_TRACK_SHADOW_SEC", "1.1")
        super().__init__()

        self.track_low_appearance_weight = float(
            os.environ.get("ML_TRACK_LOW_APPEARANCE_WEIGHT", "0.16")
        )
        self.track_low_appearance_floor = float(
            os.environ.get("ML_TRACK_LOW_APPEARANCE_FLOOR", "0.45")
        )
        self.track_live_duplicate_iou = float(
            os.environ.get("ML_TRACK_LIVE_DUPLICATE_IOU", "0.72")
        )
        self.track_lost_velocity_half_life = float(
            os.environ.get("ML_TRACK_LOST_VELOCITY_HALF_LIFE_SEC", "0.9")
        )
        appearance_weight = min(
            0.30, max(0.0, float(os.environ.get("ML_TRACK_APPEARANCE_WEIGHT", "0.22")))
        )
        confirm_hits = max(1, int(os.environ.get("ML_TRACK_CONFIRM_HITS", "2")))
        tentative_ttl_sec = max(
            0.3, float(os.environ.get("ML_TRACK_TENTATIVE_TTL_SEC", "0.9"))
        )

        self.trackers = MultiCameraObservationRecoveryTracker(
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
            low_appearance_weight=self.track_low_appearance_weight,
            low_appearance_floor=self.track_low_appearance_floor,
            live_duplicate_iou=self.track_live_duplicate_iou,
            lost_velocity_half_life_sec=self.track_lost_velocity_half_life,
        )

    def run(self) -> int:
        print(
            "ML_STEP4_V3_PROFILE "
            "algorithm=observation-recovery-sparse cpu_only=1 "
            f"max_lost={self.track_max_lost_sec:.1f}s "
            f"low_recovery={self.track_low_recovery_sec:.1f}s "
            f"low_app_weight={self.track_low_appearance_weight:.2f} "
            f"low_app_floor={self.track_low_appearance_floor:.2f} "
            f"live_duplicate_iou={self.track_live_duplicate_iou:.2f} "
            f"lost_velocity_half_life={self.track_lost_velocity_half_life:.1f}s "
            "last_observation_anchor=1 low_hijack_guard=1 gpu_tracker=0 nvdcf=0 reid=0 global_id=0",
            flush=True,
        )
        return super().run()


def main() -> int:
    service = DetectorSubstreamTrackingV3Service()
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
