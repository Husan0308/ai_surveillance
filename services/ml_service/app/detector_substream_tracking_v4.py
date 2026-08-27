from __future__ import annotations

import os

from services.ml_service.app.detector_substream_tracking_v3 import DetectorSubstreamTrackingV3Service
from services.ml_service.app.local_tracker_sparse_v4 import MultiCameraBoxStableTracker
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W


class DetectorSubstreamTrackingV4Service(DetectorSubstreamTrackingV3Service):
    """Frozen V14 detector + V3 association + V4 published-box stability."""

    def __init__(self) -> None:
        super().__init__()

        self.track_nested_duplicate_ios = float(
            os.environ.get("ML_TRACK_NESTED_DUPLICATE_IOS", "0.82")
        )
        self.track_nested_duplicate_app_floor = float(
            os.environ.get("ML_TRACK_NESTED_DUPLICATE_APP_FLOOR", "0.58")
        )
        self.track_nested_duplicate_center_frac = float(
            os.environ.get("ML_TRACK_NESTED_DUPLICATE_CENTER_FRAC", "0.28")
        )
        self.track_render_anchor_alpha = float(
            os.environ.get("ML_TRACK_RENDER_ANCHOR_ALPHA", "0.72")
        )
        self.track_render_size_alpha = float(
            os.environ.get("ML_TRACK_RENDER_SIZE_ALPHA", "0.20")
        )
        self.track_render_recovery_size_alpha = float(
            os.environ.get("ML_TRACK_RENDER_RECOVERY_SIZE_ALPHA", "0.34")
        )
        self.track_render_max_size_step = float(
            os.environ.get("ML_TRACK_RENDER_MAX_SIZE_STEP", "0.28")
        )
        self.track_render_velocity_gain = float(
            os.environ.get("ML_TRACK_RENDER_VELOCITY_GAIN", "0.30")
        )

        appearance_weight = min(
            0.30, max(0.0, float(os.environ.get("ML_TRACK_APPEARANCE_WEIGHT", "0.22")))
        )
        confirm_hits = max(1, int(os.environ.get("ML_TRACK_CONFIRM_HITS", "2")))
        tentative_ttl_sec = max(
            0.3, float(os.environ.get("ML_TRACK_TENTATIVE_TTL_SEC", "0.9"))
        )

        self.trackers = MultiCameraBoxStableTracker(
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
            nested_duplicate_ios=self.track_nested_duplicate_ios,
            nested_duplicate_app_floor=self.track_nested_duplicate_app_floor,
            nested_duplicate_center_frac=self.track_nested_duplicate_center_frac,
            render_anchor_alpha=self.track_render_anchor_alpha,
            render_size_alpha=self.track_render_size_alpha,
            render_recovery_size_alpha=self.track_render_recovery_size_alpha,
            render_max_size_step=self.track_render_max_size_step,
            render_velocity_gain=self.track_render_velocity_gain,
        )

    def run(self) -> int:
        print(
            "ML_STEP4_V4_PROFILE "
            "algorithm=v3-observation-recovery+box-stability cpu_only=1 "
            f"nested_ios={self.track_nested_duplicate_ios:.2f} "
            f"nested_app_floor={self.track_nested_duplicate_app_floor:.2f} "
            f"render_anchor_alpha={self.track_render_anchor_alpha:.2f} "
            f"render_size_alpha={self.track_render_size_alpha:.2f} "
            f"render_max_size_step={self.track_render_max_size_step:.2f} "
            "association_box=raw render_box=smoothed bottom_anchor=1 pose=0 gpu_tracker=0",
            flush=True,
        )
        return super().run()


def main() -> int:
    service = DetectorSubstreamTrackingV4Service()
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
