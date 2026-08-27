from __future__ import annotations

import os

from services.ml_service.app.detector_substream_tracking_v4 import DetectorSubstreamTrackingV4Service


class DetectorSubstreamTrackingV5Service(DetectorSubstreamTrackingV4Service):
    """Step 4 v5: V4 box stability with a real ByteTrack-style low-score lane.

    V4 launched the TensorRT detector at 0.18 confidence and then configured the
    tracker low threshold to the same value. That made detections below 0.18
    impossible to reach the second-stage recovery logic. V5 restores the proven
    high-recall detector floor while keeping new-track creation conservative.
    """

    def __init__(self) -> None:
        # These are defaults, not hard locks: explicit deployment overrides still win.
        # Keep weak person candidates for association, but never let them create a new
        # identity unless they pass the higher new-track threshold.
        os.environ.setdefault("ML_DETECTOR_CONF", "0.08")
        os.environ.setdefault("ML_TRACK_LOW_THRESH", "0.08")
        os.environ.setdefault("ML_TRACK_HIGH_THRESH", "0.25")
        os.environ.setdefault("ML_TRACK_NEW_THRESH", "0.30")
        os.environ.setdefault("ML_TRACK_LOW_RECOVERY_THRESH", "0.10")
        super().__init__()

        if self.conf > self.track_high_thresh:
            raise RuntimeError(
                "V5 invalid thresholds: detector confidence must not exceed "
                f"track_high_thresh ({self.conf:.3f}>{self.track_high_thresh:.3f})"
            )
        if self.track_low_thresh > self.track_high_thresh:
            raise RuntimeError(
                "V5 invalid thresholds: track_low_thresh must be <= track_high_thresh"
            )

    def run(self) -> int:
        print(
            "ML_STEP4_V5_PROFILE "
            "algorithm=v4-box-stability+low-score-recovery "
            f"detector_conf={self.conf:.2f} low={self.track_low_thresh:.2f} "
            f"high={self.track_high_thresh:.2f} new={self.track_new_thresh:.2f} "
            "weak_detection_can_recover=1 weak_detection_can_create=0 "
            "pose=0 gpu_tracker=0 nvdcf=0",
            flush=True,
        )
        return super().run()


def main() -> int:
    service = DetectorSubstreamTrackingV5Service()
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
