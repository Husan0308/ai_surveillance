from __future__ import annotations

from .heatmap_publisher_v2 import HeatmapJpegPublisher as _PersistentPosePublisher


class HeatmapJpegPublisher(_PersistentPosePublisher):
    """Operator-console defaults: Pose is visible but analytics remain independent."""

    def __init__(self, *args, **kwargs):
        # The old config had overlay_default=false and a fairly strict 0.30
        # display threshold. Keep inference unchanged, but make useful keypoints
        # visible immediately in the restored dashboard.
        kwargs["pose_visible"] = True
        kwargs["pose_overlay_conf"] = min(
            0.22, float(kwargs.get("pose_overlay_conf", 0.30))
        )
        kwargs["pose_overlay_max_age_ms"] = max(
            12000.0, float(kwargs.get("pose_overlay_max_age_ms", 1600.0))
        )
        super().__init__(*args, **kwargs)
