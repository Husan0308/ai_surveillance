from __future__ import annotations

"""Backward-compatible entry point for the clean local tracking + heatmap wall.

The former adaptive/KPR/Qwen ReID experiments have been removed. Keep this module
only so existing launch commands do not break while the active runtime is exactly
CameraPersonTrackingHeatmap with camera-local NvDCF IDs.
"""

from .person_tracking_heatmap import CameraPersonTrackingHeatmap


class CameraPersonTrackingAdaptiveHeatmap(CameraPersonTrackingHeatmap):
    """Compatibility alias; no adaptive or cross-camera identity code is loaded."""


def main() -> int:
    return CameraPersonTrackingAdaptiveHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
