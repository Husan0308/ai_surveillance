"""Core-v1 surveillance runtime bootstrap.

The detector hot path stays untouched. ReID and presentation-only wrappers are
installed through package-level aliases before app.py imports their classes.
"""

from . import reid_service as _reid_service
from . import heatmap_publisher as _heatmap_publisher
from .reid_hardening import HardenedReIDCoordinator
from .heatmap_publisher_v3 import HeatmapJpegPublisher as StableHeatmapJpegPublisher

_reid_service.ReIDCoordinator = HardenedReIDCoordinator
_heatmap_publisher.HeatmapJpegPublisher = StableHeatmapJpegPublisher

__all__ = ["HardenedReIDCoordinator", "StableHeatmapJpegPublisher"]
