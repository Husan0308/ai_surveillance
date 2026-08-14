from __future__ import annotations

from .jpeg_publisher import LatestJpegPublisher


class HeatmapJpegPublisher(LatestJpegPublisher):
    """JPEG publisher that blends camera-space heat before drawing boxes/labels."""

    def __init__(self, *args, heatmap_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmap_provider = heatmap_provider

    def _draw_detection(
        self,
        image,
        source_width,
        source_height,
        now,
        display_frame_id,
        display_frame_time,
    ):
        provider = self.heatmap_provider
        if provider is not None:
            try:
                image = provider.overlay(self.camera_id, image)
            except Exception:
                # Presentation must never break frame publication or detections.
                pass
        return super()._draw_detection(
            image,
            source_width,
            source_height,
            now,
            display_frame_id,
            display_frame_time,
        )
