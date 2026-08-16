from __future__ import annotations

from .mmap_frame_reader import SmoothMmapFrameReader


class SmoothFrameReader(SmoothMmapFrameReader):
    """Compatibility name for the Sentinel UI's local latest-only camera reader.

    The Sentinel UI was originally written against SmoothFrameReader. Keep that
    public name stable, but route it to SmoothMmapFrameReader so six camera tiles
    receive the backend's annotated BGR frames from mmap without JPEG encode,
    HTTP transport, or JPEG decode in the presentation hot path.
    """

    def __init__(self, camera_id: str):
        super().__init__(camera_id)
        # Older UI diagnostics read this historical attribute. mmap never JPEG
        # decodes, so the compatible value remains zero.
        self.decode_failures = 0
