"""Scale-only raw-frame adapter for an isolated local-tracking diagnostic.

This is deliberately not a rectification or calibration transform. It preserves
the complete 16:9 raw camera view and only rescales it to the fixed DeepStream
appsrc/mux input geometry. UI boxes can be mapped back to native source pixels
with :func:`detector_box_to_source`.
"""
from __future__ import annotations

import cv2
import numpy as np


DETECTOR_WIDTH = 1920
DETECTOR_HEIGHT = 1080
SOURCE_SIZE = {
    "CAM-02": (3200, 1800),
    "CAM-05": (2560, 1440),
}


class NativeFullFovScale:
    """Validate a known native source geometry and resize without crop/warp."""

    def __init__(self, camera_id: str):
        if camera_id not in SOURCE_SIZE:
            raise ValueError(f"unsupported camera for Entrance test: {camera_id}")
        self.camera_id = camera_id
        self.width, self.height = SOURCE_SIZE[camera_id]

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape != (self.height, self.width, 3)):
            shape = getattr(frame, "shape", None)
            dtype = getattr(frame, "dtype", None)
            raise ValueError(
                f"{self.camera_id}: expected native uint8 BGR {self.width}x{self.height}; "
                f"got {shape}/{dtype}"
            )
        # Both known sources and the target are 16:9. INTER_AREA performs a
        # scale-only full-frame resample; there is no crop, padding, undistort,
        # projective warp, or change to the source-frame timestamp.
        resized = cv2.resize(
            frame, (DETECTOR_WIDTH, DETECTOR_HEIGHT), interpolation=cv2.INTER_AREA
        )
        return np.ascontiguousarray(resized)


def detector_box_to_source(obj: dict, camera_id: str) -> dict:
    """Return a copy of a 1920x1080 detector box in native source pixels."""
    try:
        width, height = SOURCE_SIZE[camera_id]
    except KeyError as exc:
        raise ValueError(f"unsupported camera for Entrance test: {camera_id}") from exc
    scale_x = width / DETECTOR_WIDTH
    scale_y = height / DETECTOR_HEIGHT
    mapped = dict(obj)
    for field, scale in (("left", scale_x), ("width", scale_x),
                         ("top", scale_y), ("height", scale_y)):
        if field in mapped:
            mapped[field] = float(mapped[field]) * scale
    return mapped
