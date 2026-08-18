from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class RoomProjection:
    room_id: str
    camera_id: str
    map_width: int
    map_height: int
    homography: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "RoomProjection":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = np.asarray(data["homography"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"invalid homography shape in {path}: {matrix.shape}")
        return cls(
            room_id=str(data["room_id"]),
            camera_id=str(data["camera_id"]),
            map_width=int(data["map_size"][0]),
            map_height=int(data["map_size"][1]),
            homography=matrix,
        )

    def project_point(self, x: float, y: float) -> tuple[float, float] | None:
        source = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, self.homography)[0, 0]
        mx, my = float(mapped[0]), float(mapped[1])
        if not np.isfinite(mx) or not np.isfinite(my):
            return None
        return mx, my

    def project_bbox_footpoint(self, bbox) -> tuple[float, float] | None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return None
        return self.project_point((x1 + x2) * 0.5, y2)

    def inside_map(self, point: tuple[float, float], margin: float = 0.0) -> bool:
        x, y = point
        return (
            -margin <= x <= self.map_width - 1 + margin
            and -margin <= y <= self.map_height - 1 + margin
        )


def solve_homography(camera_points, map_points, ransac_threshold_px: float = 4.0):
    src = np.asarray(camera_points, dtype=np.float32)
    dst = np.asarray(map_points, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("camera_points and map_points must both be Nx2")
    if len(src) < 4:
        raise ValueError("at least 4 point pairs are required")

    if len(src) == 4:
        matrix = cv2.getPerspectiveTransform(src, dst)
        mask = np.ones((4, 1), dtype=np.uint8)
    else:
        matrix, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold_px),
        )
        if matrix is None:
            raise RuntimeError("OpenCV could not solve the room homography")
    return matrix, mask


def reprojection_errors(camera_points, map_points, homography):
    src = np.asarray(camera_points, dtype=np.float32).reshape(-1, 1, 2)
    expected = np.asarray(map_points, dtype=np.float32)
    actual = cv2.perspectiveTransform(src, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)
    return np.linalg.norm(actual - expected, axis=1)
