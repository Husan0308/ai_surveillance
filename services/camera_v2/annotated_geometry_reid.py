from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manual_geometry_reid import ManualGeometryAdaptiveTrackletReID, RoomGeometry


@dataclass
class AnnotatedCameraCalibration:
    source_id: int
    room_id: int
    matrix: np.ndarray
    reprojection_rmse_m: float
    model: str
    units: str


class AnnotatedRoomCalibration:
    """Calibration loader for the user's colored room-corner annotations.

    Four points use a projective homography. Three points use the unique affine
    transform instead of inventing a fourth landmark outside the camera FOV.
    Destination coordinates may be normalized room coordinates; they are not
    required to be physical metres.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        default = Path(__file__).resolve().parents[2] / "config" / "reid_room_calibration.json"
        self.path = Path(path or os.environ.get("CAMERA_V2_REID_CALIBRATION", str(default))).expanduser()
        self.cameras: dict[int, AnnotatedCameraCalibration] = {}
        self.rooms: dict[int, RoomGeometry] = {}
        self.errors: list[str] = []
        self._load()

    def _load(self) -> None:
        import cv2

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.errors.append(f"calibration file: {exc}")
            return

        for room_key, raw in dict(data.get("rooms", {})).items():
            try:
                room_id = int(room_key)
                units = str(raw.get("units", "normalized"))
                self.rooms[room_id] = RoomGeometry(
                    room_id=room_id,
                    match_distance_m=float(raw.get("match_distance", raw.get("match_distance_m", 0.16))),
                    veto_distance_m=float(raw.get("veto_distance", raw.get("veto_distance_m", 0.42))),
                    max_time_delta_s=float(raw.get("max_time_delta_s", 1.10)),
                    min_common_points=max(2, int(raw.get("min_common_points", 3))),
                )
            except Exception as exc:
                self.errors.append(f"room {room_key}: {exc}")

        for camera_name, raw in dict(data.get("cameras", {})).items():
            if not bool(raw.get("enabled", False)):
                continue
            try:
                source_id = int(raw["source_id"])
                room_id = int(raw["room_id"])
                units = str(data.get("rooms", {}).get(str(room_id), {}).get("units", "normalized"))
                src = np.asarray(raw.get("image_points_norm", []), dtype=np.float64)
                dst = np.asarray(raw.get("room_points_xy", raw.get("room_points_m", [])), dtype=np.float64)
                if src.ndim != 2 or dst.ndim != 2 or src.shape[1:] != (2,) or dst.shape[1:] != (2,):
                    raise ValueError("points must be Nx2")
                if len(src) != len(dst) or len(src) < 3:
                    raise ValueError("need at least 3 paired points")
                if not np.isfinite(src).all() or not np.isfinite(dst).all():
                    raise ValueError("non-finite point")
                if (src < -0.02).any() or (src > 1.02).any():
                    raise ValueError("image_points_norm outside 0..1")

                if len(src) == 3:
                    affine = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
                    matrix = np.vstack((affine.astype(np.float64), [0.0, 0.0, 1.0]))
                    model = "affine3"
                else:
                    method = cv2.RANSAC if len(src) > 4 else 0
                    matrix, _mask = cv2.findHomography(src, dst, method, 0.035 if units == "normalized" else 0.12)
                    model = "homography"
                if matrix is None or matrix.shape != (3, 3):
                    raise ValueError(f"{model} solve failed")
                if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
                    raise ValueError("singular calibration")

                projected = cv2.perspectiveTransform(
                    src.astype(np.float32).reshape(-1, 1, 2), matrix.astype(np.float64)
                ).reshape(-1, 2)
                rmse = math.sqrt(float(np.mean(np.sum((projected - dst) ** 2, axis=1))))
                max_error = float(raw.get("max_reprojection_error", 0.05 if units == "normalized" else 0.25))
                if rmse > max_error:
                    raise ValueError(f"reprojection RMSE {rmse:.4f} > {max_error:.4f}")

                self.cameras[source_id] = AnnotatedCameraCalibration(
                    source_id, room_id, matrix.astype(np.float64), rmse, model, units
                )
            except Exception as exc:
                self.errors.append(f"{camera_name}: {exc}")

    def project(self, source_id: int, x_norm: float, y_norm: float):
        calib = self.cameras.get(int(source_id))
        if calib is None:
            return None
        v = calib.matrix @ np.asarray([float(x_norm), float(y_norm), 1.0], dtype=np.float64)
        if abs(float(v[2])) < 1e-10:
            return None
        x, y = float(v[0] / v[2]), float(v[1] / v[2])
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        if calib.units == "normalized" and not (-0.35 <= x <= 1.35 and -0.35 <= y <= 1.35):
            return None
        return calib.room_id, x, y

    def room(self, room_id: int) -> RoomGeometry:
        return self.rooms.get(int(room_id), RoomGeometry(int(room_id), 0.16, 0.42, 1.10, 3))

    def snapshot(self) -> dict:
        return {
            "path": str(self.path),
            "ready_cameras": len(self.cameras),
            "camera_sources": sorted(self.cameras),
            "rmse": {str(k): v.reprojection_rmse_m for k, v in sorted(self.cameras.items())},
            "models": {str(k): v.model for k, v in sorted(self.cameras.items())},
            "units": {str(k): v.units for k, v in sorted(self.cameras.items())},
            "errors": list(self.errors),
        }


class AnnotatedGeometryAdaptiveTrackletReID(ManualGeometryAdaptiveTrackletReID):
    """Stable tracklet ReID using the user's explicit colored-corner calibration."""

    def __init__(self, manager, *, frame_width: int, frame_height: int) -> None:
        super().__init__(manager, frame_width=frame_width, frame_height=frame_height)
        self.calibration = AnnotatedRoomCalibration()
