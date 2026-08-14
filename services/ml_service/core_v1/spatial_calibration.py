from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np
import yaml


def _finite_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _inside_polygon(point: tuple[float, float], polygon: list) -> bool:
    points = [_finite_point(item) for item in polygon]
    points = [item for item in points if item is not None]
    if len(points) < 3:
        return False
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, point, False) >= 0


class RoomSpatialMapper:
    """Persistent camera-floor calibration and lightweight spatial scoring.

    Homographies are only used after a validated assisted/automatic calibration.
    Uncalibrated cameras fail closed and leave the existing appearance ReID path
    unchanged.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data.get("rooms"), dict):
                raise ValueError("room mapping requires a rooms mapping")
            if not isinstance(data.get("calibrations"), dict):
                raise ValueError("room mapping requires calibrations")
            self._data = data

    def _save_locked(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self._data, handle, sort_keys=False, allow_unicode=True)
        temporary.replace(self.path)

    @property
    def fusion_config(self) -> dict:
        with self._lock:
            return dict(self._data.get("fusion") or {})

    def room_for_camera(self, camera_id: str) -> str | None:
        camera_id = str(camera_id)
        with self._lock:
            calibration = self._data.get("calibrations", {}).get(camera_id) or {}
            room_id = calibration.get("room_id")
            if room_id:
                return str(room_id)
            for candidate, room in self._data.get("rooms", {}).items():
                if camera_id in (room.get("cameras") or []):
                    return str(candidate)
        return None

    def camera_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        with self._lock:
            for room in self._data.get("rooms", {}).values():
                cameras = [str(item) for item in room.get("cameras") or []]
                if len(cameras) == 2:
                    pairs.append((cameras[0], cameras[1]))
        return pairs

    def _valid_homography_locked(self, camera_id: str):
        calibration = self._data.get("calibrations", {}).get(str(camera_id)) or {}
        if calibration.get("status") not in {"good", "calibrated", "automatic"}:
            return None, calibration
        try:
            matrix = np.asarray(calibration.get("homography"), dtype=np.float64).reshape(3, 3)
        except (TypeError, ValueError):
            return None, calibration
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
            return None, calibration
        return matrix, calibration

    def project_point(self, camera_id: str, point: tuple[float, float], source_size: tuple[float, float] | None = None) -> dict | None:
        source = _finite_point(point)
        if source is None:
            return None
        with self._lock:
            matrix, calibration = self._valid_homography_locked(camera_id)
            if matrix is None:
                return None
            calibration_size = calibration.get("image_size")
            if source_size and calibration_size and len(calibration_size) == 2:
                source_width=max(1.0,float(source_size[0]));source_height=max(1.0,float(source_size[1]))
                source=(source[0]*float(calibration_size[0])/source_width,source[1]*float(calibration_size[1])/source_height)
            projected = cv2.perspectiveTransform(
                np.asarray([[source]], dtype=np.float64), matrix
            )[0, 0]
            x, y = float(projected[0]), float(projected[1])
            margin = float(
                (self._data.get("calibration") or {}).get("max_outside_margin", 0.08)
            )
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            if x < -margin or y < -margin or x > 1.0 + margin or y > 1.0 + margin:
                return None
            room_id = str(calibration.get("room_id") or self.room_for_camera(camera_id) or "")
            overlap = (self._data.get("rooms", {}).get(room_id) or {}).get(
                "overlap_polygon"
            ) or []
            return {
                "room_id": room_id,
                "x": max(0.0, min(1.0, x)),
                "y": max(0.0, min(1.0, y)),
                "inside_overlap": _inside_polygon((x, y), overlap),
                "calibration_confidence": float(calibration.get("confidence") or 0.0),
                "calibration_method": str(calibration.get("method") or "unknown"),
            }

    def project_box_footpoint(self, camera_id: str, box: object, source_size: tuple[float, float] | None = None) -> dict | None:
        try:
            footpoint = (
                (float(box.x1) + float(box.x2)) * 0.5,
                float(box.y2),
            )
        except (AttributeError, TypeError, ValueError):
            return None
        return self.project_point(camera_id, footpoint, source_size=source_size)

    def calibrate(
        self,
        camera_id: str,
        image_points: list,
        room_points: list,
        image_size: list | tuple | None = None,
        method: str = "assisted",
    ) -> dict:
        camera_id = str(camera_id)
        source = [_finite_point(item) for item in image_points]
        target = [_finite_point(item) for item in room_points]
        if any(item is None for item in source + target) or len(source) != len(target):
            raise ValueError("image_points and room_points must be finite matching pairs")
        config = self._data.get("calibration") or {}
        minimum = max(4, int(config.get("minimum_points", 6)))
        if len(source) < minimum:
            raise ValueError(f"at least {minimum} floor landmark pairs are required")
        target_array = np.asarray(target, dtype=np.float64)
        if np.any(target_array < 0.0) or np.any(target_array > 1.0):
            raise ValueError("room points must use normalized coordinates 0..1")
        source_array = np.asarray(source, dtype=np.float64)
        threshold = max(1e-5, float(config.get("ransac_threshold_normalized", 0.025)))
        matrix, mask = cv2.findHomography(source_array, target_array, cv2.RANSAC, threshold)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            raise ValueError("homography could not be estimated from these points")
        projected = cv2.perspectiveTransform(source_array.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - target_array, axis=1)
        inliers = mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < minimum:
            raise ValueError(f"only {inlier_count}/{len(source)} landmarks agree")
        mean_error = float(errors[inliers].mean())
        good_error = max(1e-5, float(config.get("good_error_normalized", 0.035)))
        point_quality = min(1.0, inlier_count / max(8.0, float(len(source))))
        error_quality = max(0.0, 1.0 - mean_error / (good_error * 2.0))
        confidence = point_quality * error_quality
        status = "good" if mean_error <= good_error and confidence >= 0.55 else "uncertain"
        if status != "good":
            matrix_to_store = None
        else:
            matrix_to_store = [[float(v) for v in row] for row in matrix]
        with self._lock:
            existing = self._data.get("calibrations", {}).get(camera_id)
            if existing is None:
                raise ValueError(f"unknown camera: {camera_id}")
            existing.update(
                {
                    "status": status,
                    "method": str(method),
                    "confidence": round(confidence, 6),
                    "homography": matrix_to_store,
                    "image_size": list(image_size) if image_size else None,
                    "control_points": [
                        {"image": list(src), "room": list(dst)}
                        for src, dst in zip(source, target)
                    ],
                    "reprojection_error_normalized": round(mean_error, 8),
                    "inliers": inlier_count,
                    "point_count": len(source),
                }
            )
            self._save_locked()
            return deepcopy(existing)

    def clear_calibration(self, camera_id: str) -> dict:
        with self._lock:
            calibration = self._data.get("calibrations", {}).get(str(camera_id))
            if calibration is None:
                raise ValueError(f"unknown camera: {camera_id}")
            room_id = calibration.get("room_id")
            calibration.clear()
            calibration.update(
                {
                    "room_id": room_id,
                    "status": "uncalibrated",
                    "method": "none",
                    "confidence": 0.0,
                    "homography": None,
                    "image_size": None,
                    "control_points": [],
                    "camera_position": None,
                    "fov_polygon": [],
                }
            )
            self._save_locked()
            return deepcopy(calibration)

    def snapshot(self) -> dict:
        with self._lock:
            result = deepcopy(self._data)
        usable = {
            camera_id
            for camera_id, item in result.get("calibrations", {}).items()
            if item.get("status") in {"good", "calibrated", "automatic"}
            and item.get("homography")
        }
        active_rooms = [
            room_id
            for room_id, room in result.get("rooms", {}).items()
            if len(room.get("cameras") or []) == 2
            and all(camera_id in usable for camera_id in room.get("cameras") or [])
        ]
        result["summary"] = {
            "calibrated_cameras": len(usable),
            "total_cameras": len(result.get("calibrations", {})),
            "active_rooms": active_rooms,
            "spatial_fusion_active": bool(active_rooms),
        }
        return result

    @staticmethod
    def automatic_pair_evidence(left_image, right_image) -> dict:
        """On-demand, CPU-only visual relation check; never fabricates a floor map."""
        if left_image is None or right_image is None:
            return {"status": "insufficient", "confidence": 0.0, "reason": "frame_missing"}
        left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        left = cv2.resize(left, (640, 360), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (640, 360), interpolation=cv2.INTER_AREA)
        orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
        key_left, desc_left = orb.detectAndCompute(left, None)
        key_right, desc_right = orb.detectAndCompute(right, None)
        if desc_left is None or desc_right is None or len(key_left) < 20 or len(key_right) < 20:
            return {"status": "insufficient", "confidence": 0.0, "reason": "too_few_features"}
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_left, desc_right, k=2)
        good = [a for a, b in matches if a.distance < 0.72 * b.distance]
        if len(good) < 12:
            return {"status": "insufficient", "confidence": min(0.25, len(good) / 48.0), "matches": len(good), "reason": "too_few_common_landmarks"}
        src = np.float32([key_left[item.queryIdx].pt for item in good])
        dst = np.float32([key_right[item.trainIdx].pt for item in good])
        _matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        ratio = inliers / max(1, len(good))
        confidence = min(0.95, ratio * min(1.0, inliers / 40.0))
        return {
            "status": "relation_detected" if confidence >= 0.55 else "insufficient",
            "confidence": round(confidence, 4),
            "matches": len(good),
            "inliers": inliers,
            "inlier_ratio": round(ratio, 4),
            "floor_calibration_ready": False,
            "requires_assisted_floor_landmarks": True,
            "reason": "image_relation_is_not_a_room_floor_homography",
        }
