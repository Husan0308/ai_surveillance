from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adaptive_reid import AdaptiveTrackletReID
from .stable_adaptive_reid import StableAdaptiveTrackletReID

LocalKey = tuple[int, int]


@dataclass
class CameraCalibration:
    source_id: int
    room_id: int
    matrix: np.ndarray
    reprojection_rmse_m: float


@dataclass
class RoomGeometry:
    room_id: int
    match_distance_m: float
    veto_distance_m: float
    max_time_delta_s: float
    min_common_points: int


class ManualRoomHomography:
    """Explicit user calibration from normalized camera pixels to room metres.

    Nothing is inferred or self-learned. Each enabled camera must contain at least
    four user-supplied point correspondences in config/reid_room_calibration.json.
    Camera coordinates are normalized to [0, 1], so calibration remains valid when
    the runtime display/detector resolution changes.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        default = Path(__file__).resolve().parents[2] / "config" / "reid_room_calibration.json"
        self.path = Path(
            path or os.environ.get("CAMERA_V2_REID_CALIBRATION", str(default))
        ).expanduser()
        self.cameras: dict[int, CameraCalibration] = {}
        self.rooms: dict[int, RoomGeometry] = {}
        self.errors: list[str] = []
        self._load()

    def _load(self) -> None:
        try:
            import cv2
        except Exception as exc:
            self.errors.append(f"opencv unavailable: {exc}")
            return
        if not self.path.exists():
            self.errors.append(f"calibration file missing: {self.path}")
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.errors.append(f"invalid calibration json: {exc}")
            return

        for room_key, raw in dict(data.get("rooms", {})).items():
            try:
                room_id = int(room_key)
                self.rooms[room_id] = RoomGeometry(
                    room_id=room_id,
                    match_distance_m=max(0.05, float(raw.get("match_distance_m", 0.70))),
                    veto_distance_m=max(0.10, float(raw.get("veto_distance_m", 1.60))),
                    max_time_delta_s=max(0.10, float(raw.get("max_time_delta_s", 0.90))),
                    min_common_points=max(2, int(raw.get("min_common_points", 3))),
                )
            except Exception as exc:
                self.errors.append(f"room {room_key}: {exc}")

        max_rmse = max(
            0.02, float(os.environ.get("CAMERA_V2_CALIB_MAX_RMSE_M", "0.25"))
        )
        for camera_name, raw in dict(data.get("cameras", {})).items():
            if not bool(raw.get("enabled", False)):
                continue
            try:
                source_id = int(raw["source_id"])
                room_id = int(raw["room_id"])
                src = np.asarray(raw.get("image_points_norm", []), dtype=np.float64)
                dst = np.asarray(raw.get("room_points_m", []), dtype=np.float64)
                if src.ndim != 2 or dst.ndim != 2 or src.shape[1:] != (2,) or dst.shape[1:] != (2,):
                    raise ValueError("points must be Nx2 arrays")
                if len(src) != len(dst) or len(src) < 4:
                    raise ValueError("need >=4 matching image_points_norm and room_points_m")
                if not np.isfinite(src).all() or not np.isfinite(dst).all():
                    raise ValueError("points contain NaN/inf")
                if (src < -0.02).any() or (src > 1.02).any():
                    raise ValueError("image_points_norm must be normalized 0..1")

                # With >4 points RANSAC rejects a mistaken click. Four points use
                # the exact planar transform. Both produce a 3x3 image->room map.
                method = cv2.RANSAC if len(src) > 4 else 0
                matrix, mask = cv2.findHomography(src, dst, method, 0.12)
                if matrix is None or matrix.shape != (3, 3):
                    raise ValueError("findHomography failed")
                if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
                    raise ValueError("homography is singular/invalid")

                projected = cv2.perspectiveTransform(
                    src.astype(np.float32).reshape(-1, 1, 2),
                    matrix,
                ).reshape(-1, 2)
                if mask is not None and len(mask) == len(src):
                    inliers = mask.reshape(-1).astype(bool)
                    if int(inliers.sum()) >= 4:
                        projected_eval = projected[inliers]
                        dst_eval = dst[inliers]
                    else:
                        projected_eval = projected
                        dst_eval = dst
                else:
                    projected_eval = projected
                    dst_eval = dst
                rmse = math.sqrt(
                    float(np.mean(np.sum((projected_eval - dst_eval) ** 2, axis=1)))
                )
                if rmse > max_rmse:
                    raise ValueError(
                        f"reprojection RMSE {rmse:.3f}m > allowed {max_rmse:.3f}m"
                    )
                if room_id not in self.rooms:
                    self.rooms[room_id] = RoomGeometry(room_id, 0.70, 1.60, 0.90, 3)
                self.cameras[source_id] = CameraCalibration(
                    source_id=source_id,
                    room_id=room_id,
                    matrix=np.asarray(matrix, dtype=np.float64),
                    reprojection_rmse_m=rmse,
                )
            except Exception as exc:
                self.errors.append(f"{camera_name}: {exc}")

    def project(self, source_id: int, x_norm: float, y_norm: float) -> tuple[int, float, float] | None:
        calib = self.cameras.get(int(source_id))
        if calib is None:
            return None
        v = calib.matrix @ np.asarray([float(x_norm), float(y_norm), 1.0], dtype=np.float64)
        if abs(float(v[2])) <= 1e-10:
            return None
        x = float(v[0] / v[2])
        y = float(v[1] / v[2])
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return calib.room_id, x, y

    def room(self, room_id: int) -> RoomGeometry:
        return self.rooms.get(int(room_id), RoomGeometry(int(room_id), 0.70, 1.60, 0.90, 3))

    def snapshot(self) -> dict:
        return {
            "path": str(self.path),
            "ready_cameras": len(self.cameras),
            "camera_sources": sorted(self.cameras),
            "rmse": {
                str(source): calibration.reprojection_rmse_m
                for source, calibration in sorted(self.cameras.items())
            },
            "errors": list(self.errors),
        }


class ManualGeometryAdaptiveTrackletReID(StableAdaptiveTrackletReID):
    """Stable ReID fused with user-calibrated same-time room ground positions.

    Appearance still supplies identity. Geometry is a second independent signal:
      * close same-time world tracks strengthen an ambiguous appearance match;
      * clearly distant world tracks are vetoed;
      * confirmed peer leases are not released while calibrated geometry agrees.

    This mirrors the practical multi-view principle of comparing ground/foot
    locations at common timestamps, while keeping the existing lightweight 2D
    detector + NvDCF + external ReID stack on this Pascal machine.
    """

    def __init__(self, manager, *, frame_width: int, frame_height: int) -> None:
        self.frame_width = max(1.0, float(frame_width))
        self.frame_height = max(1.0, float(frame_height))
        self.calibration = ManualRoomHomography()
        self.world_history: dict[LocalKey, deque[tuple[float, float, float]]] = {}
        self.world_window = max(6, min(32, int(os.environ.get("CAMERA_V2_WORLD_WINDOW", "16"))))
        self.world_ttl = max(2.0, float(os.environ.get("CAMERA_V2_WORLD_TTL", "5.0")))
        self.geometry_weight = min(0.60, max(0.10, float(os.environ.get("CAMERA_V2_GEOMETRY_WEIGHT", "0.38"))))
        self.geometry_min_reid = float(os.environ.get("CAMERA_V2_GEOMETRY_MIN_REID", "0.24"))
        self.foot_lift = min(0.12, max(0.0, float(os.environ.get("CAMERA_V2_WORLD_FOOT_LIFT", "0.04"))))
        self.geometry_matches = 0
        self.geometry_vetoes = 0
        self.geometry_missing = 0
        self.geometry_pairs = 0
        self.last_world_rmse = -1.0
        self.last_world_score = -1.0
        self.last_world_common = 0
        super().__init__(manager)

    @staticmethod
    def _bbox(row: dict) -> tuple[float, float, float, float] | None:
        raw = row.get("bbox")
        if raw is None or len(raw) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(raw[i]) for i in range(4)]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def observe_rows(self, rows: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        for row in rows:
            source_id = int(row.get("source_id", -1))
            object_id = int(row.get("object_id", -1))
            bbox = self._bbox(row)
            if source_id < 0 or object_id < 0 or bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            h = max(1.0, y2 - y1)
            x_norm = ((x1 + x2) * 0.5) / self.frame_width
            y_norm = (y2 - self.foot_lift * h) / self.frame_height
            mapped = self.calibration.project(source_id, x_norm, y_norm)
            if mapped is None:
                continue
            _room_id, world_x, world_y = mapped
            key = (source_id, object_id)
            captured = float(row.get("captured_at", now) or now)
            history = self.world_history.get(key)
            if history is None:
                history = deque(maxlen=self.world_window)
                self.world_history[key] = history
            history.append((world_x, world_y, captured))

        super().observe_rows(rows, now)
        stale = [
            key
            for key, history in self.world_history.items()
            if not history or now - history[-1][2] > max(self.world_ttl * 2.0, self.bank_ttl * 1.5)
        ]
        for key in stale:
            self.world_history.pop(key, None)

    def _world_measure(self, a: LocalKey, b: LocalKey, now: float) -> tuple[float, float, int] | None:
        calib_a = self.calibration.cameras.get(a[0])
        calib_b = self.calibration.cameras.get(b[0])
        if calib_a is None or calib_b is None or calib_a.room_id != calib_b.room_id:
            return None
        room = self.calibration.room(calib_a.room_id)
        ah = [p for p in self.world_history.get(a, ()) if now - p[2] <= self.world_ttl]
        bh = [p for p in self.world_history.get(b, ()) if now - p[2] <= self.world_ttl]
        if not ah or not bh:
            return None

        used_b: set[int] = set()
        distances: list[float] = []
        for ax, ay, at in ah:
            best_j = None
            best_dt = 1e9
            for j, (bx, by, bt) in enumerate(bh):
                if j in used_b:
                    continue
                dt = abs(at - bt)
                if dt <= room.max_time_delta_s and dt < best_dt:
                    best_dt = dt
                    best_j = j
            if best_j is None:
                continue
            bx, by, _bt = bh[best_j]
            used_b.add(best_j)
            distances.append(math.hypot(ax - bx, ay - by))

        if len(distances) < room.min_common_points:
            return None
        # Robust RMS: if there are enough points, trim the single worst jitter/outlier.
        distances.sort()
        eval_dist = distances[:-1] if len(distances) >= 6 else distances
        rmse = math.sqrt(sum(d * d for d in eval_dist) / len(eval_dist))
        score = 1.0 / (1.0 + rmse)
        return rmse, score, len(eval_dist)

    def tracklet_similarity(self, a, b, now: float) -> float:
        appearance = super().tracklet_similarity(a, b, now)
        if appearance < -0.5:
            return appearance
        measured = self._world_measure(a, b, now)
        if measured is None:
            self.geometry_missing += 1
            return appearance

        rmse, geometry_score, common = measured
        self.geometry_pairs += 1
        self.last_world_rmse = rmse
        self.last_world_score = geometry_score
        self.last_world_common = common
        room_id = self.calibration.cameras[a[0]].room_id
        room = self.calibration.room(room_id)

        if rmse >= room.veto_distance_m:
            self.geometry_vetoes += 1
            return -1.0

        if appearance < self.geometry_min_reid:
            # Geometry alone must not identify two people with almost unrelated
            # appearance; it is a strong constraint, not a replacement ReID model.
            return appearance

        if rmse <= room.match_distance_m:
            self.geometry_matches += 1
            weight = self.geometry_weight
            combined = (1.0 - weight) * appearance + weight * geometry_score
            return max(-1.0, min(1.0, combined))

        # Between match and veto radii geometry is deliberately neutral.
        return appearance

    def _audit_peer_locks(self, now: float) -> None:
        """Release a confirmed peer only on repeated fresh contradiction.

        If calibrated positions still agree, weak appearance does not release the
        ID lease. If calibrated positions are clearly impossible, they count as a
        stronger contradiction than appearance noise.
        """
        seen: set[frozenset[LocalKey]] = set()
        for a, b in list(self.peer_owner.items()):
            pair = frozenset((a, b))
            if pair in seen:
                continue
            seen.add(pair)
            if self.peer_owner.get(b) != a:
                self.peer_owner.pop(a, None)
                continue
            if not self.manager._is_active(a, now) or not self.manager._is_active(b, now):
                self.peer_owner.pop(a, None)
                self.peer_owner.pop(b, None)
                self.lock_bad_votes.pop(pair, None)
                self.last_lock_audit.pop(pair, None)
                self.stats["lock_releases"] += 1
                continue

            obs_a = float(self.last_observed.get(a, 0.0))
            obs_b = float(self.last_observed.get(b, 0.0))
            prev_a, prev_b = self.last_lock_audit.get(pair, (0.0, 0.0))
            if obs_a <= prev_a + 1e-6 or obs_b <= prev_b + 1e-6:
                continue
            self.last_lock_audit[pair] = (obs_a, obs_b)

            measured = self._world_measure(a, b, now)
            geometry_good = False
            geometry_bad = False
            if measured is not None:
                rmse, _geo, _common = measured
                room_id = self.calibration.cameras[a[0]].room_id
                room = self.calibration.room(room_id)
                geometry_good = rmse <= room.match_distance_m
                geometry_bad = rmse >= room.veto_distance_m

            if geometry_good:
                self.lock_bad_votes[pair] = 0
                continue

            appearance = AdaptiveTrackletReID.tracklet_similarity(self, a, b, now)
            bad = geometry_bad or appearance <= self.release_floor
            self.lock_bad_votes[pair] = self.lock_bad_votes[pair] + 1 if bad else 0
            if self.lock_bad_votes[pair] < self.release_votes:
                continue

            loser = a if self._binding_strength(a, now) < self._binding_strength(b, now) else b
            self.peer_owner.pop(a, None)
            self.peer_owner.pop(b, None)
            self.lock_bad_votes.pop(pair, None)
            self.last_lock_audit.pop(pair, None)
            self._fresh_anchor(loser, now)
            self.stats["corrections"] += 1
            self.stats["lock_corrections"] += 1

    def snapshot(self) -> dict:
        row = super().snapshot()
        calibration = self.calibration.snapshot()
        row.update(
            {
                "calibration_ready_cameras": calibration["ready_cameras"],
                "calibration_sources": calibration["camera_sources"],
                "calibration_errors": calibration["errors"],
                "world_tracks": len(self.world_history),
                "geometry_pairs": self.geometry_pairs,
                "geometry_matches": self.geometry_matches,
                "geometry_vetoes": self.geometry_vetoes,
                "geometry_missing": self.geometry_missing,
                "world_rmse_m": self.last_world_rmse,
                "world_score": self.last_world_score,
                "world_common": self.last_world_common,
            }
        )
        return row
