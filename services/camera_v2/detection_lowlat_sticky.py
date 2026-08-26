from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass

from .detection_lowlat_nv12 import DetectionLowLatencyNv12


Box = tuple[float, float, float, float]
Row = tuple[Box, float]


@dataclass
class _StickyTrack:
    track_id: int
    bbox: Box
    display_bbox: Box
    confidence: float
    created_at: float
    last_match: float
    last_display: float
    hits: int = 1
    misses: int = 0
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class DetectionLowLatencySticky(DetectionLowLatencyNv12):
    """One-box-per-person presentation stabilizer on top of the low-latency detector.

    This is deliberately *not* a second detector and not NvDCF.  YOLO26 E2E stays
    NMS-free.  The code below only stabilizes already accepted person detections:

      * strict near-identical/nested duplicate collapse;
      * one-to-one temporal association per camera;
      * one missed detector refresh is tolerated;
      * short damped motion prediction + frame-rate display smoothing;
      * hard expiry so a departed person cannot leave a permanent ghost box.

    The wall path remains NV12 -> nvdsosd -> EGL.  All temporal work is tiny CPU
    bookkeeping over a handful of boxes and cannot block the detector GPU path.
    """

    def __init__(self) -> None:
        self._sticky_lock = threading.RLock()
        self._sticky_tracks: dict[str, list[_StickyTrack]] = {}
        self._sticky_next_id = 1
        self._sticky_updates = 0
        self._sticky_dup_removed = 0
        self._sticky_merged = 0

        self._sticky_dup_iou = self._env_float("CAMERA_V2_STICKY_DUP_IOU", 0.82, 0.70, 0.98)
        self._sticky_containment = self._env_float(
            "CAMERA_V2_STICKY_CONTAINMENT", 0.93, 0.80, 0.995
        )
        self._sticky_center_gate = self._env_float(
            "CAMERA_V2_STICKY_CENTER_GATE", 0.46, 0.15, 0.80
        )
        self._sticky_match_iou = self._env_float(
            "CAMERA_V2_STICKY_MATCH_IOU", 0.12, 0.02, 0.60
        )
        self._sticky_smooth_sec = self._env_float(
            "CAMERA_V2_STICKY_SMOOTH_SEC", 0.22, 0.05, 1.0
        )
        self._sticky_predict_sec = self._env_float(
            "CAMERA_V2_STICKY_PREDICT_SEC", 0.65, 0.0, 1.5
        )
        self._sticky_hard_ttl = self._env_float(
            "CAMERA_V2_STICKY_HARD_TTL_SEC", 8.0, 3.5, 15.0
        )
        self._sticky_miss_limit = max(
            0, min(3, int(os.environ.get("CAMERA_V2_STICKY_MISS_LIMIT", "1")))
        )
        super().__init__()
        self._sticky_tracks = {cid: [] for cid in self.sources}
        print(
            "CAMERA_STICKY_POLICY "
            f"dup_iou={self._sticky_dup_iou:.2f} containment={self._sticky_containment:.2f} "
            f"match_iou={self._sticky_match_iou:.2f} center_gate={self._sticky_center_gate:.2f} "
            f"miss_limit={self._sticky_miss_limit} hard_ttl={self._sticky_hard_ttl:.1f}s "
            f"smooth={self._sticky_smooth_sec:.2f}s predict={self._sticky_predict_sec:.2f}s "
            "nvdcf=0 display_ids=0",
            flush=True,
        )

    @staticmethod
    def _env_float(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(os.environ.get(name, str(default)))
        except Exception:
            value = default
        return max(low, min(high, value))

    @staticmethod
    def _area(box: Box) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    @classmethod
    def _iou(cls, a: Box, b: Box) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        union = cls._area(a) + cls._area(b) - inter
        return inter / max(1e-6, union)

    @classmethod
    def _containment(cls, a: Box, b: Box) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter / max(1e-6, min(cls._area(a), cls._area(b)))

    @staticmethod
    def _center_size(box: Box) -> tuple[float, float, float, float]:
        w = max(1.0, box[2] - box[0])
        h = max(1.0, box[3] - box[1])
        return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, w, h)

    @classmethod
    def _center_distance(cls, a: Box, b: Box) -> float:
        ax, ay, aw, ah = cls._center_size(a)
        bx, by, bw, bh = cls._center_size(b)
        norm = max(12.0, 0.5 * (math.hypot(aw, ah) + math.hypot(bw, bh)))
        return math.hypot(ax - bx, ay - by) / norm

    @classmethod
    def _size_similarity(cls, a: Box, b: Box) -> float:
        aa = cls._area(a)
        bb = cls._area(b)
        if aa <= 0.0 or bb <= 0.0:
            return 0.0
        return min(aa, bb) / max(aa, bb)

    def _is_near_duplicate(self, a: Box, b: Box) -> bool:
        iou = self._iou(a, b)
        if iou >= self._sticky_dup_iou:
            return True
        containment = self._containment(a, b)
        if containment < self._sticky_containment:
            return False
        center = self._center_distance(a, b)
        size = self._size_similarity(a, b)
        # Conservative nested-box guard: same centre and broadly similar scale.
        return center <= 0.16 and size >= 0.32

    def _collapse_duplicates(self, rows: list[Row]) -> tuple[list[Row], int]:
        ordered = sorted(rows, key=lambda item: float(item[1]), reverse=True)
        kept: list[Row] = []
        removed = 0
        for box, conf in ordered:
            box = tuple(float(v) for v in box)
            if any(self._is_near_duplicate(box, existing[0]) for existing in kept):
                removed += 1
                continue
            kept.append((box, float(conf)))
        return kept, removed

    def _match_score(self, track: _StickyTrack, detection: Box) -> float | None:
        iou = self._iou(track.bbox, detection)
        center = self._center_distance(track.bbox, detection)
        size = self._size_similarity(track.bbox, detection)
        if iou < self._sticky_match_iou and not (center <= self._sticky_center_gate and size >= 0.28):
            return None
        center_score = max(0.0, 1.0 - center / max(1e-6, self._sticky_center_gate))
        return 0.68 * iou + 0.22 * center_score + 0.10 * size

    @staticmethod
    def _blend(a: Box, b: Box, alpha: float) -> Box:
        beta = 1.0 - alpha
        return tuple(beta * float(x) + alpha * float(y) for x, y in zip(a, b))  # type: ignore[return-value]

    def _update_matched(self, track: _StickyTrack, box: Box, conf: float, now: float) -> None:
        dt = max(0.08, min(8.0, now - track.last_match))
        old = track.bbox
        raw_velocity = tuple((float(n) - float(o)) / dt for o, n in zip(old, box))
        # Damped and clamped.  It is only used for a short prediction horizon.
        max_vx = float(self.frame_width) * 0.22
        max_vy = float(self.frame_height) * 0.22
        limits = (max_vx, max_vy, max_vx, max_vy)
        velocity = []
        for old_v, raw_v, limit in zip(track.velocity, raw_velocity, limits):
            v = 0.55 * float(old_v) + 0.45 * float(raw_v)
            velocity.append(max(-limit, min(limit, v)))
        track.velocity = tuple(velocity)  # type: ignore[assignment]
        # Keep enough of the old rectangle to remove detector jitter, but follow
        # a genuine new measurement quickly.
        track.bbox = self._blend(old, box, 0.72)
        track.confidence = max(float(conf), 0.65 * track.confidence)
        track.last_match = now
        track.hits += 1
        track.misses = 0

    def _merge_duplicate_tracks(self, tracks: list[_StickyTrack]) -> tuple[list[_StickyTrack], int]:
        if len(tracks) < 2:
            return tracks, 0
        ordered = sorted(tracks, key=lambda t: (t.hits, -t.track_id), reverse=True)
        kept: list[_StickyTrack] = []
        merged = 0
        for track in ordered:
            duplicate = None
            for existing in kept:
                if self._is_near_duplicate(track.bbox, existing.bbox):
                    duplicate = existing
                    break
            if duplicate is None:
                kept.append(track)
                continue
            # Preserve the more established track and its visual continuity.
            if track.last_match > duplicate.last_match:
                duplicate.bbox = track.bbox
                duplicate.confidence = max(duplicate.confidence, track.confidence)
                duplicate.last_match = track.last_match
                duplicate.velocity = track.velocity
            duplicate.hits = max(duplicate.hits, track.hits)
            duplicate.misses = min(duplicate.misses, track.misses)
            merged += 1
        return kept, merged

    def _update_sticky_tracks(self, cid: str, rows: list[Row], now: float) -> dict[str, int]:
        candidates, dup_removed = self._collapse_duplicates(rows)
        tracks = list(self._sticky_tracks.get(cid, ()))

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(tracks):
            for di, (box, _conf) in enumerate(candidates):
                score = self._match_score(track, box)
                if score is not None:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            box, conf = candidates[di]
            self._update_matched(tracks[ti], box, conf, now)
            matched_tracks.add(ti)
            matched_dets.add(di)

        held = 0
        removed = 0
        survivors: list[_StickyTrack] = []
        for ti, track in enumerate(tracks):
            if ti not in matched_tracks:
                track.misses += 1
            age = now - track.last_match
            if age > self._sticky_hard_ttl or track.misses > self._sticky_miss_limit:
                removed += 1
                continue
            if ti not in matched_tracks:
                held += 1
                # Do not let stale velocity drift forever after a detector miss.
                track.velocity = tuple(v * 0.35 for v in track.velocity)  # type: ignore[assignment]
            survivors.append(track)

        created = 0
        for di, (box, conf) in enumerate(candidates):
            if di in matched_dets:
                continue
            track = _StickyTrack(
                track_id=self._sticky_next_id,
                bbox=box,
                display_bbox=box,
                confidence=float(conf),
                created_at=now,
                last_match=now,
                last_display=now,
            )
            self._sticky_next_id += 1
            survivors.append(track)
            created += 1

        survivors, track_merges = self._merge_duplicate_tracks(survivors)
        self._sticky_tracks[cid] = survivors
        self._sticky_dup_removed += dup_removed
        self._sticky_merged += track_merges
        self._sticky_updates += 1
        return {
            "input": len(rows),
            "dedup_removed": dup_removed,
            "matched": len(matched_tracks),
            "new": created,
            "held": held,
            "removed": removed,
            "track_merges": track_merges,
            "tracks": len(survivors),
        }

    def _store_native_detection(
        self,
        cid: str,
        captured_t: float,
        native_rows,
        *,
        count_call: bool,
        batch_ms: float,
    ) -> None:
        rows: list[Row] = [
            (tuple(float(v) for v in coords), float(conf))
            for coords, conf in native_rows
        ]
        now = time.monotonic()
        with self._sticky_lock:
            diag = self._update_sticky_tracks(cid, rows, now)
        # Keep detector truth untouched for diagnostics/rescue policy.  The sticky
        # layer is presentation state only.
        super()._store_native_detection(
            cid,
            captured_t,
            native_rows,
            count_call=count_call,
            batch_ms=batch_ms,
        )
        print(
            "CAMERA_STICKY_UPDATE "
            f"cid={cid} in={diag['input']} dedup_removed={diag['dedup_removed']} "
            f"matched={diag['matched']} new={diag['new']} held={diag['held']} "
            f"removed={diag['removed']} merged={diag['track_merges']} tracks={diag['tracks']}",
            flush=True,
        )

    def _predicted_display_box(self, track: _StickyTrack, now: float) -> Box:
        since_match = max(0.0, now - track.last_match)
        horizon = min(self._sticky_predict_sec, since_match)
        predicted = tuple(
            float(value) + float(velocity) * horizon
            for value, velocity in zip(track.bbox, track.velocity)
        )
        max_x = float(self.frame_width - 1)
        max_y = float(self.frame_height - 1)
        x1 = max(0.0, min(max_x - 1.0, predicted[0]))
        y1 = max(0.0, min(max_y - 1.0, predicted[1]))
        x2 = max(x1 + 1.0, min(max_x, predicted[2]))
        y2 = max(y1 + 1.0, min(max_y, predicted[3]))
        target = (x1, y1, x2, y2)

        frame_dt = max(0.001, min(0.25, now - track.last_display))
        alpha = min(1.0, frame_dt / max(0.01, self._sticky_smooth_sec))
        track.display_bbox = self._blend(track.display_bbox, target, alpha)
        track.last_display = now
        return track.display_bbox

    def _wall_bbox_rows(self, now: float):
        tile_w = float(self.wall_width) / float(max(1, self.tiler_columns))
        tile_h = float(self.wall_height) / float(max(1, self.tiler_rows))
        sx = tile_w / float(max(1, self.frame_width))
        sy = tile_h / float(max(1, self.frame_height))
        output = []

        with self._sticky_lock:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                tracks = self._sticky_tracks.get(cid, ())
                col = index % max(1, self.tiler_columns)
                row = index // max(1, self.tiler_columns)
                ox = float(col) * tile_w
                oy = float(row) * tile_h
                for track in tracks:
                    if now - track.last_match > self._sticky_hard_ttl:
                        continue
                    x1, y1, x2, y2 = self._predicted_display_box(track, now)
                    wx1 = max(ox, min(ox + tile_w - 1.0, ox + x1 * sx))
                    wy1 = max(oy, min(oy + tile_h - 1.0, oy + y1 * sy))
                    wx2 = max(wx1 + 1.0, min(ox + tile_w, ox + x2 * sx))
                    wy2 = max(wy1 + 1.0, min(oy + tile_h, oy + y2 * sy))
                    output.append((wx1, wy1, wx2, wy2, float(track.confidence)))
        return output

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._sticky_lock:
            per_camera = {
                cid: len(self._sticky_tracks.get(cid, ()))
                for cid in self.sources
            }
            total = sum(per_camera.values())
        summary = " ".join(f"{cid}:{per_camera[cid]}" for cid in self.sources)
        print(
            "CAMERA_STICKY_STATS "
            f"tracks={total} per_camera=[{summary}] updates={self._sticky_updates} "
            f"dup_removed={self._sticky_dup_removed} merged={self._sticky_merged}",
            flush=True,
        )
        return keep


def main() -> int:
    return DetectionLowLatencySticky().run()


if __name__ == "__main__":
    raise SystemExit(main())
