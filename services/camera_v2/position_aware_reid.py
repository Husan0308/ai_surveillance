from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from .adaptive_reid import AdaptiveTrackletReID
from .stable_adaptive_reid import StableAdaptiveTrackletReID

LocalKey = tuple[int, int]


@dataclass
class SeatAnchor:
    seat_id: int
    x: float
    y: float
    samples: int
    last_seen: float


class PositionAwareAdaptiveTrackletReID(StableAdaptiveTrackletReID):
    """Sticky tracklet ReID plus self-learned stationary seat correspondence.

    The office is dominated by seated workers. Instead of training on a repetitive
    office dataset, this layer learns camera-local stationary position anchors from
    bbox foot-points. Only already-confirmed peer-camera identity leases are allowed
    to teach a seat correspondence. Once learned, the physical seat mapping becomes
    a strong geometric prior: the mapped peer seat boosts the true ReID candidate,
    while a known different seat is a hard veto for simultaneous peer tracks.

    No camera calibration is required for this first geometry layer and no neural
    weights are updated. A later homography/world-ground-plane layer can replace
    the learned seat map without changing the identity controller contract.
    """

    def __init__(self, manager, *, frame_width: int, frame_height: int) -> None:
        self.frame_width = max(1.0, float(frame_width))
        self.frame_height = max(1.0, float(frame_height))

        self.position_window = max(4, min(20, int(os.environ.get("CAMERA_V2_POS_WINDOW", "8"))))
        self.position_ttl = max(3.0, float(os.environ.get("CAMERA_V2_POS_TTL", "16")))
        self.position_min_samples = max(4, int(os.environ.get("CAMERA_V2_POS_MIN_SAMPLES", "4")))
        self.stationary_spread = max(0.010, float(os.environ.get("CAMERA_V2_POS_STATIONARY_SPREAD", "0.040")))
        self.seat_radius = max(0.025, float(os.environ.get("CAMERA_V2_POS_SEAT_RADIUS", "0.085")))
        self.seat_anchor_alpha = min(0.25, max(0.02, float(os.environ.get("CAMERA_V2_POS_ANCHOR_ALPHA", "0.08"))))
        self.seat_boost = min(0.20, max(0.02, float(os.environ.get("CAMERA_V2_POS_MATCH_BOOST", "0.11"))))
        self.seat_link_votes_required = max(3, int(os.environ.get("CAMERA_V2_POS_LINK_VOTES", "3")))
        self.seat_link_min_reid = float(os.environ.get("CAMERA_V2_POS_LINK_MIN_REID", "0.52"))
        self.max_seats_per_camera = max(4, min(32, int(os.environ.get("CAMERA_V2_POS_MAX_SEATS", "16"))))
        self.anchor_ttl = max(1800.0, float(os.environ.get("CAMERA_V2_POS_ANCHOR_TTL", "21600")))

        self.positions: dict[LocalKey, deque[tuple[float, float, float]]] = {}
        self.seat_anchors: dict[int, list[SeatAnchor]] = defaultdict(list)
        self.track_seat: dict[LocalKey, int] = {}
        self.next_seat_id: dict[int, int] = defaultdict(lambda: 1)

        # (source, local_seat, peer_source) -> peer_local_seat
        self.seat_map: dict[tuple[int, int, int], int] = {}
        self.link_votes: dict[tuple[int, int, int, int], int] = defaultdict(int)
        self.last_link_observed: dict[tuple[int, int, int, int], tuple[float, float]] = {}

        self.position_matches = 0
        self.position_vetoes = 0
        self.seat_links_learned = 0
        self.seat_link_conflicts = 0
        self.stationary_tracks = 0

        super().__init__(manager)

    @staticmethod
    def _bbox(row: dict) -> tuple[float, float, float, float] | None:
        raw = row.get("bbox")
        if raw is None or len(raw) < 4:
            return None
        try:
            x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _observe_position(self, key: LocalKey, row: dict, now: float) -> None:
        bbox = self._bbox(row)
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        h = max(1.0, y2 - y1)
        # Bottom-center is NVIDIA's standard image-plane proxy for the person's
        # ground contact. Lift 5% of bbox height to reduce chair/box edge jitter.
        x = ((x1 + x2) * 0.5) / self.frame_width
        y = (y2 - 0.05 * h) / self.frame_height
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        history = self.positions.get(key)
        if history is None:
            history = deque(maxlen=self.position_window)
            self.positions[key] = history
        history.append((x, y, now))

    def _stationary_center(self, key: LocalKey, now: float) -> tuple[float, float] | None:
        history = self.positions.get(key)
        if not history:
            return None
        rows = [(x, y) for x, y, seen in history if now - seen <= self.position_ttl]
        if len(rows) < self.position_min_samples:
            return None
        xs = sorted(x for x, _ in rows)
        ys = sorted(y for _, y in rows)
        cx = xs[len(xs) // 2]
        cy = ys[len(ys) // 2]
        rms = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in rows) / len(rows))
        if rms > self.stationary_spread:
            return None
        return cx, cy

    def _assign_seat(self, key: LocalKey, now: float) -> int | None:
        center = self._stationary_center(key, now)
        if center is None:
            return None
        source_id = int(key[0])
        cx, cy = center
        anchors = self.seat_anchors[source_id]

        # Remove extremely stale auto anchors only if the camera has accumulated
        # many alternatives. Normal office seats otherwise remain persistent.
        if len(anchors) >= self.max_seats_per_camera:
            anchors[:] = [a for a in anchors if now - a.last_seen <= self.anchor_ttl]

        best = None
        best_dist = 1e9
        for anchor in anchors:
            dist = math.hypot(cx - anchor.x, cy - anchor.y)
            if dist < best_dist:
                best = anchor
                best_dist = dist

        if best is None or best_dist > self.seat_radius:
            if len(anchors) >= self.max_seats_per_camera:
                return None
            best = SeatAnchor(
                seat_id=self.next_seat_id[source_id],
                x=cx,
                y=cy,
                samples=1,
                last_seen=now,
            )
            self.next_seat_id[source_id] += 1
            anchors.append(best)
        else:
            alpha = self.seat_anchor_alpha
            best.x = (1.0 - alpha) * best.x + alpha * cx
            best.y = (1.0 - alpha) * best.y + alpha * cy
            best.samples += 1
            best.last_seen = now

        self.track_seat[key] = best.seat_id
        return best.seat_id

    def _seat_for(self, key: LocalKey, now: float) -> int | None:
        # Seat priors are valid only while the current track remains stationary.
        if self._stationary_center(key, now) is None:
            return None
        seat = self.track_seat.get(key)
        if seat is None:
            seat = self._assign_seat(key, now)
        return seat

    @staticmethod
    def _canonical_link(source_a: int, seat_a: int, source_b: int, seat_b: int) -> tuple[int, int, int, int]:
        if source_a <= source_b:
            return source_a, seat_a, source_b, seat_b
        return source_b, seat_b, source_a, seat_a

    def _position_relation(self, a: LocalKey, b: LocalKey, now: float) -> str:
        if a[0] == b[0] or self.manager.room_of(a[0]) != self.manager.room_of(b[0]):
            return "none"
        seat_a = self._seat_for(a, now)
        seat_b = self._seat_for(b, now)
        if seat_a is None or seat_b is None:
            return "none"

        mapped_a = self.seat_map.get((a[0], seat_a, b[0]))
        mapped_b = self.seat_map.get((b[0], seat_b, a[0]))
        if mapped_a is not None and mapped_a != seat_b:
            return "conflict"
        if mapped_b is not None and mapped_b != seat_a:
            return "conflict"
        if mapped_a == seat_b or mapped_b == seat_a:
            return "match"
        return "unmapped"

    def tracklet_similarity(self, a, b, now: float) -> float:
        score = super().tracklet_similarity(a, b, now)
        if score < -0.5:
            return score
        relation = self._position_relation(a, b, now)
        if relation == "conflict":
            self.position_vetoes += 1
            return -1.0
        if relation == "match":
            self.position_matches += 1
            return min(1.0, score + self.seat_boost)
        return score

    def observe_rows(self, rows: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        for row in rows:
            sid = int(row.get("source_id", -1))
            oid = int(row.get("object_id", -1))
            if sid < 0 or oid < 0:
                continue
            key = (sid, oid)
            self._observe_position(key, row, now)
            self._assign_seat(key, now)
        super().observe_rows(rows, now)

        stale = [
            key
            for key, history in self.positions.items()
            if not history or now - history[-1][2] > self.bank_ttl * 1.5
        ]
        for key in stale:
            self.positions.pop(key, None)
            self.track_seat.pop(key, None)

    def _raw_tracklet_similarity(self, a: LocalKey, b: LocalKey, now: float) -> float:
        # Bypass sticky/position bonuses when teaching geometry. Seat links must be
        # learned from the specialized appearance signal, not from themselves.
        return AdaptiveTrackletReID.tracklet_similarity(self, a, b, now)

    def _learn_seat_links(self, now: float) -> None:
        seen_pairs: set[frozenset[LocalKey]] = set()
        stationary = 0
        for key in list(self.manager.bindings):
            if self.manager._is_active(key, now) and self._seat_for(key, now) is not None:
                stationary += 1
        self.stationary_tracks = stationary

        for a, b in list(self.peer_owner.items()):
            pair = frozenset((a, b))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if self.peer_owner.get(b) != a:
                continue
            if not self.manager._is_active(a, now) or not self.manager._is_active(b, now):
                continue
            if a[0] == b[0] or self.manager.room_of(a[0]) != self.manager.room_of(b[0]):
                continue

            seat_a = self._seat_for(a, now)
            seat_b = self._seat_for(b, now)
            if seat_a is None or seat_b is None:
                continue

            raw_score = self._raw_tracklet_similarity(a, b, now)
            camera_pair = (min(a[0], b[0]), max(a[0], b[0]))
            learn_threshold = max(self.seat_link_min_reid, self._camera_threshold(camera_pair))
            if raw_score < learn_threshold:
                continue

            canonical = self._canonical_link(a[0], seat_a, b[0], seat_b)
            a_seen = self.positions[a][-1][2] if self.positions.get(a) else 0.0
            b_seen = self.positions[b][-1][2] if self.positions.get(b) else 0.0
            prev_a, prev_b = self.last_link_observed.get(canonical, (0.0, 0.0))
            if a_seen <= prev_a + 1e-6 or b_seen <= prev_b + 1e-6:
                continue
            self.last_link_observed[canonical] = (a_seen, b_seen)

            existing_a = self.seat_map.get((a[0], seat_a, b[0]))
            existing_b = self.seat_map.get((b[0], seat_b, a[0]))
            if (existing_a is not None and existing_a != seat_b) or (
                existing_b is not None and existing_b != seat_a
            ):
                self.seat_link_conflicts += 1
                continue

            self.link_votes[canonical] += 1
            if self.link_votes[canonical] < self.seat_link_votes_required:
                continue

            if existing_a is None and existing_b is None:
                self.seat_map[(a[0], seat_a, b[0])] = seat_b
                self.seat_map[(b[0], seat_b, a[0])] = seat_a
                self.seat_links_learned += 1
            self.link_votes[canonical] = 0

    def reconcile(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        super().reconcile(now)
        self._learn_seat_links(now)

    def snapshot(self) -> dict:
        row = super().snapshot()
        row.update(
            {
                "position_tracks": len(self.positions),
                "stationary_tracks": self.stationary_tracks,
                "seat_anchors": sum(len(rows) for rows in self.seat_anchors.values()),
                "seat_links": len(self.seat_map) // 2,
                "seat_links_learned": self.seat_links_learned,
                "seat_link_conflicts": self.seat_link_conflicts,
                "position_matches": self.position_matches,
                "position_vetoes": self.position_vetoes,
                "seat_boost": self.seat_boost,
            }
        )
        return row
