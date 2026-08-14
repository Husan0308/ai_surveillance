from __future__ import annotations

import math

import cv2
import numpy as np

from .instant_reid_safe import SafeInstantGlobalReIdCoordinator
from .local_tracker import linear_sum_assignment


class RoomConsensusGlobalReIdCoordinator(SafeInstantGlobalReIdCoordinator):
    """Cross-camera ReID tuned for fixed paired CCTV cameras.

    Local tracking remains authoritative. Global IDs are reconciled by one-to-one
    room-pair consensus using multi-shot OSNet appearance plus a weak chroma cue.
    Short same-camera fragmentation is repaired from the real publisher active-set,
    not from a coarse timeout, so a temporary detector miss does not mint a new ID.
    """

    def __init__(self, *args, **kwargs):
        config = dict(args[2] if len(args) >= 3 else kwargs.get("config") or {})
        super().__init__(*args, **kwargs)
        self.room_embedding_weight = float(config.get("room_embedding_weight", 0.90))
        self.room_colour_weight = float(config.get("room_colour_weight", 0.10))
        total = max(1e-6, self.room_embedding_weight + self.room_colour_weight)
        self.room_embedding_weight /= total
        self.room_colour_weight /= total
        self.room_min_embedding_similarity = float(
            config.get("room_min_embedding_similarity", 0.66)
        )
        self.room_pair_similarity = float(config.get("room_pair_similarity", 0.75))
        self.room_single_similarity = float(config.get("room_single_similarity", 0.71))
        self.room_pair_margin = max(0.0, float(config.get("room_pair_margin", 0.035)))
        self.room_confirmed_merge_similarity = float(
            config.get("room_confirmed_merge_similarity", 0.90)
        )
        self.same_camera_handoff_sec = max(
            0.2, float(config.get("same_camera_handoff_sec", 2.6))
        )
        self.same_camera_handoff_similarity = float(
            config.get("same_camera_handoff_similarity", 0.84)
        )
        self.same_camera_handoff_distance = max(
            0.05, float(config.get("same_camera_handoff_distance", 0.85))
        )
        self.colour_ema_alpha = min(
            1.0, max(0.05, float(config.get("colour_ema_alpha", 0.28)))
        )

        self._colour_signatures: dict[tuple[str, int], np.ndarray] = {}
        self._active_track_keys: set[tuple[str, int]] = set()
        self._room_assignment_cycles = 0
        self._room_pair_matches = 0
        self._room_pair_rejects = 0
        self._room_pair_ambiguous = 0
        self._same_camera_handoffs = 0
        self._last_room_decisions: list[dict] = []

    @staticmethod
    def _bbox_center_distance(a, b) -> float:
        if not a or not b:
            return math.inf
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        aw = max(1.0, ax2 - ax1)
        ah = max(1.0, ay2 - ay1)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        scale = max(1.0, 0.5 * (math.hypot(aw, ah) + math.hypot(bw, bh)))
        return math.hypot(acx - bcx, acy - bcy) / scale

    @staticmethod
    def _torso_colour_signature(image: np.ndarray, bbox) -> np.ndarray | None:
        if image is None or image.size == 0 or not bbox or len(bbox) != 4:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 20:
            return None
        ix1 = max(0, int(x1 + 0.16 * bw))
        ix2 = min(w, int(x2 - 0.16 * bw))
        iy1 = max(0, int(y1 + 0.18 * bh))
        iy2 = min(h, int(y1 + 0.72 * bh))
        if ix2 - ix1 < 6 or iy2 - iy1 < 12:
            return None
        crop = image[iy1:iy2, ix1:ix2]
        crop = cv2.resize(crop, (32, 48), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [1, 2], None, [8, 8], [0, 256, 0, 256])
        hist = hist.astype(np.float32).reshape(-1)
        total = float(hist.sum())
        if total <= 1e-9:
            return None
        return hist / total

    @staticmethod
    def _colour_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if a is None or b is None:
            return None
        aa = np.asarray(a, dtype=np.float32).reshape(-1, 1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1, 1)
        distance = float(cv2.compareHist(aa, bb, cv2.HISTCMP_BHATTACHARYYA))
        return max(0.0, min(1.0, 1.0 - distance))

    def _update_colour_signatures(self, now: float) -> None:
        active: set[tuple[str, int]] = set()
        for camera_id, store in self.stores.items():
            frame, _version = store.get()
            publisher = self.publishers.get(camera_id)
            if frame is None or publisher is None:
                continue
            tracks = publisher.track_snapshot()
            for track in tracks:
                try:
                    track_id = int(track.get("track_id") or 0)
                except (TypeError, ValueError):
                    continue
                bbox = track.get("bbox") or []
                if track_id <= 0 or len(bbox) != 4:
                    continue
                key = (str(camera_id), track_id)
                active.add(key)
                state = self._tracks.get(key)
                if state is not None:
                    state.last_seen = max(state.last_seen, now)
                    state.bbox = tuple(float(v) for v in bbox)
                signature = self._torso_colour_signature(frame.image, bbox)
                if signature is None:
                    continue
                old = self._colour_signatures.get(key)
                if old is None:
                    self._colour_signatures[key] = signature
                else:
                    alpha = self.colour_ema_alpha
                    merged = (1.0 - alpha) * old + alpha * signature
                    denom = float(merged.sum())
                    self._colour_signatures[key] = merged / max(1e-9, denom)
        self._active_track_keys = active

    def _collect_candidates(self, now: float):
        candidates = super()._collect_candidates(now)
        with self._lock:
            self._update_colour_signatures(now)
            self._repair_same_camera_fragments(now)
            self._reconcile_overlap_pairs(now)
        return candidates

    def _pair_score(self, a, b) -> tuple[float, float, float | None]:
        emb = self._cosine_pair(
            self._tracklet_prototype(a), self._tracklet_prototype(b)
        )
        colour = self._colour_similarity(
            self._colour_signatures.get((a.camera_id, a.track_id)),
            self._colour_signatures.get((b.camera_id, b.track_id)),
        )
        if colour is None:
            return emb, emb, None
        score = self.room_embedding_weight * emb + self.room_colour_weight * colour
        return float(score), float(emb), float(colour)

    def _handoff_merge(self, current, previous, similarity: float, now: float) -> str | None:
        current_gid = self._canonical_gid(current.global_id)
        previous_gid = self._canonical_gid(previous.global_id)
        if not current_gid or not previous_gid:
            return None
        if current_gid == previous_gid:
            return current_gid

        # Publisher active-set is authoritative here. Refuse only if the previous
        # identity is truly attached to another currently visible track in camera.
        for key in self._active_track_keys:
            if key == (current.camera_id, current.track_id):
                continue
            state = self._tracks.get(key)
            if state is not None and self._canonical_gid(state.global_id) == previous_gid:
                return None

        target = self._globals.get(previous_gid)
        source = self._globals.get(current_gid)
        if target is None or source is None:
            return None

        self._aliases[current_gid] = previous_gid
        current.global_id = previous_gid
        current.assignment_similarity = float(similarity)
        current.assignment_reason = "same_camera_track_handoff"
        target.last_seen = max(target.last_seen, now)
        target.last_camera = current.camera_id
        if similarity >= self.prototype_update_similarity:
            alpha = min(0.12, self.prototype_update_alpha)
            target.prototype = (
                (1.0 - alpha) * target.prototype + alpha * source.prototype
            )
            norm = float(np.linalg.norm(target.prototype))
            if norm > 1e-9:
                target.prototype = target.prototype / norm
        self._provisional_globals.discard(current_gid)
        self._same_camera_handoffs += 1
        return previous_gid

    def _repair_same_camera_fragments(self, now: float) -> None:
        active_states = [
            state
            for key, state in self._tracks.items()
            if key in self._active_track_keys and state.embeddings and state.global_id
        ]
        for current in active_states:
            current_key = (current.camera_id, current.track_id)
            current_gid = self._canonical_gid(current.global_id)
            if not current_gid:
                continue
            current_proto = self._tracklet_prototype(current)
            best = None
            for key, previous in self._tracks.items():
                if key == current_key or key in self._active_track_keys:
                    continue
                if previous.camera_id != current.camera_id:
                    continue
                gap = now - previous.last_seen
                if gap <= 0.0 or gap > self.same_camera_handoff_sec:
                    continue
                prev_gid = self._canonical_gid(previous.global_id)
                if not prev_gid or prev_gid == current_gid or not previous.embeddings:
                    continue
                distance = self._bbox_center_distance(previous.bbox, current.bbox)
                if distance > self.same_camera_handoff_distance:
                    continue
                similarity = self._cosine_pair(
                    current_proto, self._tracklet_prototype(previous)
                )
                if similarity < self.same_camera_handoff_similarity:
                    continue
                candidate = (similarity, -gap, previous)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if best is None:
                continue
            similarity, _neg_gap, previous = best
            self._handoff_merge(current, previous, float(similarity), now)

    def _reconcile_overlap_pairs(self, now: float):
        self._room_assignment_cycles += 1
        decisions: list[dict] = []
        for group in self.overlap_groups:
            cameras = sorted(group)
            if len(cameras) != 2:
                continue
            left = [
                state
                for key, state in self._tracks.items()
                if key in self._active_track_keys
                and state.camera_id == cameras[0]
                and state.embeddings
                and state.global_id
            ]
            right = [
                state
                for key, state in self._tracks.items()
                if key in self._active_track_keys
                and state.camera_id == cameras[1]
                and state.embeddings
                and state.global_id
            ]
            if not left or not right:
                continue

            scores = np.full((len(left), len(right)), -1.0, dtype=np.float32)
            embeddings = np.full_like(scores, -1.0)
            colours = np.full_like(scores, np.nan)
            for i, a in enumerate(left):
                for j, b in enumerate(right):
                    score, emb, colour = self._pair_score(a, b)
                    scores[i, j] = score
                    embeddings[i, j] = emb
                    if colour is not None:
                        colours[i, j] = colour

            rows, cols = linear_sum_assignment(1.0 - scores)
            single = len(left) == 1 and len(right) == 1
            for row, col in zip(rows.tolist(), cols.tolist()):
                a, b = left[row], right[col]
                score = float(scores[row, col])
                emb = float(embeddings[row, col])
                colour = None if np.isnan(colours[row, col]) else float(colours[row, col])
                gid_a = self._canonical_gid(a.global_id)
                gid_b = self._canonical_gid(b.global_id)
                row_values = np.sort(scores[row])[::-1]
                col_values = np.sort(scores[:, col])[::-1]
                row_margin = score - float(row_values[1]) if len(row_values) > 1 else 1.0
                col_margin = score - float(col_values[1]) if len(col_values) > 1 else 1.0
                margin = min(row_margin, col_margin)
                decision = {
                    "pair": f"{cameras[0]}|{cameras[1]}",
                    "left_track": int(a.track_id),
                    "right_track": int(b.track_id),
                    "score": round(score, 4),
                    "embedding": round(emb, 4),
                    "colour": round(colour, 4) if colour is not None else None,
                    "margin": round(margin, 4),
                    "matched": False,
                    "reason": "",
                }
                decisions.append(decision)
                if not gid_a or not gid_b or gid_a == gid_b:
                    decision["reason"] = "already_same_or_missing"
                    continue
                if emb < self.room_min_embedding_similarity:
                    self._room_pair_rejects += 1
                    decision["reason"] = "embedding_gate"
                    continue

                threshold = self.room_single_similarity if single else self.room_pair_similarity
                both_confirmed = (
                    gid_a not in self._provisional_globals
                    and gid_b not in self._provisional_globals
                )
                if both_confirmed:
                    threshold = max(threshold, self.room_confirmed_merge_similarity)
                if score < threshold:
                    self._room_pair_rejects += 1
                    decision["reason"] = "score_gate"
                    continue
                if margin < self.room_pair_margin:
                    self._room_pair_ambiguous += 1
                    decision["reason"] = "assignment_margin"
                    continue

                merged = self._merge_global_ids(gid_a, gid_b, now, score)
                if merged is None:
                    self._room_pair_rejects += 1
                    decision["reason"] = "active_conflict"
                    continue
                a.global_id = merged
                b.global_id = merged
                a.assignment_similarity = score
                b.assignment_similarity = score
                a.assignment_reason = "room_pair_consensus"
                b.assignment_reason = "room_pair_consensus"
                self._room_pair_matches += 1
                self._pair_reconciles += 1
                decision["matched"] = True
                decision["reason"] = "room_pair_consensus"
        self._last_room_decisions = decisions[-12:]

    def metrics(self):
        payload = super().metrics()
        payload.update(
            {
                "mode": "cpu-osnet-ain-room-consensus",
                "global_id_policy": "room-pair-hungarian+appearance+colour+handoff",
                "room_assignment_cycles": self._room_assignment_cycles,
                "room_pair_matches": self._room_pair_matches,
                "room_pair_rejects": self._room_pair_rejects,
                "room_pair_ambiguous": self._room_pair_ambiguous,
                "same_camera_handoffs": self._same_camera_handoffs,
                "colour_signatures": len(self._colour_signatures),
                "room_embedding_weight": self.room_embedding_weight,
                "room_colour_weight": self.room_colour_weight,
                "last_room_decisions": list(self._last_room_decisions),
            }
        )
        return payload
