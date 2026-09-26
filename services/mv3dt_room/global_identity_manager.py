#!/usr/bin/env python3
"""Experiment-only persistent application identity manager for Dev Room."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from services.mv3dt_room.identity_gallery_store import IdentityGalleryStore

CAMS = ("CAM-01", "CAM-04")
FPS = 20.0

# Empirical gates derived from the accepted synchronized two-person replay.
# Same-person cross-camera world distances reached 8.3062 m; the closest
# different-person pair was 8.6353 m. Same-person OSNet similarity reached
# 0.5050, while geometry separates the different-person pairs.
IDENTITY_GATES = {
    "source_fps": FPS,
    "cross_camera_time_ms": 50.0,  # one 20 FPS source period plus timestamp jitter
    "cross_camera_world_candidate_m": 8.5,
    "cross_camera_world_tight_m": 1.5,
    "cross_camera_similarity_min": 0.50,
    # Appearance without geometric/native continuity needs independent proof.
    # Captured true reacquisitions: 0.792-0.916; first false match: 0.530.
    "gallery_only_similarity_min": 0.70,
    # Recent room-position memory covers the observed 10+ minute test/restart
    # interval, but expires before the one-hour appearance-only test.
    "recent_anchor_max_age_ms": 30.0 * 60.0 * 1000.0,
    "recent_anchor_similarity_min": 0.60,
    "same_camera_direct_gap_frames": 12,
    # Captured CAM-01 fragment: 36-frame gap, 0.5 m world shift, IoU >0.9.
    # The longer handoff needs all three independent causal supports.
    "same_camera_strong_image_gap_frames": 40,
    "same_camera_strong_image_distance_m": 1.0,
    "same_camera_strong_image_iou": 0.70,
    # In the accepted replay a confirmed CAM-04 track fragmented for 43
    # source frames before its replacement. Its box still overlapped by
    # 0.57 IoU, moved 4 m in 2.35 s, and had three independent usable crops
    # with three consistent similarities 0.47-0.53. No different-person pair
    # in the replay
    # meets this joint image/world/time gate. Never use this without ReID.
    "same_camera_supported_gap_frames": 50,
    "same_camera_supported_distance_m": 4.5,
    "same_camera_supported_iou": 0.55,
    "same_camera_supported_time_ms": 2500.0,
    "same_camera_supported_min_crops": 3,
    "same_camera_supported_similarity_min": 0.45,
    "same_camera_direct_distance_m": 3.0,
    "same_camera_speed_m_per_frame": 0.25,
    "same_person_world_max_observed_m": 8.3062,
    "different_person_world_min_observed_m": 8.6353,
    "same_person_similarity_min_observed": 0.5050,
    # Ambiguous persistent-gallery matches remain PENDING. This is an
    # ambiguity guard, not a relaxed match threshold.
    "persistent_reacquisition_similarity_margin": 0.05,
}


def observation_time_ms(left, right):
    """Return absolute source-time difference, falling back to frame cadence."""
    lt, rt = left.get("timestamp"), right.get("timestamp")
    if lt and rt:
        try:
            a = datetime.fromisoformat(str(lt).replace("Z", "+00:00"))
            b = datetime.fromisoformat(str(rt).replace("Z", "+00:00"))
            return abs((a - b).total_seconds()) * 1000.0
        except (TypeError, ValueError):
            pass
    lf, rf = left.get("frame"), right.get("frame")
    if lf is None or rf is None:
        return None
    return abs(int(lf) - int(rf)) * 1000.0 / FPS


def _quality_score(obs):
    quality = obs.get("crop_quality") or {}
    if not quality:
        return 1.0
    if obs.get("crop_quality_usable") is False or quality.get("border_truncated"):
        return 0.0
    area = min(1.0, float(quality.get("area_fraction", 0.0)) / 0.02)
    inside = min(1.0, max(0.0, float(quality.get("inside_fraction", 0.0))))
    height = min(1.0, float(quality.get("height", 0.0)) / 400.0)
    return 0.45 * area + 0.35 * inside + 0.20 * height


def bbox_of(obj):
    b = obj.get("bbox", {})
    return (
        float(b.get("leftX", 0.0)),
        float(b.get("topY", 0.0)),
        float(b.get("rightX", 0.0)),
        float(b.get("bottomY", 0.0)),
    )


def world_of(obj):
    xyz = obj.get("bbox3d", {}).get("coordinates", [])
    if len(xyz) < 2:
        return None
    point = float(xyz[0]), float(xyz[1])
    return point if all(math.isfinite(v) for v in point) else None


def distance(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bbox_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)

def image_continuity(a, b, gap):
    if a is None or b is None or gap < 0 or gap > 30:
        return False
    acx, bcx = (a[0] + a[2]) / 2.0, (b[0] + b[2]) / 2.0
    if a[1] <= 3.0 and b[1] <= 3.0:
        return abs(acx - bcx) <= 50.0 + 12.0 * gap
    ah, bh = max(a[3] - a[1], 1.0), max(b[3] - b[1], 1.0)
    foot_delta = math.hypot(acx - bcx, a[3] - b[3])
    return bbox_iou(a, b) >= 0.15 and foot_delta <= 0.35 * max(ah, bh) + 8.0 * gap


@dataclass
class GalleryEntry:
    vector: np.ndarray
    quality: float
    frame: int


@dataclass
class Identity:
    number: int
    created_frame: int
    galleries: dict = field(default_factory=lambda: defaultdict(list))
    gallery_quality: dict = field(default_factory=lambda: defaultdict(list))
    last_by_camera: dict = field(default_factory=dict)
    native_keys: set = field(default_factory=set)
    samples: int = 0
    persisted_anchor: dict | None = None
    last_persisted_frame: int = -1
    # Bounded history prevents a fragmented native track from erasing an
    # older compatible reacquisition point.
    history_by_camera: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=64)))


class GlobalIdentityManager:
    def __init__(self, embeddings, gallery_store: IdentityGalleryStore | None = None):
        self.embeddings = embeddings
        self.gallery_store = gallery_store
        self.identities = {}
        self.bindings = {}
        self.binding_last_frame = {}
        self.parent = {}
        self.next_number = 1
        self.events = []
        self.latencies_us = []
        self.current_frame_members = defaultdict(list)
        self.pair_support = {}
        self.pair_conflicts = defaultdict(int)
        self.reassignment_attempts = 0
        self.reassignment_rejected = 0
        self.reassignment_accepted = 0
        self.conflict_guard_events = 0
        self.attempted_cross_camera_matches = 0
        self.accepted_cross_camera_matches = 0
        self.rejected_cross_camera_matches = 0
        self.cross_camera_world_distances = []
        self.cross_camera_time_differences_ms = []
        self.cross_camera_similarities = []
        self.persistent_reacquisition_similarities = []
        self.decision_traces = []
        self.persistence_stats = defaultdict(int)
        self._persistent_match_ambiguous = False
        self.application_ids = {}
        self._load_persistent_galleries()

    def _load_persistent_galleries(self):
        if self.gallery_store is None:
            return
        for record in self.gallery_store.identities():
            gid = int(record["canonical_id"])
            self.parent[gid] = gid
            ident = Identity(gid, int(record.get("created_frame", -1)))
            if record.get("last_world") is not None:
                ident.persisted_anchor = {
                    "world": record["last_world"],
                    "timestamp": record.get("last_source_timestamp"),
                    "camera_id": record.get("last_camera_id"),
                }
            for entry in self.gallery_store.gallery(gid):
                ident.galleries[entry["camera_id"]].append(
                    GalleryEntry(entry["vector"], float(entry["quality"]), int(entry["frame"]))
                )
                ident.gallery_quality[entry["camera_id"]].append(float(entry["quality"]))
                ident.samples += 1
            self.identities[gid] = ident
            self.application_ids[gid] = str(record["application_id"])
            self.persistence_stats["identities_loaded"] += 1
            self.persistence_stats["gallery_entries_loaded"] += ident.samples
        if self.identities:
            self.next_number = max(self.identities) + 1

    def application_id(self, gid):
        root = self.root(gid)
        return self.application_ids.get(root, f"Person_{root:02d}")

    def root(self, gid):
        while self.parent.get(gid, gid) != gid:
            self.parent[gid] = self.parent[self.parent[gid]]
            gid = self.parent[gid]
        return gid

    def identity(self, gid):
        return self.identities[self.root(gid)]

    def create(self, frame, reason):
        gid = self.next_number
        self.next_number += 1
        self.parent[gid] = gid
        self.identities[gid] = Identity(gid, frame)
        self.application_ids[gid] = f"Person_{gid:02d}"
        if self.gallery_store is not None:
            self.gallery_store.ensure_identity(gid, self.application_ids[gid], frame)
            self.persistence_stats["identity_writes"] += 1
        self.events.append({"event": "CREATE", "frame": frame, "global_person_id": gid, "reason": reason})
        return gid

    def appearance(self, vector, ident):
        values = []
        per_camera = {}
        for camera, gallery in ident.galleries.items():
            sims = sorted((float(vector @ (item.vector if isinstance(item, GalleryEntry) else item)) for item in gallery), reverse=True)
            if sims:
                take = sims[: min(3, len(sims))]
                per_camera[camera] = float(statistics.fmean(take))
                values.extend(take)
        if not values:
            return None, per_camera
        values.sort(reverse=True)
        return float(statistics.fmean(values[: min(4, len(values))])), per_camera

    def identity_similarity(self, left, right):
        left_vectors = [v.vector if isinstance(v, GalleryEntry) else v for gallery in left.galleries.values() for v in gallery]
        right_vectors = [v.vector if isinstance(v, GalleryEntry) else v for gallery in right.galleries.values() for v in gallery]
        if not left_vectors or not right_vectors:
            return None
        a = np.mean(np.stack(left_vectors), axis=0)
        b = np.mean(np.stack(right_vectors), axis=0)
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        b = b / max(float(np.linalg.norm(b)), 1e-9)
        return float(a @ b)

    def hard_possible(self, ident, obs, frame_assignments, vector=None, allow_long_gap=False):
        cam, world = obs["camera_id"], obs["world"]
        appearance, _ = self.appearance(vector, ident) if vector is not None else (None, {})
        same_camera_fragment = False
        # ReID/gallery resolution is asynchronous; the current frame's active
        # assignment list can be empty even though this identity was observed
        # from another native target in this same camera only milliseconds
        # earlier. Honor that retained observation exactly like a simultaneous
        # entry in frame_assignments, or a visually similar different person
        # can steal the canonical ID and create a map-position jump.
        last_same_camera = ident.last_by_camera.get(cam)
        if (
            last_same_camera is not None
            and last_same_camera.get("native_track_id") != obs["native_track_id"]
        ):
            same_camera_gap = int(obs["frame"]) - int(last_same_camera.get("frame", -1))
            same_camera_dt = observation_time_ms(obs, last_same_camera)
            last_distance = distance(world, last_same_camera.get("world"))
            recently_co_visible = (
                0 <= same_camera_gap <= IDENTITY_GATES["same_camera_direct_gap_frames"]
            )
            image_overlap = bbox_iou(obs.get("bbox"), last_same_camera.get("bbox"))
            retained_fragment = (
                recently_co_visible
                and vector is not None
                and appearance is not None
                and appearance >= IDENTITY_GATES["cross_camera_similarity_min"]
                and last_distance is not None
                and last_distance <= IDENTITY_GATES["cross_camera_world_tight_m"]
                and image_overlap >= 0.20
            )
            image_supported_motion_fragment = (
                recently_co_visible
                and vector is not None
                and appearance is not None
                and appearance >= IDENTITY_GATES["cross_camera_similarity_min"]
                and last_distance is not None
                and last_distance <= (
                    IDENTITY_GATES["same_camera_direct_distance_m"]
                    + IDENTITY_GATES["same_camera_speed_m_per_frame"] * same_camera_gap
                )
                and image_continuity(
                    last_same_camera.get("bbox"),
                    obs.get("bbox"),
                    same_camera_gap,
                )
            )
            if recently_co_visible and not (retained_fragment or image_supported_motion_fragment):
                # A historical embedding/world sample must not pull a second,
                # spatially distinct same-camera target into an identity whose
                # native track was still observed within the local track-break
                # window. This was observed live: a target ~11 m from the
                # retained native track reused an older 1.9 m historical
                # sample and produced duplicate simultaneous Person_XX tracks.
                return False, "same_camera_simultaneous", False
            if retained_fragment or image_supported_motion_fragment:
                same_camera_fragment = True
            elif same_camera_dt is not None and same_camera_dt <= IDENTITY_GATES["cross_camera_time_ms"]:
                return False, "same_camera_simultaneous", False
        for other in frame_assignments:
            if self.root(other["global_person_id_num"]) != ident.number:
                continue
            if other["camera_id"] == cam and other["native_track_id"] != obs["native_track_id"]:
                # A tracker fragment can coexist briefly with its predecessor.
                # Treat it as the same track only when all existing evidence
                # agrees: tight world proximity, a usable appearance match,
                # overlapping image support, and source-time compatibility.
                fragment = (
                    vector is not None
                    and appearance is not None
                    and appearance >= IDENTITY_GATES["cross_camera_similarity_min"]
                    and distance(world, other.get("world")) is not None
                    and distance(world, other.get("world")) <= IDENTITY_GATES["cross_camera_world_tight_m"]
                    and bbox_iou(obs.get("bbox"), other.get("bbox")) >= 0.20
                    and observation_time_ms(obs, other) is not None
                    and observation_time_ms(obs, other) <= IDENTITY_GATES["cross_camera_time_ms"]
                )
                if fragment:
                    same_camera_fragment = True
                    continue
                return False, "same_camera_simultaneous", False
            d = distance(world, other["world"])
            if (other["camera_id"] != cam and d is not None
                    and d > IDENTITY_GATES["cross_camera_world_candidate_m"]
                    and not same_camera_fragment):
                return False, "cross_camera_spatial_impossibility", False
        # `frame_assignments` can be empty when a camera's observation is
        # resolved asynchronously. In that case, still honor a contemporaneous
        # observation retained in this identity's other-camera latest state;
        # otherwise historical same-camera ReID can override an active
        # cross-camera spatial impossibility and merge two people.
        for other_camera, other in ident.last_by_camera.items():
            if other_camera == cam:
                continue
            cross_d = distance(world, other.get("world"))
            cross_dt = observation_time_ms(obs, other)
            if (
                cross_d is not None
                and cross_d > IDENTITY_GATES["cross_camera_world_candidate_m"]
                and cross_dt is not None
                and cross_dt <= IDENTITY_GATES["cross_camera_time_ms"]
                and not same_camera_fragment
            ):
                return False, "cross_camera_spatial_impossibility", False
        last = ident.last_by_camera.get(cam)
        active_cross_compatible = False
        for other in frame_assignments:
            if self.root(other["global_person_id_num"]) != ident.number or other["camera_id"] == cam:
                continue
            cross_d = distance(world, other.get("world"))
            cross_dt = observation_time_ms(obs, other)
            if (cross_d is not None and cross_dt is not None
                    and cross_d <= IDENTITY_GATES["cross_camera_world_candidate_m"]
                    and cross_dt <= IDENTITY_GATES["cross_camera_time_ms"]):
                active_cross_compatible = True
                break
        if not active_cross_compatible:
            for other_camera, other in ident.last_by_camera.items():
                if other_camera == cam:
                    continue
                cross_d = distance(world, other.get("world"))
                cross_dt = observation_time_ms(obs, other)
                if (cross_d is not None and cross_dt is not None
                        and cross_d <= IDENTITY_GATES["cross_camera_world_candidate_m"]
                        and cross_dt <= IDENTITY_GATES["cross_camera_time_ms"]):
                    active_cross_compatible = True
                    break
        if last is not None and not active_cross_compatible and not allow_long_gap:
            gap = obs["frame"] - last["frame"]
            d = distance(world, last["world"])
            if 0 <= gap <= 80 and d is not None:
                allowed = 2.0 + 0.20 * gap
                if d > min(8.0, allowed) and not image_continuity(last.get("bbox"), obs["bbox"], gap):
                    # An inactive/stale native fragment may have overwritten
                    # last_by_camera while an older compatible point remains.
                    # That is uncertain evidence, not proof of a new person.
                    if not self._historical_same_camera_candidate(ident, obs):
                        return False, "same_camera_temporal_speed_impossibility", False
        return True, "", same_camera_fragment

    def _historical_same_camera_candidate(self, ident, obs):
        history = ident.history_by_camera.get(obs["camera_id"], ())
        current_frame = int(obs["frame"])
        for previous in reversed(history):
            if int(previous.get("frame", -1)) >= current_frame:
                continue
            d = distance(obs.get("world"), previous.get("world"))
            if d is not None and d <= IDENTITY_GATES["same_camera_direct_distance_m"]:
                return previous
        return None

    def evidence(self, ident, obs, vector, frame_assignments):
        possible, rejection, same_camera_fragment = self.hard_possible(ident, obs, frame_assignments, vector)
        if not possible:
            return None, {"rejected": rejection}
        app, per_camera = self.appearance(vector, ident) if vector is not None else (None, {})
        same_camera_fragment_distance = None
        if same_camera_fragment:
            distances = [
                distance(obs.get("world"), other.get("world"))
                for other in frame_assignments
                if other["camera_id"] == obs["camera_id"]
                and other["native_track_id"] != obs["native_track_id"]
                and self.root(other["global_person_id_num"]) == ident.number
            ]
            same_camera_fragment_distance = min((value for value in distances if value is not None), default=None)
        native_shared = any(key[1] == obs["native_track_id"] and key[0] != obs["camera_id"] for key in ident.native_keys)
        same_frame_dist = None
        same_frame_time = None
        cross_camera_candidate = False
        for other in frame_assignments:
            if self.root(other.get("global_person_id_num")) != ident.number:
                continue
            if other["camera_id"] == obs["camera_id"]:
                continue
            cross_camera_candidate = True
            same_frame_dist = distance(obs["world"], other["world"])
            same_frame_time = observation_time_ms(obs, other)
            break

        recent_cross_dist = None
        recent_cross_time = None
        for camera, last in ident.last_by_camera.items():
            if camera == obs["camera_id"]:
                continue
            d = distance(obs["world"], last.get("world"))
            dt = observation_time_ms(obs, last)
            if d is None or dt is None:
                continue
            if recent_cross_time is None or dt < recent_cross_time:
                recent_cross_dist, recent_cross_time = d, dt
        if recent_cross_dist is not None:
            cross_camera_candidate = True

        last = ident.last_by_camera.get(obs["camera_id"])
        same_camera_continuity = None
        if last is not None:
            gap = int(obs["frame"]) - int(last["frame"])
            d = distance(obs["world"], last["world"])
            allowed = IDENTITY_GATES["same_camera_direct_distance_m"] + IDENTITY_GATES["same_camera_speed_m_per_frame"] * max(gap, 0)
            if (0 <= gap <= IDENTITY_GATES["same_camera_direct_gap_frames"]
                    and d is not None and d <= allowed
                    and (image_continuity(last.get("bbox"), obs.get("bbox"), gap) or d <= 2.0)):
                same_camera_continuity = (gap, d)

        accepted = False
        score = -1.0
        reason = "insufficient_evidence"
        top_edge_handoff = False
        if last is not None:
            gap = int(obs["frame"]) - int(last["frame"])
            last_box = last.get("bbox")
            current_box = obs.get("bbox")
            top_edge_handoff = (
                0 <= gap <= IDENTITY_GATES["same_camera_direct_gap_frames"]
                and last_box is not None and current_box is not None
                and last_box[1] <= 3.0 and current_box[1] <= 3.0
                and image_continuity(last_box, current_box, gap)
            )
        if same_camera_fragment and app is not None and same_camera_fragment_distance is not None:
            candidate = 9.0 + app - min(same_camera_fragment_distance, IDENTITY_GATES["cross_camera_world_tight_m"])
            accepted, score, reason = True, candidate, "same_camera_fragment_handoff"
        if top_edge_handoff:
            accepted, score, reason = True, 6.5, "same_camera_top_edge_handoff"
        if same_camera_continuity is not None:
            gap, d = same_camera_continuity
            candidate = 8.0 - min(d, 6.0) - gap / 40.0
            if candidate > score:
                accepted, score, reason = True, candidate, "same_camera_direct_continuity"
        if last is not None:
            gap = int(obs["frame"]) - int(last["frame"])
            d = distance(obs.get("world"), last.get("world"))
            last_box, current_box = last.get("bbox"), obs.get("bbox")
            dt = observation_time_ms(obs, last)
            if (IDENTITY_GATES["same_camera_direct_gap_frames"] < gap
                    <= IDENTITY_GATES["same_camera_strong_image_gap_frames"]
                    and d is not None and d <= IDENTITY_GATES["same_camera_strong_image_distance_m"]
                    and last_box is not None and current_box is not None
                    and bbox_iou(last_box, current_box) >= IDENTITY_GATES["same_camera_strong_image_iou"]
                    and dt is not None and dt <= 2000.0):
                candidate = 7.5 - d - gap / 80.0
                if candidate > score:
                    accepted, score, reason = True, candidate, "same_camera_strong_image_handoff"
            if (IDENTITY_GATES["same_camera_strong_image_gap_frames"] < gap
                    <= IDENTITY_GATES["same_camera_supported_gap_frames"]
                    and d is not None and d <= IDENTITY_GATES["same_camera_supported_distance_m"]
                    and last_box is not None and current_box is not None
                    and bbox_iou(last_box, current_box) >= IDENTITY_GATES["same_camera_supported_iou"]
                    and dt is not None and dt <= IDENTITY_GATES["same_camera_supported_time_ms"]
                    and app is not None and app >= IDENTITY_GATES["same_camera_supported_similarity_min"]
                    and int(obs.get("pending_good_crops", 0)) >= IDENTITY_GATES["same_camera_supported_min_crops"]):
                candidate = 4.0 + app - d / 10.0 - dt / 5000.0
                if candidate > score:
                    accepted, score, reason = True, candidate, "same_camera_supported_image_reid_handoff"
        cross_dist = same_frame_dist if same_frame_dist is not None else recent_cross_dist
        cross_time = same_frame_time if same_frame_time is not None else recent_cross_time
        if (native_shared and cross_dist is not None and cross_time is not None
                and cross_time <= IDENTITY_GATES["cross_camera_time_ms"]
                and cross_dist <= IDENTITY_GATES["cross_camera_world_candidate_m"]):
            candidate = 7.5 - cross_dist / 10.0 + (app or 0.0)
            if candidate > score:
                accepted, score, reason = True, candidate, "native_mv3dt_plus_world"
        if cross_dist is not None and cross_time is not None and cross_time <= IDENTITY_GATES["cross_camera_time_ms"]:
            if cross_dist <= IDENTITY_GATES["cross_camera_world_tight_m"]:
                candidate = 7.0 - cross_dist + (app or 0.0)
                if candidate > score:
                    accepted, score, reason = True, candidate, "cross_camera_geometry_time"
            elif cross_dist <= IDENTITY_GATES["cross_camera_world_candidate_m"]:
                if app is not None and app >= IDENTITY_GATES["cross_camera_similarity_min"]:
                    candidate = 5.0 + app - cross_dist / 20.0
                    if candidate > score:
                        accepted, score, reason = True, candidate, "cross_camera_reid_geometry"
                elif vector is None and not accepted:
                    # The wide candidate radius is useful for ReID-supported
                    # matching, but it is not a safe identity confirmation by
                    # itself. In the live Dev Room run a crop-less target was
                    # 8.01 m from an active peer only 38 ms apart and was
                    # incorrectly merged, blocking the actual 1.38 m match.
                    # Keep crop-less geometry-only adoption to the tight
                    # calibrated region; wider candidates remain pending until
                    # ReID or native MV3DT continuity supplies independent
                    # evidence. This does not affect persistent/long-gap rules.
                    reason = "geometry_only_requires_reid_or_native_evidence"
        if same_camera_continuity is None and last is not None and vector is not None and app is not None:
            gap = int(obs["frame"]) - int(last["frame"])
            d = distance(obs["world"], last["world"])
            if 0 <= gap <= 40 and d is not None and d <= 5.0 and app >= 0.70:
                accepted, score, reason = True, 3.5 + app - d / 10.0, "strong_reid_plus_same_camera_motion"
            elif (d is not None and d <= IDENTITY_GATES["same_camera_direct_distance_m"]
                  and app >= IDENTITY_GATES["cross_camera_similarity_min"]
                  and bbox_iou(last.get("bbox"), obs.get("bbox")) >= 0.20):
                candidate = 4.5 + app - d / 10.0
                if candidate > score:
                    accepted, score, reason = True, candidate, "strong_reid_same_camera_reacquisition"
        historical = self._historical_same_camera_candidate(ident, obs)
        if historical is not None and vector is not None and app is not None and app >= IDENTITY_GATES["cross_camera_similarity_min"]:
            historical_d = distance(obs.get("world"), historical.get("world"))
            candidate = 5.0 + app - (historical_d or 0.0) / 10.0
            if candidate > score:
                accepted, score, reason = True, candidate, "historical_same_camera_reid"
        if vector is not None and app is not None and ident.samples >= 2:
            if app >= IDENTITY_GATES["gallery_only_similarity_min"]:
                candidate = 2.0 + app
                if candidate > score:
                    accepted, score, reason = True, candidate, "rolling_reid"
            elif not accepted and app >= IDENTITY_GATES["cross_camera_similarity_min"]:
                reason = "appearance_below_gallery_only_gate"
        evidence = {
            "appearance_similarity": app,
            "per_camera_similarity": per_camera,
            "same_frame_world_distance": same_frame_dist,
            "same_frame_time_difference_ms": same_frame_time,
            "recent_cross_camera_world_distance": recent_cross_dist,
            "recent_cross_camera_time_difference_ms": recent_cross_time,
            "native_mv3dt_shared": native_shared,
            "same_camera_fragment": same_camera_fragment,
            "same_camera_fragment_world_distance": same_camera_fragment_distance,
            "historical_same_camera_world_distance": distance(obs.get("world"), historical.get("world")) if historical else None,
            "historical_same_camera_frame": historical.get("frame") if historical else None,
            "cross_camera_candidate": cross_camera_candidate,
            "reason": reason,
        }
        return (score if accepted else None), evidence

    def merge(self, keep, remove, frame, evidence):
        keep, remove = self.root(keep), self.root(remove)
        if keep == remove:
            return keep
        if self.identities[remove].created_frame < self.identities[keep].created_frame:
            keep, remove = remove, keep
        a, b = self.identities[keep], self.identities[remove]
        for camera, gallery in b.galleries.items():
            a.galleries[camera].extend(gallery)
            a.galleries[camera] = a.galleries[camera][-12:]
            a.gallery_quality[camera].extend(b.gallery_quality.get(camera, []))
            a.gallery_quality[camera] = a.gallery_quality[camera][-12:]
        for camera, last in b.last_by_camera.items():
            if camera not in a.last_by_camera or last["frame"] > a.last_by_camera[camera]["frame"]:
                a.last_by_camera[camera] = last
        for camera, history in b.history_by_camera.items():
            a.history_by_camera[camera].extend(history)
        a.native_keys.update(b.native_keys)
        a.samples += b.samples
        self.application_ids[keep] = self.application_ids.get(keep, f"Person_{keep:02d}")
        self.application_ids.pop(remove, None)
        if self.gallery_store is not None:
            self.gallery_store.merge_identity(keep, remove)
            self.persistence_stats["identity_merges"] += 1
        self.parent[remove] = keep
        for key, gid in list(self.bindings.items()):
            if self.root(gid) == keep or gid == remove:
                self.bindings[key] = keep
        self.events.append({
            "event": "MERGE",
            "frame": frame,
            "kept_global_person_id": keep,
            "removed_global_person_id": remove,
            "evidence": evidence,
        })
        return keep

    def pair_key(self, left, right):
        a, b = self.root(left), self.root(right)
        return tuple(sorted((a, b)))

    def record_pair_evidence(self, current_gid, obs, frame_assignments):
        for other in frame_assignments:
            other_gid = self.root(other["global_person_id_num"])
            current_gid = self.root(current_gid)
            if other_gid == current_gid:
                continue
            key = self.pair_key(current_gid, other_gid)
            d = distance(obs["world"], other["world"])
            if other["camera_id"] == obs["camera_id"]:
                if bbox_iou(obs["bbox"], other["bbox"]) < 0.20 and (d is None or d > 1.0):
                    self.pair_conflicts[key] += 1
            elif d is not None and d <= 3.0:
                state = self.pair_support.get(key, {"count": 0, "last_frame": -1000, "min_distance": 99.0})
                if obs["frame"] - state["last_frame"] > 5:
                    state["count"] = max(0, state["count"] / 2)
                state["count"] += 1
                state["last_frame"] = obs["frame"]
                state["min_distance"] = min(state["min_distance"], d)
                self.pair_support[key] = state

    def unbind_track(self, camera_id, native_track_id):
        key = (camera_id, native_track_id)
        self.bindings.pop(key, None)
        self.binding_last_frame.pop(key, None)

    def add_embedding(self, ident, obs, vector):
        if vector is None or obs.get("crop_quality_usable") is False:
            return False
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8 or not np.isfinite(vector).all():
            return False
        vector = vector / norm
        camera = obs["camera_id"]
        gallery = ident.galleries[camera]
        qualities = ident.gallery_quality[camera]
        quality = _quality_score(obs)
        if quality <= 0.0:
            return False
        for existing in gallery:
            existing_vector = existing.vector if isinstance(existing, GalleryEntry) else existing
            if float(vector @ existing_vector) >= 0.995:
                return False
        entry = GalleryEntry(vector, quality, int(obs["frame"]))
        if len(gallery) < 12:
            gallery.append(entry)
            qualities.append(quality)
        else:
            index = int(np.argmin(np.asarray(qualities, dtype=np.float32)))
            if quality <= qualities[index]:
                return False
            gallery[index] = entry
            qualities[index] = quality
        if self.gallery_store is not None:
            result = self.gallery_store.add_embedding(
                ident.number,
                obs["camera_id"],
                int(obs["frame"]),
                quality,
                vector,
                obs.get("timestamp"),
            )
            self.persistence_stats[f"gallery_{result}"] += 1
        return True

    def _finalize_observation(self, chosen, obs, vector, diagnostics, tic):
        chosen = self.root(chosen)
        ident = self.identities[chosen]
        key = (obs["camera_id"], obs["native_track_id"])
        ident.native_keys.add(key)
        if self.add_embedding(ident, obs, vector):
            ident.samples += 1
        ident.last_by_camera[obs["camera_id"]] = {
            "frame": obs["frame"],
            "world": obs["world"],
            "native_track_id": obs["native_track_id"],
            "bbox": obs["bbox"],
            "timestamp": obs.get("timestamp"),
        }
        if (self.gallery_store is not None and obs.get("world") is not None
                and (ident.last_persisted_frame < 0
                     or int(obs["frame"]) - ident.last_persisted_frame >= 40)):
            self.gallery_store.update_last_observation(
                chosen, obs["camera_id"], obs["world"], obs.get("timestamp")
            )
            ident.persisted_anchor = {
                "world": obs["world"], "timestamp": obs.get("timestamp"),
                "camera_id": obs["camera_id"],
            }
            ident.last_persisted_frame = int(obs["frame"])
            self.persistence_stats["continuity_anchor_writes"] += 1
        history = ident.history_by_camera[obs["camera_id"]]
        history_item = {
            "frame": obs["frame"],
            "world": obs["world"],
            "native_track_id": obs["native_track_id"],
            "bbox": obs["bbox"],
            "timestamp": obs.get("timestamp"),
        }
        # Retain native-fragment transitions, not every dense frame.  This
        # keeps the bounded history useful across long live runs.
        if not history or history[-1].get("native_track_id") != obs["native_track_id"]:
            history.append(history_item)
        else:
            history[-1] = history_item
        self.bindings[key] = chosen
        self.binding_last_frame[key] = obs["frame"]
        elapsed_us = (time.perf_counter_ns() - tic) / 1000.0
        self.latencies_us.append(elapsed_us)
        return chosen, diagnostics, elapsed_us

    def _persistent_reacquisition_scores(self, obs, vector, frame_assignments):
        """Score durable-gallery candidates without stale motion rejection.

        A long absence is not negative identity evidence. Current active
        assignments still pass through ``hard_possible`` so a persistent
        gallery cannot merge incompatible simultaneous people.
        """
        if vector is None:
            return []
        self._persistent_match_ambiguous = False
        candidates = []
        for gid, ident in self.identities.items():
            if self.root(gid) != gid:
                continue
            appearance, per_camera = self.appearance(vector, ident)
            anchor = ident.persisted_anchor
            anchor_age_ms = (observation_time_ms(obs, anchor)
                             if anchor and obs.get("timestamp") and anchor.get("timestamp") else None)
            anchor_distance = distance(obs.get("world"), anchor.get("world")) if anchor else None
            # Two independent usable crops plus a recently confirmed room
            # position can support a weaker viewpoint-shifted gallery match.
            # Appearance-only matches retain their stronger validated gate.
            other_appearances = []
            for other_gid, other in self.identities.items():
                if self.root(other_gid) != other_gid or other_gid == gid:
                    continue
                other_appearance, other_per_camera = self.appearance(vector, other)
                if other_appearance is None:
                    continue
                other_anchor = other.persisted_anchor
                other_anchor_age_ms = (
                    observation_time_ms(obs, other_anchor)
                    if other_anchor and obs.get("timestamp") and other_anchor.get("timestamp")
                    else None
                )
                other_anchor_distance = (
                    distance(obs.get("world"), other_anchor.get("world"))
                    if other_anchor else None
                )
                other_anchor_supported = (
                    other_appearance >= IDENTITY_GATES["recent_anchor_similarity_min"]
                    and int(obs.get("pending_good_crops", 0)) >= 2
                    and other_anchor_age_ms is not None
                    and other_anchor_age_ms <= IDENTITY_GATES["recent_anchor_max_age_ms"]
                    and other_anchor_distance is not None
                    and other_anchor_distance <= IDENTITY_GATES["cross_camera_world_candidate_m"]
                )
                other_spatial_anchor_supported = (
                    other_anchor_age_ms is not None
                    and other_anchor_age_ms <= IDENTITY_GATES["recent_anchor_max_age_ms"]
                    and other_anchor_distance is not None
                    and other_anchor_distance <= IDENTITY_GATES["cross_camera_world_candidate_m"]
                )
                other_gallery_supported = (
                    other_appearance >= IDENTITY_GATES["gallery_only_similarity_min"]
                    and (
                        other_per_camera.get(obs["camera_id"], 0.0)
                        >= IDENTITY_GATES["gallery_only_similarity_min"]
                        or other_spatial_anchor_supported
                    )
                )
                # Ambiguity margins must compare identities that could
                # actually be selected. A low-similarity or spatially stale
                # gallery is not a viable competitor and must not veto a
                # candidate backed by a recent world anchor plus independent
                # crops. It remains in diagnostics and can never be selected
                # by this filtering step.
                if not (other_anchor_supported or other_gallery_supported):
                    continue
                if not self.hard_possible(
                    other, obs, frame_assignments, vector, allow_long_gap=True
                )[0]:
                    continue
                other_appearances.append(float(other_appearance))
            other_best = max((value for value in other_appearances if value is not None), default=None)
            anchor_margin = (appearance - other_best if appearance is not None and other_best is not None
                             else None)
            anchor_supported = (appearance is not None
                and appearance >= IDENTITY_GATES["recent_anchor_similarity_min"]
                and int(obs.get("pending_good_crops", 0)) >= 2
                and anchor_age_ms is not None
                and anchor_age_ms <= IDENTITY_GATES["recent_anchor_max_age_ms"]
                and anchor_distance is not None
                and anchor_distance <= IDENTITY_GATES["cross_camera_world_candidate_m"])
            if appearance is None:
                continue
            possible, rejection, _ = self.hard_possible(
                ident, obs, frame_assignments, vector, allow_long_gap=True
            )
            if not possible:
                self.persistence_stats[f"reacquisition_rejected_{rejection}"] += 1
                continue
            # A gallery match cannot bridge an impossible displacement while
            # this identity was seen recently in either camera. The same
            # existing world candidate and per-frame speed gates bound it;
            # after a true long absence the bound naturally permits re-entry.
            recent = max(ident.last_by_camera.values(), key=lambda item: int(item["frame"]), default=None)
            if recent is not None:
                frame_gap = int(obs["frame"]) - int(recent["frame"])
                recent_distance = distance(obs.get("world"), recent.get("world"))
                allowed_distance = (
                    IDENTITY_GATES["cross_camera_world_candidate_m"]
                    + IDENTITY_GATES["same_camera_speed_m_per_frame"] * max(frame_gap, 0)
                )
                if frame_gap < 0 or (recent_distance is not None and recent_distance > allowed_distance):
                    self.persistence_stats["reacquisition_rejected_recent_world_motion"] += 1
                    continue
            last = ident.last_by_camera.get(obs["camera_id"])
            timestamp_gap_seconds = None
            if last is not None:
                timestamp_gap_seconds = observation_time_ms(obs, last)
                if timestamp_gap_seconds is not None:
                    timestamp_gap_seconds /= 1000.0
            frame_gap = int(obs["frame"]) - int(last["frame"]) if last is not None else None
            long_gap = (
                last is None
                or (frame_gap is not None and frame_gap > IDENTITY_GATES["same_camera_direct_gap_frames"])
                or (timestamp_gap_seconds is not None and timestamp_gap_seconds > 1.0)
            )
            if not long_gap:
                continue
            # A stale gallery with no plausible persisted position is not
            # sufficient evidence when only an aggregate cross-camera score
            # clears the gate. The empty-room CAM-04 false positive scored
            # 0.7056 globally but only 0.6721 against CAM-04 and 0.6660
            # against CAM-01; its anchor was 2.54 hours old and 16.8 m away.
            # Keep the numeric threshold unchanged, but require same-camera
            # gallery support for this gallery-only path. Recent, spatially
            # supported reacquisition and live geometry/MV3DT paths are
            # unaffected.
            camera_appearance = per_camera.get(obs["camera_id"])
            spatial_anchor_supported = (
                anchor is not None
                and anchor_age_ms is not None
                and anchor_age_ms <= IDENTITY_GATES["recent_anchor_max_age_ms"]
                and anchor_distance is not None
                and anchor_distance <= IDENTITY_GATES["cross_camera_world_candidate_m"]
            )
            gallery_supported = (
                appearance >= IDENTITY_GATES["gallery_only_similarity_min"]
                and (
                    (camera_appearance is not None
                     and camera_appearance >= IDENTITY_GATES["gallery_only_similarity_min"])
                    or spatial_anchor_supported
                )
            )
            if not gallery_supported and not anchor_supported:
                self.persistence_stats["reacquisition_rejected_gallery_only_similarity"] += 1
                if (appearance >= IDENTITY_GATES["gallery_only_similarity_min"]
                        and (camera_appearance is None
                             or camera_appearance < IDENTITY_GATES["gallery_only_similarity_min"])):
                    self.persistence_stats["reacquisition_rejected_gallery_camera_support"] += 1
                continue
            score = (8.0 if anchor_supported and appearance < IDENTITY_GATES["gallery_only_similarity_min"]
                     else 10.0) + float(appearance)
            self.persistent_reacquisition_similarities.append(float(appearance))
            evidence = {
                "reason": ("persistent_recent_world_reacquisition"
                           if anchor_supported and appearance < IDENTITY_GATES["gallery_only_similarity_min"]
                           else "persistent_gallery_reacquisition"),
                "persisted_anchor_world_distance_m": anchor_distance,
                "persisted_anchor_age_ms": anchor_age_ms,
                "independent_good_crops": int(obs.get("pending_good_crops", 0)),
                "similarity_margin_to_next_gallery": anchor_margin,
                "appearance_similarity": float(appearance),
                "per_camera_similarity": per_camera,
                "current_camera_gallery_similarity": camera_appearance,
                "persistent_gallery": True,
                "timestamp_gap_seconds": timestamp_gap_seconds,
                "world_distance_to_last": distance(obs.get("world"), last.get("world")) if last else None,
                "active_visibility_compatible": True,
                "candidate_score": score,
            }
            candidates.append((score, gid, evidence))
        candidates.sort(reverse=True)
        if len(candidates) > 1:
            margin = candidates[0][0] - candidates[1][0]
            if margin < IDENTITY_GATES["persistent_reacquisition_similarity_margin"]:
                self.persistence_stats["reacquisition_ambiguous"] += 1
                self._persistent_match_ambiguous = True
                return []
        return candidates

    def _candidate_scores(self, obs, vector, frame_assignments, allow_long_gap_reacquisition=False):
        candidates = []
        for gid, ident in self.identities.items():
            if self.root(gid) != gid:
                continue
            score, evidence = self.evidence(ident, obs, vector, frame_assignments)
            if evidence.get("cross_camera_candidate"):
                self.attempted_cross_camera_matches += 1
                for value, target in ((evidence.get("same_frame_world_distance"), self.cross_camera_world_distances),
                                      (evidence.get("recent_cross_camera_world_distance"), self.cross_camera_world_distances),
                                      (evidence.get("same_frame_time_difference_ms"), self.cross_camera_time_differences_ms),
                                      (evidence.get("recent_cross_camera_time_difference_ms"), self.cross_camera_time_differences_ms)):
                    if value is not None:
                        target.append(float(value))
                if evidence.get("appearance_similarity") is not None:
                    self.cross_camera_similarities.append(float(evidence["appearance_similarity"]))
            if score is not None:
                candidates.append((float(score), gid, evidence))
            elif evidence.get("rejected") in {
                "same_camera_simultaneous",
                "cross_camera_spatial_impossibility",
                "same_camera_temporal_speed_impossibility",
            }:
                self.conflict_guard_events += 1
        if allow_long_gap_reacquisition and not any(
            evidence.get("reason") in {
                "same_camera_fragment_handoff", "same_camera_top_edge_handoff",
                "same_camera_direct_continuity", "strong_reid_same_camera_reacquisition",
                "historical_same_camera_reid", "native_mv3dt_plus_world",
            }
            for _, _, evidence in candidates
        ):
            persistent = self._persistent_reacquisition_scores(obs, vector, frame_assignments)
            if self._persistent_match_ambiguous:
                return []
            for candidate in persistent:
                # The durable gallery is authoritative for a long absence;
                # replace a weaker rolling-reid score for the same identity.
                candidates = [item for item in candidates if item[1] != candidate[1]]
                candidates.append(candidate)
        return sorted(candidates, reverse=True)

    def candidate_trace(self, obs, vector, frame_assignments):
        """Complete non-mutating audit of every canonical candidate."""
        trace = []
        active_by_gid = defaultdict(list)
        for active in frame_assignments:
            gid = self.root(active["global_person_id_num"])
            active_by_gid[gid].append(active)
        for gid, ident in sorted(self.identities.items()):
            if self.root(gid) != gid:
                continue
            score, evidence = self.evidence(ident, obs, vector, frame_assignments)
            active = active_by_gid.get(gid, [])
            last_seen = max((int(item.get("frame", -1)) for item in ident.last_by_camera.values()), default=-1)
            last = ident.last_by_camera.get(obs["camera_id"])
            app, per_camera = self.appearance(vector, ident) if vector is not None else (None, {})
            gallery_entries = [entry for values in ident.galleries.values() for entry in values]
            similarities = []
            if vector is not None:
                candidate_vector = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(candidate_vector))
                if norm > 1e-8:
                    candidate_vector = candidate_vector / norm
                    similarities = sorted((float(candidate_vector @ (entry.vector if isinstance(entry, GalleryEntry) else entry)) for entry in gallery_entries), reverse=True)
            historical = self._historical_same_camera_candidate(ident, obs)
            rejection = evidence.get("rejected")
            last_same_camera = ident.last_by_camera.get(obs["camera_id"])
            retained_same_camera_conflict = (
                last_same_camera is not None
                and last_same_camera.get("native_track_id") != obs["native_track_id"]
                and 0 <= int(obs["frame"]) - int(last_same_camera.get("frame", -1))
                <= IDENTITY_GATES["same_camera_direct_gap_frames"]
            )
            same_camera_conflict = retained_same_camera_conflict or any(
                item["camera_id"] == obs["camera_id"]
                and item["native_track_id"] != obs["native_track_id"]
                for item in active
            )
            same_camera_fragment = bool(evidence.get("same_camera_fragment"))
            cross_spatial_conflict = any(
                item["camera_id"] != obs["camera_id"]
                and distance(obs.get("world"), item.get("world")) is not None
                and distance(obs.get("world"), item.get("world")) > IDENTITY_GATES["cross_camera_world_candidate_m"]
                for item in active
            )
            geometry_distances = [distance(obs.get("world"), item.get("world")) for item in active if distance(obs.get("world"), item.get("world")) is not None]
            geometry_distance = min(geometry_distances) if geometry_distances else (distance(obs.get("world"), last.get("world")) if last else None)
            geometry_gate = geometry_distance is not None and geometry_distance <= IDENTITY_GATES["cross_camera_world_candidate_m"]
            temporal_gap = (int(obs["frame"]) - int(last["frame"])) if last else None
            temporal_gate = temporal_gap is None or temporal_gap >= 0
            persisted_anchor = ident.persisted_anchor
            anchor_age_ms = (
                observation_time_ms(obs, persisted_anchor)
                if persisted_anchor and obs.get("timestamp") and persisted_anchor.get("timestamp")
                else None
            )
            anchor_distance = distance(obs.get("world"), persisted_anchor.get("world")) if persisted_anchor else None
            recent_anchor_supported = (
                app is not None
                and app >= IDENTITY_GATES["recent_anchor_similarity_min"]
                and int(obs.get("pending_good_crops", 0)) >= 2
                and anchor_age_ms is not None
                and anchor_age_ms <= IDENTITY_GATES["recent_anchor_max_age_ms"]
                and anchor_distance is not None
                and anchor_distance <= IDENTITY_GATES["cross_camera_world_candidate_m"]
            )
            camera_gallery_similarity = per_camera.get(obs["camera_id"])
            gallery_camera_similarity_supported = (
                app is not None
                and app >= IDENTITY_GATES["gallery_only_similarity_min"]
                and camera_gallery_similarity is not None
                and camera_gallery_similarity >= IDENTITY_GATES["gallery_only_similarity_min"]
            )
            spatial_anchor_supported = (
                persisted_anchor is not None
                and anchor_age_ms is not None
                and anchor_age_ms <= IDENTITY_GATES["recent_anchor_max_age_ms"]
                and anchor_distance is not None
                and anchor_distance <= IDENTITY_GATES["cross_camera_world_candidate_m"]
            )
            if (score is None and app is not None
                    and app >= IDENTITY_GATES["gallery_only_similarity_min"]
                    and not gallery_camera_similarity_supported
                    and not spatial_anchor_supported):
                rejection = "stale_gallery_missing_current_camera_support"
            trace.append({
                "global_person_id": gid,
                "application_id": f"Person_{gid:02d}",
                "active_cameras": sorted({str(item["camera_id"]) for item in active}),
                "active": bool(active),
                "current_recent_world_positions": {camera: item.get("world") for camera, item in ident.last_by_camera.items()},
                "world_distance_m": geometry_distance,
                "timestamp_difference_ms": observation_time_ms(obs, last) if last else None,
                "last_seen_age_frames": int(obs["frame"]) - last_seen if last_seen >= 0 else None,
                "gallery_size": len(gallery_entries),
                "persisted_anchor_world": ident.persisted_anchor.get("world") if ident.persisted_anchor else None,
                "persisted_anchor_timestamp": ident.persisted_anchor.get("timestamp") if ident.persisted_anchor else None,
                "persisted_anchor_distance_m": anchor_distance,
                "persisted_anchor_age_ms": anchor_age_ms,
                "gallery_camera_similarity": camera_gallery_similarity,
                "gallery_camera_supported": gallery_camera_similarity_supported,
                "spatial_anchor_supported": spatial_anchor_supported,
                "recent_anchor_supported": recent_anchor_supported,
                "crop_quality": obs.get("crop_quality"),
                "current_embedding_available": vector is not None,
                "cosine_best_gallery": similarities[0] if similarities else None,
                "cosine_gallery_mean_representatives": float(statistics.fmean(similarities[:min(4, len(similarities))])) if similarities else None,
                "cosine_per_camera": per_camera,
                "reid_gate": app is not None and app >= IDENTITY_GATES["cross_camera_similarity_min"] if app is not None else None,
                "geometry_gate": geometry_gate,
                "temporal_gate": temporal_gate,
                "active_conflict_gate": not ((same_camera_conflict and not same_camera_fragment) or cross_spatial_conflict),
                "same_camera_conflict_gate": not same_camera_conflict or same_camera_fragment,
                "cross_camera_continuity_evidence": {
                    "same_frame_world_distance": evidence.get("same_frame_world_distance"),
                    "same_frame_time_difference_ms": evidence.get("same_frame_time_difference_ms"),
                    "recent_world_distance": evidence.get("recent_cross_camera_world_distance"),
                    "recent_time_difference_ms": evidence.get("recent_cross_camera_time_difference_ms"),
                },
                "native_mva_continuity_evidence": bool(evidence.get("native_mv3dt_shared")),
                "historical_same_camera_evidence": {
                    "frame": historical.get("frame") if historical else None,
                    "world": historical.get("world") if historical else None,
                    "distance_m": evidence.get("historical_same_camera_world_distance"),
                },
                "final_candidate_score": score,
                "reid_gate_reason": "pass" if app is not None and app >= IDENTITY_GATES["cross_camera_similarity_min"] else ("no_embedding" if app is None else "below_existing_gate"),
                "geometry_gate_reason": "pass" if geometry_gate else "outside_existing_candidate_gate",
                "temporal_gate_reason": "pass" if temporal_gate else "future_observation",
                "active_conflict_reason": rejection if rejection in {"same_camera_simultaneous", "cross_camera_spatial_impossibility"} else None,
                "same_camera_conflict_reason": ("same_camera_fragment_handoff" if same_camera_fragment else "same_camera_simultaneous") if same_camera_conflict else None,
                "exact_rejection_reason": rejection or (None if score is not None else evidence.get("reason", "insufficient_evidence")),
            })
        return trace

    def novelty_evidence(self, obs, vector, frame_assignments, good_crops, attempts):
        """Positive novelty only; timeout/weak evidence never allocates."""
        candidates = self.candidate_trace(obs, vector, frame_assignments)
        if not candidates:
            return {"positive": int(good_crops) >= 2 and int(attempts) >= 2, "reason": "no_existing_identity"}
        if int(good_crops) < 2 or int(attempts) < 2:
            return {"positive": False, "reason": "insufficient_independent_crop_evidence", "candidates": candidates}
        for candidate in candidates:
            if candidate["final_candidate_score"] is not None:
                return {"positive": False, "reason": "existing_candidate_accepted", "candidates": candidates}
        conflict_reasons = {"same_camera_simultaneous", "cross_camera_spatial_impossibility"}
        active_candidates = [candidate for candidate in candidates if candidate["active"]]
        historical_active_conflict = any(
            candidate["historical_same_camera_evidence"]["frame"] is not None
            and candidate["active"]
            and candidate["exact_rejection_reason"] in conflict_reasons
            for candidate in candidates
        )
        if historical_active_conflict:
            # Old same-camera proximity must not keep a person PENDING when
            # that identity is currently active elsewhere (or simultaneously
            # in this camera) at an impossible position. Require stronger
            # independent crop evidence than ordinary novelty before creating
            # a separate canonical identity in this ambiguous case.
            if (
                active_candidates
                and all(candidate["exact_rejection_reason"] in conflict_reasons for candidate in active_candidates)
                and int(good_crops) >= 3
                and int(attempts) >= 3
            ):
                return {
                    "positive": True,
                    "reason": "active_spatial_conflict_overrides_stale_history",
                    "candidates": candidates,
                }
            return {"positive": False, "reason": "historical_continuity_unknown_not_novelty", "candidates": candidates}
        for candidate in candidates:
            if candidate["historical_same_camera_evidence"]["frame"] is not None:
                return {"positive": False, "reason": "historical_continuity_unknown_not_novelty", "candidates": candidates}
            if candidate["exact_rejection_reason"] not in conflict_reasons:
                return {"positive": False, "reason": "non_positive_or_stale_evidence", "candidates": candidates}
        return {"positive": True, "reason": "all_existing_identities_incompatible", "candidates": candidates}

    def resolve_existing(self, obs, vector, frame_assignments):
        """Return the best existing canonical identity, never allocate a new one."""
        candidates = self._candidate_scores(obs, vector, frame_assignments, allow_long_gap_reacquisition=True)
        if not candidates:
            self.rejected_cross_camera_matches += int(any(
                self.evidence(ident, obs, vector, frame_assignments)[1].get("cross_camera_candidate")
                for gid, ident in self.identities.items() if self.root(gid) == gid
            ))
            return None, {"reason": "pending_no_acceptable_candidate"}
        score, gid, evidence = candidates[0]
        if evidence.get("cross_camera_candidate"):
            self.accepted_cross_camera_matches += 1
        evidence = dict(evidence)
        evidence["candidate_score"] = score
        return self.root(gid), evidence

    def _batch_compatible(self, gid, obs, assigned):
        gid = self.root(gid)
        for other_obs, other_gid in assigned:
            if self.root(other_gid) != gid:
                continue
            if other_obs["camera_id"] == obs["camera_id"]:
                self.conflict_guard_events += 1
                return False
            d = distance(obs["world"], other_obs["world"])
            if d is not None and d > IDENTITY_GATES["cross_camera_world_candidate_m"]:
                self.conflict_guard_events += 1
                return False
        return True

    def process_batch(self, items, frame_assignments):
        """Resolve a ReID micro-batch with a one-to-one active assignment solver."""
        if not items:
            return []
        candidates = [self._candidate_scores(item["obs"], item.get("vector"), frame_assignments, allow_long_gap_reacquisition=True) for item in items]
        best_score = float("-inf")
        best_assignment = None

        def search(index, assigned, choices, score):
            nonlocal best_score, best_assignment
            if index == len(items):
                if score > best_score:
                    best_score = score
                    best_assignment = list(choices)
                return
            options = candidates[index][:8]
            for candidate_score, gid, evidence in options:
                if not self._batch_compatible(gid, items[index]["obs"], assigned):
                    continue
                search(index + 1, assigned + [(items[index]["obs"], gid)], choices + [(gid, evidence)], score + candidate_score)
            search(index + 1, assigned, choices + [(None, {"reason": "pending_no_acceptable_candidate"})], score)

        search(0, [], [], 0.0)
        result = []
        for item, (gid, evidence) in zip(items, best_assignment or []):
            tic = time.perf_counter_ns()
            if gid is None:
                result.append((None, evidence, 0.0))
                continue
            self.events.append({
                "event": "ACQUIRE",
                "frame": item["obs"]["frame"],
                "camera_id": item["obs"]["camera_id"],
                "native_track_id": item["obs"]["native_track_id"],
                "global_person_id": gid,
                "evidence": evidence,
            })
            chosen = self._finalize_observation(gid, item["obs"], item.get("vector"), evidence, tic)
            result.append((chosen[0], chosen[1], chosen[2]))
        return result

    def accept_existing(self, gid, obs, vector, evidence):
        """Commit an existing-canonical match after pending evaluation."""
        tic = time.perf_counter_ns()
        gid = self.root(gid)
        self.events.append({
            "event": "ACQUIRE",
            "frame": obs["frame"],
            "camera_id": obs["camera_id"],
            "native_track_id": obs["native_track_id"],
            "global_person_id": gid,
            "evidence": evidence,
        })
        return self._finalize_observation(gid, obs, vector, evidence, tic)[0]

    def confirm_new(self, obs, vector, reason="pending_window_exhausted"):
        tic = time.perf_counter_ns()
        gid = self.create(obs["frame"], reason)
        self.events.append({
            "event": "NEW_CONFIRMED",
            "frame": obs["frame"],
            "camera_id": obs["camera_id"],
            "native_track_id": obs["native_track_id"],
            "global_person_id": gid,
            "reason": reason,
        })
        return self._finalize_observation(gid, obs, vector, {"reason": "new_confirmed"}, tic)[0]

    def process(self, obs, vector, frame_assignments, allow_reassignment=True):
        tic = time.perf_counter_ns()
        key = (obs["camera_id"], obs["native_track_id"])
        bound = self.bindings.get(key)
        last_bound_frame = self.binding_last_frame.get(key)
        if allow_reassignment and bound is not None and last_bound_frame is not None and obs["frame"] - last_bound_frame > 30:
            bound = None
        chosen = self.root(bound) if bound is not None else None
        best = None
        diagnostics = {}

        if chosen is None:
            # Preserve a proven same-camera identity across a short native-track
            # break before consulting potentially noisy cross-view geometry.
            if vector is not None:
                for gid, ident in self.identities.items():
                    if self.root(gid) != gid:
                        continue
                    last = ident.last_by_camera.get(obs["camera_id"])
                    if last is None:
                        continue
                    gap = obs["frame"] - last["frame"]
                    d = distance(obs["world"], last["world"])
                    app, per_camera = self.appearance(vector, ident)
                    short_break = 0 <= gap <= 12 and d is not None and d <= 2.0 and app is not None and app >= 0.60
                    longer_strong_break = 0 <= gap <= 20 and d is not None and d <= 5.0 and app is not None and app >= 0.75
                    long_gap_precise_break = 0 <= gap <= 30 and d is not None and d <= 2.0 and app is not None and app >= 0.61
                    if ((short_break or longer_strong_break or long_gap_precise_break)
                            and image_continuity(last.get("bbox"), obs["bbox"], gap)):
                        evidence = {
                            "reason": "strong_reid_same_camera_track_break",
                            "appearance_similarity": app,
                            "same_camera_world_distance": d,
                            "frame_gap": gap,
                        }
                        candidate = (10.0 + app, gid, evidence)
                        if best is None or candidate[0] > best[0]:
                            best = candidate
            for gid, ident in self.identities.items():
                if self.root(gid) != gid:
                    continue
                score, evidence = self.evidence(ident, obs, vector, frame_assignments)
                if score is not None and (best is None or score > best[0]):
                    best = score, gid, evidence
            if best is None:
                chosen = self.create(obs["frame"], "unmatched_track")
                diagnostics = {"reason": "new_identity"}
            else:
                _, chosen, diagnostics = best
                self.events.append({
                    "event": "ACQUIRE",
                    "frame": obs["frame"],
                    "camera_id": obs["camera_id"],
                    "native_track_id": obs["native_track_id"],
                    "global_person_id": chosen,
                    "evidence": diagnostics,
                })
            self.bindings[key] = chosen
        else:
            current = self.identity(chosen)
            if not allow_reassignment:
                return self._finalize_observation(
                    chosen,
                    obs,
                    vector,
                    {"reason": "sticky_native_track"},
                    tic,
                )
            self.record_pair_evidence(chosen, obs, frame_assignments)
            self.reassignment_attempts += 1
            alternatives = []
            for gid, ident in self.identities.items():
                if self.root(gid) != gid or gid == current.number:
                    continue
                if vector is not None:
                    score, evidence = self.evidence(ident, obs, vector, frame_assignments)
                    if score is not None and current.samples < 2:
                        alternatives.append((score, gid, evidence))
                possible, _, _ = self.hard_possible(ident, obs, frame_assignments, vector)
                if not possible:
                    continue
                gallery_sim = self.identity_similarity(current, ident)
                last_same_camera = ident.last_by_camera.get(obs["camera_id"])
                if last_same_camera is not None:
                    gap = obs["frame"] - last_same_camera["frame"]
                    last_box = last_same_camera.get("bbox")
                    current_box = obs["bbox"]
                    no_reid_top_edge_handoff = (
                        gallery_sim is None and 0 <= gap <= 12 and last_box is not None
                        and last_box[1] <= 3.0 and current_box[1] <= 3.0
                        and image_continuity(last_box, current_box, gap)
                    )
                    strong_reid_handoff = gallery_sim is not None and gallery_sim >= 0.60 and image_continuity(last_box, current_box, gap)
                    if no_reid_top_edge_handoff or strong_reid_handoff:
                        alternatives.append((6.0 + (gallery_sim or 0.0), gid, {
                            "reason": "same_camera_causal_handoff",
                            "appearance_similarity": gallery_sim,
                            "frame_gap": gap,
                        }))
                same_frame_dist = None
                for other in frame_assignments:
                    if self.root(other["global_person_id_num"]) == ident.number and other["camera_id"] != obs["camera_id"]:
                        same_frame_dist = distance(obs["world"], other["world"])
                        break
                pair_key = self.pair_key(current.number, ident.number)
                support = self.pair_support.get(pair_key)
                conflicts = self.pair_conflicts.get(pair_key, 0)
                if (gallery_sim is not None and gallery_sim >= 0.80 and support is not None
                        and support["count"] >= 8 and obs["frame"] - support["last_frame"] <= 240
                        and conflicts == 0):
                    alternatives.append((7.0 + gallery_sim, gid, {
                        "reason": "rolling_reid_plus_remembered_world_support",
                        "appearance_similarity": gallery_sim,
                        "world_support_count": support["count"],
                        "world_support_min_distance": support["min_distance"],
                        "same_camera_conflicts": conflicts,
                    }))
                if gallery_sim is not None and same_frame_dist is not None:
                    accepted = (same_frame_dist <= 1.5 and gallery_sim >= 0.54) or (same_frame_dist <= 3.0 and gallery_sim >= 0.60)
                    if accepted:
                        evidence = {
                            "reason": "rolling_reid_plus_simultaneous_world",
                            "appearance_similarity": gallery_sim,
                            "same_frame_world_distance": same_frame_dist,
                        }
                        alternatives.append((5.0 + gallery_sim - same_frame_dist / 10.0, gid, evidence))
            if alternatives:
                score, gid, evidence = max(alternatives)
                chosen = self.merge(gid, chosen, obs["frame"], evidence)
                diagnostics = evidence
                self.reassignment_accepted += 1
            else:
                self.reassignment_rejected += 1
                self.events.append({
                    "event": "REASSIGN_REJECTED",
                    "frame": obs["frame"],
                    "camera_id": obs["camera_id"],
                    "native_track_id": obs["native_track_id"],
                    "reason": "no_strong_contradictory_evidence",
                })

        return self._finalize_observation(chosen, obs, vector, diagnostics, tic)


def load_kafka(path):
    latest = {}
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame = event.get("frame")
        if frame and frame.get("sensorId") in CAMS:
            latest[(frame["sensorId"], int(frame["id"]))] = frame
    return latest


def physical_label(obs):
    l, t, r, b = obs["bbox"]
    cx = (l + r) / 2.0
    if obs["camera_id"] == "CAM-01" and 750 <= cx <= 1000 and 350 <= t <= 550 and 580 <= b <= 760:
        return "white-shirt"
    if obs["camera_id"] == "CAM-04" and 1000 <= cx <= 1350 and 350 <= t <= 550 and 650 <= b <= 800:
        return "white-shirt"
    return "black-shirt"


def intervals(frames):
    result = []
    for frame in sorted(frames):
        if not result or frame > result[-1][1] + 1:
            result.append([frame, frame])
        else:
            result[-1][1] = frame
    return result


def pct(values, q):
    return float(np.quantile(values, q)) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kafka", required=True, type=Path)
    ap.add_argument("--embedding-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.load(args.embedding_dir / "embeddings.npy")
    embedding_records = [json.loads(x) for x in (args.embedding_dir / "embedding_records.jsonl").read_text().splitlines()]
    embedding_map = {
        (r["camera_id"], r["frame"], r["native_track_id"]): matrix[r["embedding_index"]]
        for r in embedding_records
    }
    extraction_meta = {
        (r["camera_id"], r["frame"], r["native_track_id"]): r for r in embedding_records
    }
    frames = load_kafka(args.kafka)
    manager = GlobalIdentityManager(matrix)
    output = []

    for frame_num in sorted({frame for _, frame in frames}):
        frame_assignments = []
        for camera in CAMS:
            payload = frames.get((camera, frame_num))
            if not payload:
                continue
            objects = [o for o in payload.get("objects", []) if o.get("type", "").lower() == "person"]
            objects.sort(key=lambda o: int(o["id"]))
            for obj in objects:
                obs = {
                    "camera_id": camera,
                    "frame": frame_num,
                    "timestamp": payload.get("timestamp"),
                    "native_track_id": int(obj["id"]),
                    "bbox": bbox_of(obj),
                    "world": world_of(obj),
                    "confidence": float(obj.get("confidence", 0.0)),
                    "visibility": float(obj.get("info", {}).get("visibility", 0.0)),
                }
                key = (camera, frame_num, obs["native_track_id"])
                vector = embedding_map.get(key)
                gid, evidence, latency_us = manager.process(obs, vector, frame_assignments)
                row = {
                    **obs,
                    "global_person_id_num": gid,
                    "global_person_id": f"Person_{gid:02d}",
                    "embedding_extracted": vector is not None,
                    "cosine_similarity": evidence.get("appearance_similarity"),
                    "decision_reason": evidence.get("reason", "bound_native_track"),
                    "identity_manager_latency_us": latency_us,
                }
                if key in extraction_meta:
                    row["reid"] = {
                        k: extraction_meta[key][k]
                        for k in ("crop_width", "crop_height", "per_crop_latency_ms")
                    }
                output.append(row)
                frame_assignments.append(row)

    # Canonicalize aliases after all causal merge events. A live UI applies the
    # same alias event prospectively; this pass makes duration metrics explicit.
    for row in output:
        root = manager.root(row["global_person_id_num"])
        row["canonical_global_person_id_num"] = root
        row["canonical_global_person_id"] = f"Person_{root:02d}"

    with (args.output_dir / "global_identity.jsonl").open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output_dir / "identity_events.json").write_text(json.dumps(manager.events, indent=2))

    chosen = defaultdict(dict)
    native_ids = defaultdict(set)
    app_ids = defaultdict(set)
    for row in output:
        label = physical_label(row)
        key = (label, row["camera_id"])
        old = chosen[key].get(row["frame"])
        if old is None or row["confidence"] > old["confidence"]:
            chosen[key][row["frame"]] = row
        native_ids[label].add((row["camera_id"], row["native_track_id"]))
        app_ids[label].add(row["canonical_global_person_id_num"])

    people = {}
    false_merges = []
    for frame_num in sorted({r["frame"] for r in output}):
        labels_by_gid = defaultdict(set)
        for label in ("white-shirt", "black-shirt"):
            for camera in CAMS:
                row = chosen[(label, camera)].get(frame_num)
                if row:
                    labels_by_gid[row["canonical_global_person_id_num"]].add(label)
        for gid, labels in labels_by_gid.items():
            if len(labels) > 1:
                false_merges.append({"frame": frame_num, "global_person_id": gid, "labels": sorted(labels)})

    for label in ("white-shirt", "black-shirt"):
        common = sorted(set(chosen[(label, CAMS[0])]) & set(chosen[(label, CAMS[1])]))
        same = [
            frame for frame in common
            if chosen[(label, CAMS[0])][frame]["canonical_global_person_id_num"]
            == chosen[(label, CAMS[1])][frame]["canonical_global_person_id_num"]
        ]
        confident = []
        confident_same = []
        for frame in common:
            a, b = chosen[(label, CAMS[0])][frame], chosen[(label, CAMS[1])][frame]
            d = distance(a["world"], b["world"])
            if a["visibility"] >= 0.5 and b["visibility"] >= 0.5 and d is not None and d <= 3.0:
                confident.append(frame)
                if a["canonical_global_person_id_num"] == b["canonical_global_person_id_num"]:
                    confident_same.append(frame)
        switches = []
        for camera in CAMS:
            seq = chosen[(label, camera)]
            fs = sorted(seq)
            for a, b in zip(fs, fs[1:]):
                if b == a + 1 and seq[a]["canonical_global_person_id_num"] != seq[b]["canonical_global_person_id_num"]:
                    switches.append({
                        "camera": camera,
                        "frame": b,
                        "old": seq[a]["canonical_global_person_id"],
                        "new": seq[b]["canonical_global_person_id"],
                    })
        people[label] = {
            "native_mv3dt_fragments": len(native_ids[label]),
            "native_keys": sorted([list(x) for x in native_ids[label]]),
            "final_global_person_id_fragments": len(app_ids[label]),
            "final_global_person_ids": [f"Person_{x:02d}" for x in sorted(app_ids[label])],
            "final_id_switches": len(switches),
            "switch_details": switches,
            "jointly_visible_frames": len(common),
            "jointly_visible_same_global_person_id_frames": len(same),
            "same_id_fraction_all_joint": len(same) / len(common) if common else None,
            "confidently_matchable_joint_frames": len(confident),
            "confident_same_global_person_id_frames": len(confident_same),
            "confident_same_id_fraction": len(confident_same) / len(confident) if confident else None,
            "same_id_intervals": intervals(same),
            "split_intervals": intervals(set(common) - set(same)),
        }

    latencies = manager.latencies_us
    report = {
        "algorithm": {
            "hard_coded_person_count": False,
            "face_recognition_used": False,
            "evidence_priority": [
                "spatial/time impossibility",
                "OSNet rolling-gallery similarity",
                "world/time continuity",
                "native MV3DT shared ID",
            ],
            "gallery_samples_per_camera": 12,
            "single_weak_embedding_decisions": False,
        },
        "identities_created_before_alias_resolution": manager.next_number - 1,
        "canonical_identities": len({manager.root(x) for x in manager.identities}),
        "merge_events": sum(e["event"] == "MERGE" for e in manager.events),
        "acquisition_events": sum(e["event"] == "ACQUIRE" for e in manager.events),
        "people": people,
        "false_white_black_merges": len(false_merges),
        "false_merge_details": false_merges,
        "identity_manager_latency_us": {
            "mean": statistics.fmean(latencies),
            "p95": pct(latencies, 0.95),
            "max": max(latencies),
        },
    }
    (args.output_dir / "identity_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
