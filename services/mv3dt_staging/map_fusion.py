"""Display-only aggregation of accepted provisional IDs in experimental map pixels.

Inspired by NVIDIA DeepStream v9.1.0 kafka_bev_visualizer.draw_objects_on_map
(objects_by_id / average_multi_cam). Unlike that world-coordinate visualizer,
this application has only experimental per-camera homographies. Host arrival
proximity is a display-time approximation, never proof of exposure alignment.
No identity decisions or original observations are modified here.
"""
from __future__ import annotations

from collections import Counter, deque
import copy
import hashlib
import math
import time
from typing import Any

CAMERA_SOURCES = {"CAM-02": 0, "CAM-05": 1}
ACCEPTED_STATUS = "AUTOMATIC_PROVISIONAL_SHADOW_NOT_VERIFIED"
IDENTITY_AUTHORITY = "OSNET_ACCEPTED_PROVISIONAL_BINDINGS"
ARRIVAL_DOMAIN = "HOST_MONOTONIC_ARRIVAL"
LOCAL_COLORS = {"CAM-02": [60, 191, 255], "CAM-05": [255, 190, 60]}


def empty_identity_snapshot(now_ns: int) -> dict[str, Any]:
    return {"authority": IDENTITY_AUTHORITY, "revision": 0,
            "sampled_monotonic_ns": int(now_ns),
            "assignments": {camera: {} for camera in CAMERA_SOURCES}}


class ExperimentalMapFusion:
    """Bounded, reversible render state; inputs must already have exact joins."""

    def __init__(self, *, layout_width: int, layout_height: int,
                 max_age_seconds: float = 0.75, max_arrival_skew_seconds: float = 0.15,
                 max_spread_layout_px: float = 75.0, trail_points: int = 16,
                 max_markers: int = 256):
        if (layout_width <= 0 or layout_height <= 0 or trail_points < 2 or max_markers < 1
                or not all(math.isfinite(v) and v > 0 for v in
                           (max_age_seconds, max_arrival_skew_seconds, max_spread_layout_px))):
            raise ValueError("invalid experimental map fusion bounds")
        self.width, self.height = int(layout_width), int(layout_height)
        self.max_age_ns = int(max_age_seconds * 1e9)
        self.max_skew_ns = int(max_arrival_skew_seconds * 1e9)
        self.max_spread = float(max_spread_layout_px)
        self.trail_points, self.max_markers = int(trail_points), int(max_markers)
        self._trails: dict[str, deque] = {}
        self._signatures: dict[str, tuple] = {}
        self._lineages: dict[str, tuple] = {}
        self._revision = -1
        self._mode = None

    def policy(self) -> dict[str, Any]:
        return {"coordinate_space": "EXPERIMENTAL_LAYOUT_PIXELS", "verified_world_frame": False,
                "position_method": "arithmetic mean of eligible layout-pixel positions",
                "temporal_basis": "host arrival proximity only; exposure synchronization UNVERIFIED",
                "max_observation_age_ms": self.max_age_ns / 1e6,
                "max_identity_snapshot_age_ms": self.max_age_ns / 1e6,
                "identity_expiry_guard": "explicit per-binding deadline from existing RAM gallery TTL",
                "max_arrival_skew_ms": self.max_skew_ns / 1e6,
                "max_spread_layout_px": self.max_spread,
                "spread_limit_is_physical_accuracy": False,
                "trail_points": self.trail_points, "max_markers": self.max_markers}

    def update(self, states: dict[str, dict], identity: dict[str, Any], *,
               now_mono_ns: int | None = None, mode: str = "experimental-global") -> dict[str, Any]:
        if mode not in {"local", "experimental-global"}:
            raise ValueError("unsupported map display mode")
        now = time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
        if mode != self._mode:
            self._trails.clear(); self._signatures.clear(); self._lineages.clear()
            self._mode = mode
        rejected = Counter()
        observations = []
        for camera, expected_sid in CAMERA_SOURCES.items():
            state = states.get(camera) or {}
            if not isinstance(state, dict):
                rejected["invalid_or_stale_source_snapshot"] += 1
                continue
            if state.get("status") != "LIVE":
                continue
            try:
                sid, frame, pts, arrival = (int(state[key]) for key in
                    ("source_id", "frame_index", "source_pts_ns", "updated_mono_ns"))
                if (sid != expected_sid or state.get("camera_id") != camera
                        or frame < 0 or pts < 0 or not 0 <= now - arrival <= self.max_age_ns):
                    raise ValueError("stale or invalid source snapshot")
            except (KeyError, ValueError, TypeError):
                rejected["invalid_or_stale_source_snapshot"] += 1
                continue
            rows = state.get("markers", [])
            if not isinstance(rows, list):
                rejected["invalid_observation_attribution_or_position"] += 1
                continue
            rejected["marker_capacity_drops"] += max(0, len(rows) - self.max_markers)
            malformed = sum(not isinstance(row, dict) for row in rows[:self.max_markers])
            rejected["invalid_observation_attribution_or_position"] += malformed
            rows = [row for row in rows[:self.max_markers] if isinstance(row, dict)]
            ids = Counter(str(row.get("local_tracker_id")) for row in rows)
            for row in rows:
                try:
                    local = str(row["local_tracker_id"])
                    if ids[local] != 1 or local in {"", "None", "-1"}:
                        raise ValueError("duplicate or invalid local ID")
                    if ((row["camera_id"], int(row["source_id"]), int(row["frame_index"]),
                         int(row["source_pts_ns"])) != (camera, sid, frame, pts)):
                        raise ValueError("source/frame/PTS mismatch")
                    x, y = map(float, row["layout_xy"])
                    if not (math.isfinite(x) and math.isfinite(y)
                            and 0 <= x < self.width and 0 <= y < self.height):
                        raise ValueError("invalid or out-of-layout position")
                except (KeyError, TypeError, ValueError):
                    rejected["invalid_observation_attribution_or_position"] += 1
                    continue
                observations.append({**copy.deepcopy(row), "local_tracker_id": local,
                    "layout_xy": [x, y], "observed_monotonic_ns": arrival,
                    "observation_time_domain": state.get("observation_time_domain"),
                    "local_key": f"local:{camera}:{local}"})

        mapping = {}
        revision = -1
        try:
            revision = int(identity["revision"])
            sampled = int(identity["sampled_monotonic_ns"])
            if (identity.get("authority") != IDENTITY_AUTHORITY
                    or revision < 0 or not 0 <= now - sampled <= self.max_age_ns
                    or revision < self._revision):
                raise ValueError("stale or unauthoritative identity snapshot")
            mapping = identity["assignments"]
            if not isinstance(mapping, dict):
                raise ValueError("invalid identity mapping")
            self._revision = revision
        except (KeyError, ValueError, TypeError):
            rejected["stale_or_invalid_identity_snapshot"] += 1
            mapping = {}

        groups: dict[str, list] = {}
        for obs in observations:
            camera_mapping = mapping.get(obs["camera_id"], {})
            binding = (camera_mapping.get(obs["local_tracker_id"], {})
                       if isinstance(camera_mapping, dict) else {})
            if not isinstance(binding, dict):
                binding = {}
            gid = binding.get("provisional_global_id")
            try:
                unexpired = now < int(binding["identity_expires_monotonic_ns"])
            except (KeyError, TypeError, ValueError):
                unexpired = False
            if (isinstance(gid, str) and gid and binding.get("status") == ACCEPTED_STATUS
                    and binding.get("identity_memory_state") == "ACTIVE"
                    and binding.get("persistent") is False and unexpired):
                obs["accepted_provisional_global_id"] = gid
                key = f"global:{gid}"
            else:
                key = obs["local_key"]
            groups.setdefault(key, []).append(obs)

        markers, diagnostics = [], []
        for key, members in sorted(groups.items()):
            gid = members[0].get("accepted_provisional_global_id")
            spread = max((math.dist(a["layout_xy"], b["layout_xy"])
                          for a in members for b in members), default=0.0)
            skew = max(m["observed_monotonic_ns"] for m in members) - min(
                m["observed_monotonic_ns"] for m in members)
            reasons = []
            if len({m["camera_id"] for m in members}) != len(members):
                reasons.append("SAME_CAMERA_MULTIPLE_TRACK_CONFLICT")
            if len(members) > 1:
                if any(m["observation_time_domain"] != ARRIVAL_DOMAIN for m in members):
                    reasons.append("UNSUPPORTED_TIMESTAMP_DOMAIN")
                if skew > self.max_skew_ns:
                    reasons.append("HOST_ARRIVAL_SKEW")
                if spread > self.max_spread:
                    reasons.append("SPATIAL_CONTRADICTION")
                if any(m.get("jump_candidate") for m in members):
                    reasons.append("LOCAL_JUMP_CANDIDATE")
            if gid:
                diagnostics.append({"provisional_global_id": gid, "contributors": len(members),
                    "spread_layout_px": spread, "arrival_skew_ms": skew / 1e6,
                    "decision": "SPLIT" if reasons else "ELIGIBLE",
                    "reasons": reasons, "identity_revision": revision})
            if mode == "experimental-global" and gid and not reasons:
                rgb = self._global_color(gid)
                markers.append({"key": key, "label": f"{gid}?", "provisional_global_id": gid,
                    "layout_xy": [sum(m["layout_xy"][axis] for m in members) / len(members)
                                  for axis in (0, 1)],
                    "color_rgb": rgb, "contributors": members,
                    "fusion_status": "EXPERIMENTAL_MAP_PIXEL_MEAN" if len(members) > 1
                                     else "SINGLE_CAMERA_PROVISIONAL",
                    "jump_candidate": bool(members[0].get("jump_candidate", False))})
            else:
                for obs in members:
                    label = f"{obs['camera_id'][-2:]}:{obs['local_tracker_id']}"
                    if mode == "local" and gid:
                        label += f" · {gid}?"
                    elif gid and reasons:
                        label += " !"
                    markers.append({"key": obs["local_key"], "label": label,
                        "provisional_global_id": None, "candidate_global_id": gid,
                        "layout_xy": obs["layout_xy"], "color_rgb": LOCAL_COLORS[obs["camera_id"]],
                        "contributors": [obs], "fusion_status": "SPLIT" if reasons else "LOCAL",
                        "jump_candidate": bool(obs.get("jump_candidate", False)) or bool(reasons)})

        if len(markers) > self.max_markers:
            rejected["marker_capacity_drops"] += len(markers) - self.max_markers
            markers = markers[:self.max_markers]
        active = {m["key"] for m in markers}
        # Remove withdrawn/split/expired groups and their trails immediately.
        for store in (self._trails, self._signatures, self._lineages):
            for key in list(store):
                if key not in active:
                    del store[key]
        for marker in markers:
            key = marker["key"]
            signature = tuple((m["camera_id"], m["local_tracker_id"], m["frame_index"],
                               m["source_pts_ns"]) for m in marker["contributors"])
            lineage = tuple((m["camera_id"], m["local_tracker_id"])
                            for m in marker["contributors"])
            trail = self._trails.setdefault(key, deque(maxlen=self.trail_points))
            if (self._lineages.get(key) != lineage or
                    (trail and math.dist(trail[-1], marker["layout_xy"]) > self.max_spread)):
                trail.clear()
            if self._signatures.get(key) != signature:
                trail.append(list(marker["layout_xy"]))
                self._signatures[key] = signature
                self._lineages[key] = lineage
            marker["trail"] = list(trail)
            marker["identity_revision"] = revision
        return {"mode": mode, "status": "EXPERIMENTAL / UNCALIBRATED",
                "identity_revision": revision, "markers": markers,
                "local_observations": observations, "groups": diagnostics,
                "rejected": dict(rejected), "policy": self.policy(),
                "snapshot_monotonic_ns": now,
                "cross_camera_exposure_synchronization": "UNVERIFIED"}

    @staticmethod
    def _global_color(gid: str) -> list[int]:
        palette = ([207, 132, 255], [120, 240, 160], [255, 120, 180], [150, 170, 255])
        index = int.from_bytes(hashlib.sha256(gid.encode()).digest()[:2], "big")
        return list(palette[index % len(palette)])


def paint_map_fusion_markers(painter, snapshot: dict[str, Any], image_box: tuple) -> dict:
    """Shared Qt paint path used by the existing map pane and CPU UI fixtures."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QPen
    from services.mv3dt_staging.live_map_projection import choose_marker_label_position

    ix, iy, iw, ih, sx, sy = image_box
    items, hidden = [], []
    font = QFont("Sans Serif", 9, QFont.Weight.Bold)
    metrics = QFontMetrics(font)
    painter.setFont(font)
    for marker in snapshot["markers"]:
        color = QColor(*marker["color_rgb"])
        x, y = marker["layout_xy"]
        px, py = ix + x * sx, iy + y * sy
        trail = marker.get("trail", [])
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 2))
        for first, second in zip(trail, trail[1:]):
            painter.drawLine(int(ix + first[0] * sx), int(iy + first[1] * sy),
                             int(ix + second[0] * sx), int(iy + second[1] * sy))
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#ff3434") if marker["jump_candidate"] else color,
                            3 if marker["jump_candidate"] else 1))
        radius = 8 if marker.get("provisional_global_id") else 6
        painter.drawEllipse(int(px - radius), int(py - radius), radius * 2, radius * 2)
        items.append({"key": marker["key"], "label": marker["label"],
                      "px": px, "py": py, "color": color,
                      "layout_xy": marker["layout_xy"],
                      "contributor_frame_keys": [[m["camera_id"], m["source_id"],
                          m["frame_index"], m["source_pts_ns"], m["local_tracker_id"]]
                          for m in marker["contributors"]]})
    occupied = [(round(item["px"] - 10), round(item["py"] - 10), 20, 20) for item in items]
    for item in items:
        width = metrics.horizontalAdvance(item["label"]) + 10
        height = metrics.height() + 6
        position = choose_marker_label_position(
            (item["px"], item["py"]), (width, height),
            (int(ix), int(iy), int(iw), int(ih)), occupied)
        if position is None:
            hidden.append(item["label"])
            item["label_rect"] = None
            continue
        x, y = position
        occupied.append((x, y, width, height))
        item["label_rect"] = [x, y, width, height]
        near_x = min(max(item["px"], x), x + width)
        near_y = min(max(item["py"], y), y + height)
        if math.dist((near_x, near_y), (item["px"], item["py"])) > 12:
            painter.setPen(QPen(item["color"], 1))
            painter.drawLine(int(item["px"]), int(item["py"]), int(near_x), int(near_y))
        painter.fillRect(x, y, width, height, QColor(0, 0, 0, 215))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(x + 5, y + 3 + metrics.ascent(), item["label"])
    return {"markers": [{k: v for k, v in item.items() if k != "color"} for item in items],
            "labels_hidden": hidden}
