"""Bounded RAM-only continuity for conservative provisional Global IDs.

This layer consumes only local-track matches that have already passed the
existing cross-camera OSNet gates. It preserves the original local NvDCF IDs,
keeps camera-specific galleries, never fuses map positions, and never writes
to a persistent identity store.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import math
from typing import Any

import numpy as np


CAMERAS = ("CAM-02", "CAM-05")
DEFAULT_TTL_SECONDS = 600.0
DEFAULT_LOST_AFTER_SECONDS = 5.0
DEFAULT_MAX_IDENTITIES = 128
DEFAULT_MAX_SAMPLES_PER_CAMERA = 24
DEFAULT_MAX_MEMBERS_PER_CAMERA = 64
DEFAULT_MAX_GLOBAL_EVENTS = 4096
DEFAULT_MAX_IDENTITY_EVENTS = 256


def _key(camera: str, local_id: Any) -> tuple[str, str]:
    if camera not in CAMERAS:
        raise ValueError(f"unsupported camera namespace: {camera}")
    return camera, str(local_id)


def _sample_rows(gallery: dict[tuple[str, str], list[dict[str, Any]]], key: tuple[str, str]) -> list[dict[str, Any]]:
    rows = gallery.get(key, [])
    return [row for row in rows if isinstance(row.get("vector"), np.ndarray)]


class ProvisionalGlobalIdentityMemory:
    """Stable provisional labels across local-ID changes, with strict abstention.

    Initial identity creation is possible only from a cross-camera pair emitted
    by the existing conservative matcher. A one-camera local-ID replacement
    may inherit an existing ID only when the already-gated pair contains the
    other camera's bound track and the two local tracks do not coexist in a
    source frame. If both local IDs change, reuse additionally requires
    reciprocal same-camera gallery matches, a competitor margin on both
    camera and identity sides, and no source-frame co-occurrence conflict.

    Similarity values are uncalibrated scores, not probabilities or verified
    identity claims.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        lost_after_seconds: float = DEFAULT_LOST_AFTER_SECONDS,
        max_identities: int = DEFAULT_MAX_IDENTITIES,
        max_samples_per_camera: int = DEFAULT_MAX_SAMPLES_PER_CAMERA,
        max_members_per_camera: int = DEFAULT_MAX_MEMBERS_PER_CAMERA,
        max_global_events: int = DEFAULT_MAX_GLOBAL_EVENTS,
        max_identity_events: int = DEFAULT_MAX_IDENTITY_EVENTS,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("identity gallery TTL must be positive")
        if not math.isfinite(lost_after_seconds) or lost_after_seconds < 0:
            raise ValueError("lost-state timeout must be nonnegative")
        if (max_identities < 1 or max_samples_per_camera < 2 or max_members_per_camera < 1
                or max_global_events < 1 or max_identity_events < 1):
            raise ValueError("identity memory bounds are invalid")
        self.ttl_ns = int(ttl_seconds * 1e9)
        self.lost_after_ns = int(lost_after_seconds * 1e9)
        self.max_identities = int(max_identities)
        self.max_samples_per_camera = int(max_samples_per_camera)
        self.max_members_per_camera = int(max_members_per_camera)
        self.max_global_events = int(max_global_events)
        self.max_identity_events = int(max_identity_events)
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._expired_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._bindings: dict[tuple[str, str], str] = {}
        self._current_local_ids: dict[str, set[str]] = {camera: set() for camera in CAMERAS}
        self._camera_observation_ns: dict[str, int | None] = {camera: None for camera in CAMERAS}
        self._next_id = 1
        self._revision = 0
        self._events: list[dict[str, Any]] = []
        self._event_sequence = 0
        self._counters = {
            "created": 0,
            "local_track_continuations": 0,
            "both_camera_id_continuations": 0,
            "ambiguous_continuity_abstentions": 0,
            "simultaneous_track_conflicts": 0,
            "binding_withdrawals": 0,
            "expired_global_ids": 0,
            "stale_decisions_rejected": 0,
            "capacity_abstentions": 0,
            "one_to_many_conflicts": 0,
        }

    @property
    def revision(self) -> int:
        return self._revision

    def expire(self, now_ns: int) -> list[str]:
        expired = [gid for gid, row in self._records.items()
                   if int(now_ns) - int(row["last_seen_ns"]) >= self.ttl_ns]
        for gid in expired:
            record = self._records.pop(gid)
            for camera in CAMERAS:
                for local_id in record["members"][camera]:
                    self._bindings.pop((camera, local_id), None)
            record["state"] = "EXPIRED"
            event = {
                "event": "CANDIDATE_WITHDRAWN",
                "provisional_global_id": gid,
                "reason": "600-second RAM-only gallery TTL expired",
                "production_identity_write": False,
                "monotonic_ns": int(now_ns),
            }
            record["revision_history"].append(copy.deepcopy(event))
            self._trim_identity_history(record)
            self._record_event(event)
            self._remember_expired(gid, record)
            self._counters["expired_global_ids"] += 1
            self._revision += 1
        self._refresh_states(int(now_ns))
        return expired

    def touch_observations(self, camera: str, local_ids: list[str], now_ns: int) -> None:
        """Update active/lost state from source-matched local observations."""
        now_ns = int(now_ns)
        self.expire(now_ns)
        active = {str(value) for value in local_ids}
        self._current_local_ids[camera] = active
        self._camera_observation_ns[camera] = now_ns
        for local_id in active:
            gid = self._bindings.get(_key(camera, local_id))
            if gid is None or gid not in self._records:
                continue
            record = self._records[gid]
            member = record["members"][camera].get(local_id)
            if member is not None:
                self._set_member_state(gid, camera, local_id, member, "ACTIVE", now_ns)
                member["last_seen_ns"] = now_ns
            self._set_identity_state(gid, record, "ACTIVE", now_ns)
            record["last_seen_ns"] = now_ns
        self._refresh_states(now_ns)

    def reconcile(
        self,
        matches: list[dict[str, Any]],
        candidate_pairs: list[dict[str, Any]],
        local_galleries: dict[tuple[str, str], list[dict[str, Any]]],
        tracks: dict[str, dict[int, list[dict[str, Any]]]],
        *,
        control_reference_max: float,
        now_ns: int,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Reconcile an already scored candidate set against retained GID galleries.

        ``expected_revision`` makes an asynchronous result fail closed if a
        newer identity decision was committed before this result returned.
        """
        now_ns = int(now_ns)
        self.expire(now_ns)
        if expected_revision is not None and int(expected_revision) != self._revision:
            self._counters["stale_decisions_rejected"] += 1
            return {"accepted": [], "abstained": [], "stale": True,
                    "revision": self._revision}
        if not math.isfinite(float(control_reference_max)):
            raise ValueError("control reference must be finite")

        old_bindings = dict(self._bindings)
        normalized: list[tuple[tuple[str, str], dict[str, Any]]] = []
        used = {camera: set() for camera in CAMERAS}
        for row in sorted(matches, key=lambda item: float(item.get("cosine_similarity_median", -1.0)),
                           reverse=True):
            pair = row.get("camera_local_pair", {})
            left = _key("CAM-02", pair.get("CAM-02"))
            right = _key("CAM-05", pair.get("CAM-05"))
            if left[1] == "None" or right[1] == "None":
                continue
            if left[1] in used["CAM-02"] or right[1] in used["CAM-05"]:
                self._counters["one_to_many_conflicts"] += 1
                continue
            used["CAM-02"].add(left[1]); used["CAM-05"].add(right[1])
            normalized.append(((left[1], right[1]), row))

        # First honor a direct accepted pair or a pair anchored by one existing
        # provisional ID. The cross-camera pair already passed the full current
        # matcher gates; the additional same-camera coexistence guard prevents
        # incompatible simultaneous tracks from sharing the ID.
        assignments: dict[tuple[str, str], str] = {}
        unresolved: list[tuple[tuple[str, str], dict[str, Any]]] = []
        for pair, row in normalized:
            left, right = pair
            gid_left = old_bindings.get(("CAM-02", left)) or self._member_gid("CAM-02", left)
            gid_right = old_bindings.get(("CAM-05", right)) or self._member_gid("CAM-05", right)
            if gid_left and gid_right and gid_left != gid_right:
                self._counters["ambiguous_continuity_abstentions"] += 1
                self._event("CONTINUITY_ABSTAINED", None, pair,
                            "candidate would join two existing provisional IDs", row, now_ns)
                continue
            gid = gid_left or gid_right
            if gid:
                camera, local_id = (("CAM-02", left) if gid_left is None
                                   else ("CAM-05", right) if gid_right is None
                                   else ("CAM-02", left))
                if self._coexists_with_gid(camera, local_id, gid, tracks):
                    self._counters["simultaneous_track_conflicts"] += 1
                    self._counters["ambiguous_continuity_abstentions"] += 1
                    self._event("CONTINUITY_ABSTAINED", gid, pair,
                                "same-camera local tracks coexist in source frames", row, now_ns)
                    continue
                assignments[("CAM-02", left)] = gid
                assignments[("CAM-05", right)] = gid
                if (gid_left is None) != (gid_right is None):
                    self._counters["local_track_continuations"] += 1
                    self._event("LOCAL_TRACK_ASSOCIATED", gid, pair,
                                "one local ID changed; current pair passed existing conservative gates",
                                row, now_ns)
                continue
            unresolved.append((pair, row))

        # New pairs can reuse an older GID only when both camera-local IDs have
        # independently reciprocal, margin-separated matches to that GID's
        # same-camera appearance galleries. No single-gallery/one-candidate
        # shortcut exists: missing competition means abstention.
        reused, ambiguous_pairs = self._resolve_both_changed(
            unresolved, candidate_pairs, local_galleries, tracks,
            float(control_reference_max), now_ns,
        )
        reused_pairs = {pair for pair, _gid, _row, _evidence in reused}
        for pair, gid, row, evidence in reused:
            assignments[("CAM-02", pair[0])] = gid
            assignments[("CAM-05", pair[1])] = gid
            self._counters["both_camera_id_continuations"] += 1
            self._event("BOTH_LOCAL_IDS_ASSOCIATED", gid, pair,
                        "both IDs changed; reciprocal same-camera gallery gates passed",
                        row, now_ns, evidence=evidence)

        for pair, row in unresolved:
            if pair in reused_pairs or pair in ambiguous_pairs:
                continue
            gid = self._allocate_gid(now_ns)
            if gid is None:
                self._counters["capacity_abstentions"] += 1
                self._event("CONTINUITY_ABSTAINED", None, pair,
                            "RAM-only provisional identity capacity reached", row, now_ns)
                continue
            assignments[("CAM-02", pair[0])] = gid
            assignments[("CAM-05", pair[1])] = gid
            self._event("CANDIDATE_CREATED", gid, pair,
                        "new pair passed existing conservative cross-camera gates",
                        row, now_ns)
            self._counters["created"] += 1

        # Any previously displayed binding not supported by a current accepted
        # pair is withdrawn from the UI. Its bounded GID gallery remains LOST
        # and can be considered for a future conservative return.
        new_bindings = dict(assignments)
        for key, old_gid in old_bindings.items():
            if key not in new_bindings:
                self._counters["binding_withdrawals"] += 1
                event = {
                    "event": "LOCAL_TRACK_BINDING_WITHDRAWN",
                    "provisional_global_id": old_gid,
                    "camera_id": key[0], "local_track_id": key[1],
                    "reason": "no current accepted candidate supports this local binding",
                    "monotonic_ns": int(now_ns),
                    "production_identity_write": False,
                }
                self._record_event(event, old_gid)
        self._bindings = new_bindings

        # Append newly observed embeddings with exact frame/PTS provenance.
        for key, gid in self._bindings.items():
            rows = _sample_rows(local_galleries, key)
            self._add_samples(gid, key[0], key[1], rows, now_ns)
        self._refresh_states(now_ns)
        self._revision += 1
        self.expire(now_ns)
        return {
            "accepted": [
                {"provisional_global_id": gid,
                 "camera_id": camera, "local_track_id": local_id}
                for (camera, local_id), gid in sorted(self._bindings.items())
            ],
            "abstained": sorted(ambiguous_pairs),
            "stale": False,
            "revision": self._revision,
        }

    def _resolve_both_changed(
        self,
        unresolved: list[tuple[tuple[str, str], dict[str, Any]]],
        candidate_pairs: list[dict[str, Any]],
        local_galleries: dict[tuple[str, str], list[dict[str, Any]]],
        tracks: dict[str, dict[int, list[dict[str, Any]]]],
        control_max: float,
        now_ns: int,
    ) -> tuple[list[tuple[tuple[str, str], str, dict[str, Any], dict[str, Any]]], set[tuple[str, str]]]:
        if not unresolved or not self._records:
            return [], set()
        pair_keys = {pair for pair, _ in unresolved}
        candidates = [pair for pair in pair_keys]
        known_members = {
            (camera, local_id)
            for identity in self._records.values()
            for camera in CAMERAS
            for local_id in identity["members"][camera]
        }
        local_ids = {
            camera: sorted(
                {pair[0 if camera == "CAM-02" else 1] for pair in candidates}
                | {
                    str(row.get("camera_local_pair", {}).get(camera))
                    for row in candidate_pairs
                    if row.get("camera_local_pair", {}).get(camera) is not None
                    and (camera, str(row["camera_local_pair"][camera])) not in known_members
                    and self._binding_or_member_gid(camera, str(row["camera_local_pair"][camera])) is None
                    and len(_sample_rows(local_galleries,
                                         (camera, str(row["camera_local_pair"][camera])))) >= 2
                }
            )
            for camera in CAMERAS
        }
        scores: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        high_similarities: set[tuple[str, str]] = set()
        for camera in CAMERAS:
            for local_id in local_ids[camera]:
                local = _sample_rows(local_galleries, (camera, local_id))
                if len(local) < 2:
                    continue
                local_vectors = np.stack([row["vector"] for row in local]).astype(np.float32)
                for gid, identity in self._records.items():
                    target_rows = identity["samples"][camera]
                    if len(target_rows) < 2:
                        continue
                    matrix = np.clip(local_vectors @ np.stack(
                        [row["vector"] for row in target_rows]).astype(np.float32).T, -1.0, 1.0)
                    means = matrix.mean(axis=1)
                    value = {
                        "median": float(np.median(matrix)),
                        "minimum": float(np.min(matrix)),
                        "dispersion": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                    }
                    scores.setdefault((camera, local_id), {})[gid] = value
                    if value["minimum"] > control_max:
                        high_similarities.add((camera, local_id))

        chosen_gid_by_track: dict[tuple[str, str], str] = {}
        track_margin_pass: dict[tuple[str, str], bool] = {}
        chosen_track_by_gid: dict[tuple[str, str], str] = {}
        gid_margin_pass: dict[tuple[str, str], bool] = {}
        for camera in CAMERAS:
            for local_id in local_ids[camera]:
                key = (camera, local_id)
                rows = scores.get(key, {})
                ranked = sorted(rows.items(), key=lambda item: item[1]["median"], reverse=True)
                if not ranked or ranked[0][1]["minimum"] <= control_max:
                    continue
                chosen_track_by_gid[(camera, ranked[0][0])] = local_id
                if len(ranked) >= 2:
                    dispersion = ranked[0][1]["dispersion"]
                    track_margin_pass[key] = (
                        ranked[0][1]["median"] - ranked[1][1]["median"] > 2.0 * dispersion
                    )
                else:
                    track_margin_pass[key] = False
                chosen_gid_by_track[key] = ranked[0][0]
            for gid in self._records:
                ranked_tracks = sorted(
                    ((local_id, scores.get((camera, local_id), {}).get(gid))
                     for local_id in local_ids[camera]),
                    key=lambda item: item[1]["median"] if item[1] else -2.0,
                    reverse=True,
                )
                if not ranked_tracks or ranked_tracks[0][1] is None:
                    continue
                top_local, top = ranked_tracks[0]
                chosen_track_by_gid[(camera, gid)] = top_local
                if len(ranked_tracks) >= 2 and ranked_tracks[1][1] is not None:
                    gid_margin_pass[(camera, gid)] = (
                        top["median"] - ranked_tracks[1][1]["median"] > 2.0 * top["dispersion"]
                    )
                else:
                    gid_margin_pass[(camera, gid)] = False

        reused = []
        ambiguous: set[tuple[str, str]] = set()
        row_by_pair = dict(unresolved)
        used_gids: set[str] = set()
        for pair, row in sorted(unresolved, key=lambda item: float(
                item[1].get("cosine_similarity_median", -1.0)), reverse=True):
            left_key = ("CAM-02", pair[0]); right_key = ("CAM-05", pair[1])
            gid_left = chosen_gid_by_track.get(left_key)
            gid_right = chosen_gid_by_track.get(right_key)
            if gid_left is None and gid_right is None:
                # If either candidate resembles retained identity memory but
                # cannot pass reciprocal/margin checks, abstain instead of
                # generating a duplicate candidate ID.
                if left_key in high_similarities or right_key in high_similarities:
                    ambiguous.add(pair)
                continue
            if gid_left is None or gid_left != gid_right:
                ambiguous.add(pair)
                continue
            gid = gid_left
            left_score = scores[left_key][gid]
            right_score = scores[right_key][gid]
            reciprocal = (
                chosen_track_by_gid.get(("CAM-02", gid)) == pair[0]
                and chosen_track_by_gid.get(("CAM-05", gid)) == pair[1]
            )
            margins = (
                track_margin_pass.get(left_key, False)
                and track_margin_pass.get(right_key, False)
                and gid_margin_pass.get(("CAM-02", gid), False)
                and gid_margin_pass.get(("CAM-05", gid), False)
            )
            minimums = left_score["minimum"] > control_max and right_score["minimum"] > control_max
            no_conflict = not self._coexists_with_gid("CAM-02", pair[0], gid, tracks) and not self._coexists_with_gid(
                "CAM-05", pair[1], gid, tracks)
            if reciprocal and margins and minimums and no_conflict and gid not in used_gids:
                used_gids.add(gid)
                reused.append((pair, gid, row, {
                    "CAM-02_same_camera_gallery_median": left_score["median"],
                    "CAM-02_same_camera_gallery_min": left_score["minimum"],
                    "CAM-05_same_camera_gallery_median": right_score["median"],
                    "CAM-05_same_camera_gallery_min": right_score["minimum"],
                    "reciprocal_best": True,
                    "competitor_margins_pass": True,
                    "control_reference_max": control_max,
                    "score_is_probability": False,
                }))
            else:
                ambiguous.add(pair)
        self._counters["ambiguous_continuity_abstentions"] += len(ambiguous)
        for pair in ambiguous:
            row = row_by_pair[pair]
            gid = chosen_gid_by_track.get(("CAM-02", pair[0])) or chosen_gid_by_track.get(("CAM-05", pair[1]))
            self._event("CONTINUITY_ABSTAINED", gid, pair,
                        "both-camera continuity gates did not all pass", row, now_ns)
        return reused, ambiguous

    def _coexists_with_gid(
        self, camera: str, local_id: str, gid: str,
        tracks: dict[str, dict[int, list[dict[str, Any]]]],
    ) -> bool:
        identity = self._records.get(gid)
        if identity is None:
            return False
        frames = tracks.get(camera, {})
        member_ids = set(identity["members"][camera]) - {str(local_id)}
        if not member_ids:
            return False
        for rows in frames.values():
            observed = {str(row.get("object_id")) for row in rows}
            if str(local_id) in observed and observed.intersection(member_ids):
                return True
        return False

    def _member_gid(self, camera: str, local_id: str) -> str | None:
        matches = [gid for gid, row in self._records.items()
                   if str(local_id) in row["members"][camera]]
        return matches[0] if len(matches) == 1 else None

    def _binding_or_member_gid(self, camera: str, local_id: str) -> str | None:
        return self._bindings.get((camera, str(local_id))) or self._member_gid(camera, str(local_id))

    def _recent_observation_ns(self, camera: str, local_id: str,
                               now_ns: int) -> int | None:
        observed_ns = self._camera_observation_ns[camera]
        if (str(local_id) not in self._current_local_ids[camera]
                or observed_ns is None
                or not 0 <= int(now_ns) - int(observed_ns) < self.lost_after_ns):
            return None
        return int(observed_ns)

    def _allocate_gid(self, now_ns: int) -> str | None:
        if len(self._records) >= self.max_identities:
            lost = [gid for gid, row in self._records.items() if row["state"] == "LOST"]
            if lost:
                self._drop_gid(lost[0], now_ns, "bounded RAM identity capacity eviction")
            else:
                return None
        gid = f"G{self._next_id:03d}"
        self._next_id += 1
        self._records[gid] = {
            "provisional_global_id": gid,
            "first_seen_ns": int(now_ns), "last_seen_ns": int(now_ns),
            "state": "ACTIVE",
            "members": {camera: {} for camera in CAMERAS},
            "samples": {camera: [] for camera in CAMERAS},
            "sample_keys": {camera: set() for camera in CAMERAS},
            "revision_history": [],
        }
        return gid

    def _drop_gid(self, gid: str, now_ns: int, reason: str) -> None:
        row = self._records.pop(gid)
        for camera in CAMERAS:
            for local_id in row["members"][camera]:
                self._bindings.pop((camera, local_id), None)
        row["state"] = "EXPIRED"
        event = {"event": "CANDIDATE_WITHDRAWN", "provisional_global_id": gid,
                 "reason": reason, "monotonic_ns": int(now_ns),
                 "production_identity_write": False}
        row["revision_history"].append(copy.deepcopy(event))
        self._trim_identity_history(row)
        self._record_event(event)
        self._remember_expired(gid, row)
        self._counters["expired_global_ids"] += 1
        self._revision += 1

    def _add_samples(self, gid: str, camera: str, local_id: str,
                     rows: list[dict[str, Any]], now_ns: int) -> None:
        identity = self._records.get(gid)
        if identity is None:
            return
        observed_ns = self._recent_observation_ns(camera, local_id, now_ns)
        if observed_ns is None:
            sample_observations = [
                int(item.get("record", {}).get("observed_monotonic_ns"))
                for item in rows
                if item.get("record", {}).get("observed_monotonic_ns") is not None
            ]
            recent_samples = [value for value in sample_observations
                              if 0 <= int(now_ns) - value < self.lost_after_ns]
            observed_ns = max(recent_samples) if recent_samples else None
        members = identity["members"][camera]
        member = members.get(local_id)
        if member is None:
            if len(members) >= self.max_members_per_camera:
                lost_members = sorted(
                    ((str(key), int(value["last_seen_ns"])) for key, value in members.items()
                     if value.get("state") == "LOST"),
                    key=lambda item: item[1],
                )
                if not lost_members:
                    return
                evicted_local_id, _last_seen = lost_members[0]
                del members[evicted_local_id]
                self._bindings.pop((camera, evicted_local_id), None)
                self._record_event({
                    "event": "LOCAL_TRACK_HISTORY_EVICTED",
                    "provisional_global_id": gid,
                    "camera_id": camera,
                    "local_track_id": evicted_local_id,
                    "reason": "bounded per-camera local-track history capacity",
                    "monotonic_ns": int(now_ns),
                    "production_identity_write": False,
                }, gid)
            provenance = [row.get("record", {}) for row in rows]
            frames = [int(r["frame"]) for r in provenance if r.get("frame") is not None]
            # A gallery result may arrive after its local track disappeared.
            # Do not turn an old crop-selection job into fresh activity.
            prior_seen_ns = int(identity["last_seen_ns"])
            member_seen_ns = int(observed_ns) if observed_ns is not None else prior_seen_ns
            member = {"first_seen_ns": int(observed_ns if observed_ns is not None else now_ns),
                      "last_seen_ns": member_seen_ns,
                      "first_source_frame": min(frames) if frames else None,
                      "last_source_frame": max(frames) if frames else None,
                      "state": "ACTIVE" if now_ns - member_seen_ns < self.lost_after_ns else "LOST"}
            members[local_id] = member
        if observed_ns is not None:
            member["last_seen_ns"] = max(int(member["last_seen_ns"]), int(observed_ns))
            self._set_member_state(
                gid, camera, local_id, member,
                "ACTIVE" if now_ns - int(member["last_seen_ns"]) < self.lost_after_ns else "LOST",
                now_ns,
            )
            identity["last_seen_ns"] = max(int(identity["last_seen_ns"]), int(observed_ns))
            self._set_identity_state(
                gid, identity,
                "ACTIVE" if now_ns - int(identity["last_seen_ns"]) < self.lost_after_ns else "LOST",
                now_ns,
            )
        for item in rows:
            record = item.get("record", {})
            sample_key = (local_id, record.get("frame"), record.get("source_pts_ns"))
            if sample_key in identity["sample_keys"][camera]:
                continue
            vector = np.asarray(item["vector"], dtype=np.float32)
            if vector.shape != (512,) or not np.isfinite(vector).all():
                continue
            norm = float(np.linalg.norm(vector.astype(np.float64)))
            if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                continue
            identity["samples"][camera].append({
                "vector": vector.copy(), "record": copy.deepcopy(record),
                "local_track_id": local_id,
            })
            identity["sample_keys"][camera].add(sample_key)
            while len(identity["samples"][camera]) > self.max_samples_per_camera:
                evicted = identity["samples"][camera].pop(0)
                evicted_record = evicted["record"]
                identity["sample_keys"][camera].discard((
                    evicted["local_track_id"], evicted_record.get("frame"),
                    evicted_record.get("source_pts_ns"),
                ))

    def _set_identity_state(self, gid: str, identity: dict[str, Any],
                            state: str, now_ns: int) -> None:
        old_state = str(identity["state"])
        if old_state == state:
            return
        identity["state"] = state
        self._record_event({
            "event": "IDENTITY_STATE_CHANGED",
            "provisional_global_id": gid,
            "from_state": old_state,
            "to_state": state,
            "monotonic_ns": int(now_ns),
            "production_identity_write": False,
        }, gid)

    def _set_member_state(self, gid: str, camera: str, local_id: str,
                          member: dict[str, Any], state: str, now_ns: int) -> None:
        old_state = str(member["state"])
        if old_state == state:
            return
        member["state"] = state
        self._record_event({
            "event": "LOCAL_TRACK_STATE_CHANGED",
            "provisional_global_id": gid,
            "camera_id": camera,
            "local_track_id": str(local_id),
            "from_state": old_state,
            "to_state": state,
            "monotonic_ns": int(now_ns),
            "production_identity_write": False,
        }, gid)

    def _event(self, event: str, gid: str | None, pair: tuple[str, str], reason: str,
               row: dict[str, Any], now_ns: int, *, evidence: dict[str, Any] | None = None) -> None:
        payload = {
            "event": event, "provisional_global_id": gid,
            "camera_local_pair": {"CAM-02": pair[0], "CAM-05": pair[1]},
            "reason": reason, "monotonic_ns": int(now_ns),
            "cross_camera_cosine_similarity": row.get("cosine_similarity_median"),
            "cross_camera_score_is_probability": False,
            "production_identity_write": False,
        }
        if evidence:
            payload["continuity_evidence"] = copy.deepcopy(evidence)
        self._record_event(payload, gid)

    def _record_event(self, event: dict[str, Any], gid: str | None = None) -> None:
        self._event_sequence += 1
        row = copy.deepcopy(event)
        row["event_sequence"] = self._event_sequence
        self._events.append(row)
        if len(self._events) > self.max_global_events:
            del self._events[:len(self._events) - self.max_global_events]
        identity = self._records.get(gid) if gid is not None else None
        if identity is not None:
            identity["revision_history"].append(copy.deepcopy(row))
            if len(identity["revision_history"]) > self.max_identity_events:
                del identity["revision_history"][:
                    len(identity["revision_history"]) - self.max_identity_events
                ]
        tombstone = self._expired_records.get(gid) if gid is not None else None
        if tombstone is not None:
            tombstone["revision_history"].append(copy.deepcopy(row))
            self._trim_identity_history(tombstone)

    def _trim_identity_history(self, identity: dict[str, Any]) -> None:
        history = identity["revision_history"]
        if len(history) > self.max_identity_events:
            del history[:len(history) - self.max_identity_events]

    def _refresh_states(self, now_ns: int) -> None:
        for gid, row in self._records.items():
            next_state = ("ACTIVE" if now_ns - int(row["last_seen_ns"]) < self.lost_after_ns
                          else "LOST")
            self._set_identity_state(gid, row, next_state, now_ns)
            for camera in CAMERAS:
                for local_id, member in row["members"][camera].items():
                    next_member_state = (
                        "ACTIVE" if now_ns - int(member["last_seen_ns"]) < self.lost_after_ns
                        else "LOST"
                    )
                    self._set_member_state(gid, camera, local_id, member,
                                           next_member_state, now_ns)

    def _remember_expired(self, gid: str, row: dict[str, Any]) -> None:
        """Keep a bounded tombstone while releasing the expired embeddings."""
        self._expired_records[gid] = {
            "provisional_global_id": gid,
            "first_seen_ns": row["first_seen_ns"],
            "last_seen_ns": row["last_seen_ns"],
            "state": "EXPIRED",
            "camera_local_track_histories": copy.deepcopy(row["members"]),
            "embedding_count_by_camera": {
                camera: len(row["samples"][camera]) for camera in CAMERAS
            },
            "embedding_provenance_by_camera": {
                camera: [
                    {
                        "local_track_id": sample["local_track_id"],
                        "source_id": sample["record"].get("source_id"),
                        "frame": sample["record"].get("frame"),
                        "source_pts_ns": sample["record"].get("source_pts_ns"),
                        "observed_monotonic_ns": sample["record"].get("observed_monotonic_ns"),
                    }
                    for sample in row["samples"][camera]
                ] for camera in CAMERAS
            },
            "embedding_dimensions": 512,
            "revision_history": copy.deepcopy(row["revision_history"]),
            "embeddings_retained": False,
        }
        while len(self._expired_records) > self.max_identities:
            self._expired_records.popitem(last=False)

    def assignment_map(self) -> dict[str, dict[str, dict[str, Any]]]:
        result: dict[str, dict[str, dict[str, Any]]] = {camera: {} for camera in CAMERAS}
        for (camera, local_id), gid in sorted(self._bindings.items()):
            row = self._records.get(gid)
            if row is None:
                continue
            result[camera][local_id] = {
                "provisional_global_id": gid,
                "display_label": f"{gid}?",
                "status": "AUTOMATIC_PROVISIONAL_SHADOW_NOT_VERIFIED",
                "identity_memory_state": row["state"],
                "identity_expires_monotonic_ns": int(row["last_seen_ns"]) + self.ttl_ns,
                "score_is_probability": False,
                "persistent": False,
                "map_points_fused": False,
            }
        return result

    def snapshot(self, now_ns: int) -> dict[str, Any]:
        self.expire(int(now_ns))
        return {
            "storage": "RAM_ONLY",
            "ttl_seconds": self.ttl_ns / 1e9,
            "max_identities": self.max_identities,
            "max_samples_per_camera": self.max_samples_per_camera,
            "max_members_per_camera": self.max_members_per_camera,
            "max_global_events": self.max_global_events,
            "max_identity_events": self.max_identity_events,
            "next_id_is_monotonic_and_not_reused": True,
            "identity_accuracy": "UNVERIFIED",
            "production_identity_write": False,
            "map_coordinates_fused": False,
            "active_mapping": self.assignment_map(),
            "identities": [
                {
                    "provisional_global_id": gid,
                    "first_seen_ns": row["first_seen_ns"],
                    "last_seen_ns": row["last_seen_ns"],
                    "state": row["state"],
                    "camera_local_track_histories": {
                        camera: {
                            local_id: {key: value for key, value in member.items()}
                            for local_id, member in row["members"][camera].items()
                        } for camera in CAMERAS
                    },
                    "embedding_count_by_camera": {
                        camera: len(row["samples"][camera]) for camera in CAMERAS
                    },
                    "embedding_provenance_by_camera": {
                        camera: [
                            {
                                "local_track_id": sample["local_track_id"],
                                "source_id": sample["record"].get("source_id"),
                                "frame": sample["record"].get("frame"),
                                "source_pts_ns": sample["record"].get("source_pts_ns"),
                                "observed_monotonic_ns": sample["record"].get("observed_monotonic_ns"),
                            }
                            for sample in row["samples"][camera]
                        ] for camera in CAMERAS
                    },
                    "embedding_dimensions": 512,
                    "revision_history": copy.deepcopy(row["revision_history"]),
                } for gid, row in self._records.items()
            ],
            "expired_id_tombstones": [copy.deepcopy(row)
                                       for row in self._expired_records.values()],
            "events": copy.deepcopy(self._events),
            "event_sequence": self._event_sequence,
            "counters": dict(self._counters),
            "revision": self._revision,
        }

    def clear(self) -> None:
        self._records.clear()
        self._expired_records.clear()
        self._bindings.clear()
        self._events.clear()
