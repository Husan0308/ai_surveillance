from __future__ import annotations

import time

from .global_reid import GlobalReIDManager, LocalBinding, _normalize


class StableGlobalReIDManager(GlobalReIDManager):
    """Global-ID store for an external tracklet association controller.

    NvDCF local tracks stay authoritative inside each camera. This manager keeps
    local IDs sticky and deliberately does NOT run its own cross-camera appearance
    reassignment on every embedding. New local tracks either continue a recent
    same-camera identity or start as a private anchor. Cross-camera merges/splits
    are owned by StableAdaptiveTrackletReID after multi-frame evidence.

    This removes the previous two-controller race where GlobalReIDManager could
    switch a binding while the adaptive room matcher was simultaneously trying to
    stabilize that same binding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.external_controller = True
        self.stats.setdefault("sticky_updates", 0)
        self.stats.setdefault("anchor_only_new", 0)

    def observe(self, rows: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self._expire(now)

        for row in rows:
            source_id = int(row.get("source_id", -1))
            object_id = int(row.get("object_id", -1))
            if source_id < 0 or object_id < 0:
                continue
            vector = _normalize(row.get("feature", ()))
            if vector is None:
                continue
            color = _normalize(row.get("color_feature", ()))
            bbox = self._bbox_from_row(row)
            key = (source_id, object_id)
            detector_conf = max(0.0, float(row.get("confidence", 0.0) or 0.0))
            tracker_conf = max(0.0, float(row.get("tracker_confidence", 0.0) or 0.0))
            quality = max(detector_conf, tracker_conf)

            agg_vector, agg_color, evidence_count = self._add_evidence(
                key, vector, color, now, quality
            )
            self.stats["observations"] += 1

            binding = self.bindings.get(key)
            if binding is not None and now - binding.last_seen <= self.binding_ttl:
                binding.last_seen = now
                binding.last_source = source_id
                binding.last_room = self.room_of(source_id)
                if bbox is not None:
                    binding.last_bbox = bbox

                # Existing local tracks are sticky. The adaptive controller may
                # explicitly call _switch_binding/_correct_to_new_anchor later,
                # but noisy frame-level appearance cannot flip the ID here.
                if binding.state == "provisional":
                    binding.confirm_votes += 1
                    if binding.confirm_votes >= self.confirm_votes_required:
                        binding.state = "confirmed"
                        self.stats["confirmed"] += 1
                        self._commit_to_profile(
                            binding,
                            key,
                            agg_vector,
                            agg_color,
                            source_id,
                            now,
                            quality,
                            force=True,
                        )
                else:
                    self._commit_to_profile(
                        binding,
                        key,
                        agg_vector,
                        agg_color,
                        source_id,
                        now,
                        quality,
                    )
                self.stats["sticky_updates"] += 1
                continue

            if evidence_count < self.min_new_track_samples:
                self.stats["pending_tracklet"] += 1
                continue

            # Preserve a recent same-camera identity after a short NvDCF break.
            # This is local continuity, not cross-camera matching.
            continuation_gid, _cont_score = self._find_same_camera_continuation(
                key, agg_vector, agg_color, bbox, now
            )
            if continuation_gid is not None:
                binding = LocalBinding(
                    global_id=continuation_gid,
                    first_seen=now,
                    last_seen=now,
                    last_source=source_id,
                    last_room=self.room_of(source_id),
                    last_bbox=bbox,
                    state="confirmed",
                    confirm_votes=self.confirm_votes_required,
                )
                self.bindings[key] = binding
                self.stats["continuation_match"] += 1
                self.stats["confirmed"] += 1
                self._commit_to_profile(
                    binding,
                    key,
                    agg_vector,
                    agg_color,
                    source_id,
                    now,
                    quality,
                    force=True,
                )
                continue

            # No direct/covisible cross-camera guess here. Start private, then let
            # the tracklet-level one-to-one controller merge only after consensus.
            self._create_anchor(key, agg_vector, agg_color, bbox, now, quality)
            self.stats["anchor_only_new"] += 1
