from __future__ import annotations

import os
import time
from collections import defaultdict

from .person_tracking_final import CameraPersonTrackingFinal
from .qwen_reid_verifier_fast import FastQwenRoomReIDVerifier as QwenRoomReIDVerifier


class CameraPersonTrackingQwen(CameraPersonTrackingFinal):
    """Camera V2 runtime with tracklet-level room association plus Qwen audit.

    NvDCF owns local geometry. TAO ReID is the specialized appearance signal.
    Same-room peer tracks are matched over several recent crops with one-to-one
    mutual-best constraints. Qwen is deliberately a secondary judge for ambiguous
    cases, not the primary identity model.
    """

    def __init__(self) -> None:
        # Live 3x2 wall / physical room pairing:
        #   CAM-01 dev-room-2   <-> CAM-04 dev-room-1
        #   CAM-02 entrance-2   <-> CAM-05 entrance-1
        #   CAM-03 main-room-1  <-> CAM-06 main-room-2
        os.environ.setdefault(
            "CAMERA_V2_REID_ROOM_MAP",
            "0:0,3:0,1:1,4:1,2:2,5:2",
        )

        # Opposite indoor cameras produce strong viewpoint/pose changes. The old
        # 0.50 peer hard gate rejected the same person before enough tracklet
        # evidence accumulated. Lower only the peer provisional gates, while
        # increasing confirmation votes so weak one-frame matches never commit.
        os.environ.setdefault("CAMERA_V2_REID_PEER_MIN_REID", "0.38")
        os.environ.setdefault("CAMERA_V2_REID_PEER_CONFIRM_REID", "0.44")
        os.environ.setdefault("CAMERA_V2_REID_SAME_ROOM", "0.54")
        os.environ.setdefault("CAMERA_V2_REID_COVISIBLE", "0.52")
        os.environ.setdefault("CAMERA_V2_REID_CONFIRM_VOTES", "4")

        # Qwen sees a wider ambiguous band, but cannot by itself force an identity.
        os.environ.setdefault("CAMERA_V2_QWEN_MIN_PEER_REID", "0.28")
        os.environ.setdefault("CAMERA_V2_QWEN_AUDIT_SAME_GID_BELOW", "0.68")
        os.environ.setdefault("CAMERA_V2_QWEN_TIMEOUT", "18")
        os.environ.setdefault("CAMERA_V2_QWEN_MAX_RESULT_AGE", "18")
        os.environ.setdefault("CAMERA_V2_QWEN_MAX_PENDING", "1")

        # Specialized multi-frame tracklet consensus is the main same-room merge.
        os.environ.setdefault("CAMERA_V2_ROOM_AUTO_SAME", "0.52")
        os.environ.setdefault("CAMERA_V2_ROOM_AUTO_MARGIN", "0.045")
        os.environ.setdefault("CAMERA_V2_ROOM_AUTO_VOTES", "4")
        os.environ.setdefault("CAMERA_V2_QWEN_DIFF_REID_MAX", "0.38")

        self.qwen_reid: QwenRoomReIDVerifier | None = None
        self.same_camera_repairs = 0
        self.same_camera_collisions = 0
        super().__init__()

        if self.reid_mode != "off":
            self.qwen_reid = QwenRoomReIDVerifier()
            q = self.qwen_reid
            print(
                "CAMERA_QWEN_REID "
                f"enabled={int(q.enabled)} url={q.url} model={q.model} "
                "scope=same-room-peer-cameras async=1 reversible=1 "
                "tracklet_consensus=1 mutual_best=1 auto_votes=4 qwen_secondary=1 "
                "same_camera_unique=1 "
                "pairs=CAM01+CAM04,CAM02+CAM05,CAM03+CAM06 "
                f"room_map={self.global_reid.room_map}",
                flush=True,
            )

    def run(self) -> int:
        try:
            return super().run()
        finally:
            if self.qwen_reid is not None:
                self.qwen_reid.close()

    def _remember_qwen_visuals(self, cid, frame, detections, match_boxes=None) -> None:
        verifier = self.qwen_reid
        if verifier is None or not verifier.enabled or frame is None or not detections:
            return

        source_id = int(self.camera_index[cid])
        now = time.monotonic()
        with self.track_snapshot_lock:
            tracks = [
                dict(row)
                for (sid, _oid), row in self.latest_tracks.items()
                if sid == source_id
                and now - float(row.get("_seen_at", 0.0)) <= self.reid_track_cache_ttl
            ]
        if not tracks:
            return

        if match_boxes is None or len(match_boxes) != len(detections):
            match_boxes = [
                (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                for row in detections
            ]

        pairs = []
        for di, det_box in enumerate(match_boxes):
            for ti, track in enumerate(tracks):
                score = self._association_score(det_box, self._track_box(track))
                if score is not None:
                    pairs.append((score, di, ti))
        pairs.sort(reverse=True)

        used_dets = set()
        used_tracks = set()
        sx = frame.shape[1] / float(self.frame_width)
        sy = frame.shape[0] / float(self.frame_height)
        fh, fw = frame.shape[:2]

        for assoc_score, di, ti in pairs:
            if di in used_dets or ti in used_tracks:
                continue
            used_dets.add(di)
            used_tracks.add(ti)
            dx1, dy1, dx2, dy2, det_conf = [float(v) for v in detections[di]]

            if (
                dx1 <= 2.0
                or dy1 <= 2.0
                or dx2 >= float(self.frame_width - 2)
                or dy2 >= float(self.frame_height - 2)
            ):
                continue

            bw = max(1.0, dx2 - dx1)
            bh = max(1.0, dy2 - dy1)
            x1 = max(0, min(fw - 1, int(round((dx1 - 0.025 * bw) * sx))))
            y1 = max(0, min(fh - 1, int(round((dy1 - 0.015 * bh) * sy))))
            x2 = max(x1 + 1, min(fw, int(round((dx2 + 0.025 * bw) * sx))))
            y2 = max(y1 + 1, min(fh, int(round((dy2 + 0.020 * bh) * sy))))
            if y2 - y1 < self.reid_min_crop_h or x2 - x1 < 14:
                continue

            track = tracks[ti]
            tracker_conf = float(track.get("tracker_confidence", 0.0) or 0.0)
            quality = max(float(det_conf), tracker_conf) * max(0.25, float(assoc_score))
            verifier.remember(
                (source_id, int(track["object_id"])),
                frame[y1:y2, x1:x2],
                quality,
            )

    def _submit_external_reid(self, cid, frame, detections, match_boxes=None) -> None:
        super()._submit_external_reid(cid, frame, detections, match_boxes)
        self._remember_qwen_visuals(cid, frame, detections, match_boxes)

    @staticmethod
    def _identity_strength(manager, key) -> tuple[int, int, int, float]:
        binding = manager.bindings.get(key)
        if binding is None:
            return (-1, -1, -1, 0.0)
        gid = manager._resolve(binding.global_id)
        profile = manager.profiles.get(gid)
        known = int(bool(profile is not None and profile.known_name))
        state_rank = {"provisional": 0, "confirmed": 1, "anchor": 2}.get(
            binding.state, 0
        )
        _vec, _color, evidence = manager._track_prototype(key)
        age_rank = -float(binding.first_seen)
        return (known, state_rank, int(evidence), age_rank)

    def _detach_to_fresh_profile(self, manager, key, now: float) -> None:
        """Give a colliding local track a private Global ID without guessing."""
        binding = manager.bindings.get(key)
        if binding is None:
            return

        old_gid = manager._resolve(binding.global_id)
        vector, color, count = manager._track_prototype(key)
        if vector is not None and count >= 2:
            (
                alt_gid,
                _score,
                _second,
                _threshold,
                _reid,
                _color_score,
                _room,
                _covis,
                _context,
                accepted,
            ) = manager._candidate_decision(
                vector,
                color,
                key,
                now,
                exclude_gid=old_gid,
            )
            if accepted and alt_gid is not None:
                manager._switch_binding(
                    binding,
                    key,
                    int(alt_gid),
                    now,
                    provisional=True,
                )
                return

            quality = max(
                (item.quality for item in manager._evidence(key)),
                default=0.5,
            )
            manager._correct_to_new_anchor(
                binding,
                key,
                vector,
                color,
                binding.last_bbox,
                now,
                quality,
            )
            return

        manager._remove_owner_contributions(old_gid, key)
        profile = manager._new_profile(key[0], now)
        binding.global_id = profile.global_id
        binding.state = "provisional"
        binding.confirm_votes = 0
        binding.bad_votes = 0
        binding.switch_candidate = None
        binding.switch_votes = 0
        binding.last_committed_at = 0.0

    def _repair_same_camera_identity_collisions(self, now: float) -> int:
        """Hard invariant: two active tracks in one camera cannot share Global ID."""
        manager = self.global_reid
        groups = defaultdict(list)
        for key, binding in manager.bindings.items():
            if not manager._is_active(key, now):
                continue
            groups[(key[0], manager._resolve(binding.global_id))].append(key)

        repaired = 0
        for (_source_id, _gid), keys in groups.items():
            if len(keys) <= 1:
                continue
            self.same_camera_collisions += len(keys) - 1
            keys.sort(
                key=lambda key: self._identity_strength(manager, key),
                reverse=True,
            )
            keeper = keys[0]
            for loser in keys[1:]:
                manager.cannot_link[frozenset((keeper, loser))] = now + 30.0
                self._detach_to_fresh_profile(manager, loser, now)
                repaired += 1

        if repaired:
            self.same_camera_repairs += repaired
            manager.stats["same_camera_repairs"] = (
                int(manager.stats.get("same_camera_repairs", 0)) + repaired
            )
        return repaired

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        verifier = self.qwen_reid
        if self.reid_mode == "off":
            return result

        buffer = info.get_buffer()
        try:
            now = time.monotonic()
            with self.reid_lock:
                if verifier is not None and verifier.enabled:
                    verifier.service(self.global_reid, now)

                repaired = self._repair_same_camera_identity_collisions(now)
                if repaired and buffer is not None:
                    assignments = self.global_reid.label_assignments()
                    self.bridge.apply_global_identity(buffer, assignments)
        except Exception as exc:
            if verifier is not None:
                verifier.last_error = f"service:{type(exc).__name__}: {exc}"
        return result

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        verifier = self.qwen_reid
        if verifier is not None:
            q = verifier.snapshot()
            print(
                "CAMERA_QWEN_REID "
                f"enabled={int(bool(q['enabled']))} visual_tracks={q['visual_tracks']} "
                f"requests={q['requests']} responses={q['responses']} pending={q['pending']} "
                f"same={q['same']} different={q['different']} uncertain={q['uncertain']} "
                f"merges={q['merges']} auto_merges={q.get('auto_merges', 0)} "
                f"splits={q['splits']} cannot_links={q['cannot_links']} "
                f"pair_score={float(q.get('last_pair_score', -1.0)):.3f} "
                f"pair_margin={float(q.get('last_pair_margin', -1.0)):.3f} "
                f"vote_pairs={q.get('appearance_vote_pairs', 0)} "
                f"samecam_collisions={self.same_camera_collisions} "
                f"samecam_repairs={self.same_camera_repairs} "
                f"stale={q.get('stale', 0)} latency_ms={float(q['latency_ms']):.0f} "
                f"failed={q['failed']} dropped={q['dropped']} "
                f"last={q['last_verdict']} error={q['error'] or 'none'}",
                flush=True,
            )
        return keep


def main() -> int:
    os.environ.setdefault("CAMERA_V2_QWEN_VERIFY", "1")
    return CameraPersonTrackingQwen().run()


if __name__ == "__main__":
    raise SystemExit(main())
