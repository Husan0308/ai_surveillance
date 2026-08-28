from __future__ import annotations

import os
import signal

from .step4_identity_shadow_v1 import V11CrossCameraIdentityShadowV1
from .step4_reid_scheduler_v1 import ReIDResult
from .step4_tracking_reid_v1 import V11Step4TrackingReIDV1


class _V11CameraTrackletHandoffShadow(V11CrossCameraIdentityShadowV1):
    """Cross-room-only peer matcher over stable per-camera tracklet identities.

    NVIDIA peer-target re-association works on peer tracklets. Room identities remain
    useful for same-room grouping/telemetry, but they are intentionally not a
    prerequisite for a room handoff because weak same-room cross-view appearance can
    fragment that grouping and reset handoff votes.
    """

    def _pair_allowed(self, left, right):
        if left.room_id and right.room_id and left.room_id == right.room_id:
            gap_sec = abs(float(left.last_seen) - float(right.last_seen))
            return False, gap_sec, "same_room"
        return super()._pair_allowed(left, right)


class V11Step4TrackingReIDV2(V11Step4TrackingReIDV1):
    """Step4 V1 pipeline with peer re-association moved to camera tracklets.

    Ordering:
      frozen Step3 local track -> camera tracklet identity ->
      (a) independent same-room identity telemetry, and
      (b) cross-room camera-tracklet handoff matcher.

    The handoff matcher never mutates Step3/local IDs and remains shadow-only.
    """

    def __init__(self) -> None:
        super().__init__()

        # Replace the V1 room-ID handoff matcher before the run starts. Camera
        # tracklet IDs are the stable peer-tracklet primitive; room identity remains
        # an independent grouping layer and no longer gates handoff evidence.
        self.identity_shadow = _V11CameraTrackletHandoffShadow(
            gallery_size=int(os.environ.get("V11_STEP4_REID_GALLERY_SIZE", "4")),
            ttl_sec=float(os.environ.get("V11_STEP4_REID_SHADOW_TTL_SEC", "30")),
            weak_similarity=float(os.environ.get("V11_STEP4_REID_WEAK_SIM", "0.66")),
            candidate_similarity=float(os.environ.get("V11_STEP4_REID_CANDIDATE_SIM", "0.76")),
            strong_similarity=float(os.environ.get("V11_STEP4_REID_STRONG_SIM", "0.84")),
            min_margin=float(os.environ.get("V11_STEP4_REID_MIN_MARGIN", "0.04")),
            strong_margin=float(os.environ.get("V11_STEP4_REID_STRONG_MARGIN", "0.025")),
            min_gallery_samples=int(os.environ.get("V11_STEP4_REID_MIN_GALLERY_SAMPLES", "3")),
            candidate_votes=int(os.environ.get("V11_STEP4_REID_CANDIDATE_VOTES", "2")),
            strong_votes=int(os.environ.get("V11_STEP4_REID_STRONG_VOTES", "3")),
            min_cross_room_gap_sec=float(
                os.environ.get("V11_STEP4_REID_MIN_CROSS_ROOM_GAP_SEC", "1.5")
            ),
            allowed_room_transitions=self.reid_room_neighbors,
            require_handoff_lifecycle=True,
            recently_active_sec=float(
                os.environ.get("V11_STEP4_REID_RECENTLY_ACTIVE_SEC", "8.0")
            ),
            recently_lost_sec=float(
                os.environ.get("V11_STEP4_REID_RECENTLY_LOST_SEC", "30.0")
            ),
        )
        self.reid_last_match.clear()
        self.reid_last_crossroom_probe.clear()
        self.reid_crossroom_probe_count = 0

        print(
            "CAMERA_V11_STEP4_HANDOFF_V2 "
            "source=camera-tracklet cross_room_only=1 room_identity_dependency=0 "
            "topology=required lifecycle=active-to-recently-lost reciprocal=required "
            f"candidate_sim={self.identity_shadow.candidate_similarity:.2f} "
            f"candidate_votes={self.identity_shadow.candidate_votes} "
            "identity_merge=0",
            flush=True,
        )

    def _sync_camera_identity_activity(self, captured_at: float) -> None:
        active_by_camera: dict[str, set[str]] = {}
        for row in self.camera_tracklet_shadow.activity_snapshot(captured_at):
            camera_id = str(row["camera_id"])
            camera_identity = str(row["camera_identity"])
            active = bool(row["active"])

            # The peer handoff lifecycle must follow canonical camera-tracklet state,
            # not fragile Step3 local T-IDs and not room-cluster activity.
            self.identity_shadow.update_track_activity(
                camera_id=camera_id,
                track_id=camera_identity,
                active=active,
                observed_at=captured_at,
            )
            if active:
                active_by_camera.setdefault(camera_id, set()).add(camera_identity)

        # Keep V1 room grouping alive for same-room telemetry/corroboration only.
        for camera in self.cameras:
            self.room_identity_shadow.update_active_tracks(
                camera_id=camera.camera_id,
                track_ids=active_by_camera.get(camera.camera_id, set()),
                captured_at=captured_at,
            )

    def _sync_room_identity_activity(self, captured_at: float) -> None:
        # Deliberately do not feed ROOM::<room> identities into the peer matcher.
        # Room identity lifecycle is already maintained internally by the room layer.
        del captured_at

    def _on_reid_result(self, result: ReIDResult) -> None:
        candidate = result.candidate
        captured_at = candidate.captured_ns / 1_000_000_000.0

        camera_decision = self.camera_tracklet_shadow.observe(
            camera_id=candidate.camera_id,
            track_id=candidate.track_id,
            room_id=candidate.room_id,
            embedding=result.embedding,
            quality=candidate.quality,
            bbox_xyxy=candidate.bbox_xyxy,
            captured_at=captured_at,
        )
        camera_state = str(camera_decision["state"])
        camera_identity = str(camera_decision["camera_identity"])
        if camera_state in {"NEW", "JOIN"}:
            print(
                "CAMERA_V11_STEP4_CAMERA_IDENTITY "
                f"camera={candidate.camera_id} camera_identity={camera_identity} event={camera_state} "
                f"track={candidate.track_id} score={float(camera_decision['score']):.4f} "
                f"margin={float(camera_decision['margin']):.4f} "
                f"gap_ms={float(camera_decision['gap_sec']) * 1000.0:.1f} "
                f"center_dist={float(camera_decision['center_distance']):.3f} "
                f"samples={int(camera_decision['samples'])} members={int(camera_decision['members'])} merge=0",
                flush=True,
            )

        self._sync_camera_identity_activity(captured_at)
        if not camera_identity:
            return

        # Room grouping remains independent. A missing/split Room ID must never
        # suppress a valid cross-room peer-tracklet handoff.
        room_decision = self.room_identity_shadow.observe(
            camera_id=candidate.camera_id,
            track_id=camera_identity,
            room_id=candidate.room_id,
            embedding=result.embedding,
            quality=candidate.quality,
            captured_at=captured_at,
        )
        room_state = str(room_decision["state"])
        room_identity = str(room_decision["room_identity"])
        if room_state in {"NEW", "JOIN", "AMBIGUOUS_NEW"}:
            print(
                "CAMERA_V11_STEP4_ROOM_IDENTITY "
                f"room={candidate.room_id} room_identity={room_identity} event={room_state} "
                f"camera={candidate.camera_id} camera_identity={camera_identity} track={candidate.track_id} "
                f"score={float(room_decision['score']):.4f} margin={float(room_decision['margin']):.4f} "
                f"samples={int(room_decision['samples'])} members={int(room_decision['members'])} "
                f"collision_rejects={int(room_decision['collision_rejects'])} merge=0",
                flush=True,
            )
        self._sync_camera_identity_activity(captured_at)

        decision = self.identity_shadow.observe(
            camera_id=candidate.camera_id,
            track_id=camera_identity,
            room_id=candidate.room_id,
            embedding=result.embedding,
            quality=candidate.quality,
            captured_at=captured_at,
        )
        state = str(decision["state"])
        reason = str(decision["reason"])
        peer_camera = str(decision["candidate_camera"])
        peer_camera_identity = str(decision["candidate_track"])
        peer_room = self.camera_rooms.get(peer_camera, "")
        key = (candidate.camera_id, camera_identity)
        signature = (state, peer_camera, peer_camera_identity)
        with self.reid_result_lock:
            previous = self.reid_last_match.get(key)
            self.reid_last_match[key] = signature

        if peer_camera_identity:
            score = float(decision["score"])
            probe_signature = (
                peer_camera,
                peer_camera_identity,
                state,
                reason,
                int(score * 20.0),
            )
            emit_probe = False
            with self.reid_result_lock:
                previous_probe = self.reid_last_crossroom_probe.get(key)
                if (
                    previous_probe is None
                    or previous_probe[0] != probe_signature
                    or candidate.captured_ns - previous_probe[1] >= self.reid_crossroom_probe_interval_ns
                ):
                    self.reid_last_crossroom_probe[key] = (probe_signature, candidate.captured_ns)
                    self.reid_crossroom_probe_count += 1
                    emit_probe = True
            if emit_probe:
                gap_sec = float(decision["cross_room_gap_sec"])
                print(
                    "CAMERA_V11_STEP4_REID_CROSSROOM_PROBE "
                    f"camera={candidate.camera_id} track={candidate.track_id} "
                    f"camera_identity={camera_identity} room={candidate.room_id} "
                    f"room_identity={room_identity or '-'} peer_camera={peer_camera} "
                    f"peer_camera_identity={peer_camera_identity} peer_room={peer_room} "
                    f"state={state} reason={reason} gallery_score={score:.4f} "
                    f"prototype_score={float(decision['prototype_score']):.4f} "
                    f"margin={float(decision['margin']):.4f} "
                    f"samples={int(decision['gallery_samples'])} "
                    f"peer_samples={int(decision['peer_samples'])} "
                    f"reciprocal={int(bool(decision['reciprocal']))} "
                    f"votes={int(decision['consistency_votes'])} "
                    f"cross_room_gap_ms={gap_sec * 1000.0:.1f} "
                    f"batch={result.batch_size} queue_wait_ms={result.queue_wait_ms:.1f} merge=0",
                    flush=True,
                )

        if state in {"CANDIDATE", "STRONG"} and signature != previous:
            gap_sec = float(decision["cross_room_gap_sec"])
            print(
                "CAMERA_V11_STEP4_REID_MATCH "
                f"mode=camera-tracklet-handoff-shadow camera={candidate.camera_id} "
                f"track={candidate.track_id} camera_identity={camera_identity} "
                f"room={candidate.room_id} room_identity={room_identity or '-'} "
                f"peer_camera={peer_camera} peer_camera_identity={peer_camera_identity} "
                f"peer_room={peer_room} state={state} score={float(decision['score']):.4f} "
                f"prototype_score={float(decision['prototype_score']):.4f} "
                f"margin={float(decision['margin']):.4f} "
                f"samples={int(decision['gallery_samples'])} peer_samples={int(decision['peer_samples'])} "
                f"reciprocal={int(bool(decision['reciprocal']))} "
                f"votes={int(decision['consistency_votes'])} "
                f"cross_room_gap_ms={gap_sec * 1000.0:.1f} "
                f"batch={result.batch_size} queue_wait_ms={result.queue_wait_ms:.1f} merge=0",
                flush=True,
            )


def main() -> int:
    service = V11Step4TrackingReIDV2()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
