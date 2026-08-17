from __future__ import annotations

import os
import time

from .kpr_guarded_reid import KPRGuardedAdaptiveTrackletReID
from .kpr_reid_verifier import KPRPairVerifier
from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .stable_global_reid import StableGlobalReIDManager


# Source IDs are the camera order in config/cameras.yaml.  These are the physical
# peer-camera pairs verified from the live 3x2 wall:
#   dev room : CAM-01 (0) <-> CAM-04 (3)
#   entrance : CAM-02 (1) <-> CAM-05 (4)
#   main room: CAM-03 (2) <-> CAM-06 (5)
# Do not use setdefault here: an old exported shell variable previously overrode
# the production mapping and made ReID compare people from different rooms.
PRODUCTION_ROOM_MAP = "0:0,3:0,1:1,4:1,2:2,5:2"


class CameraPersonTrackingAdaptiveHeatmap(CameraPersonTrackingHeatmap):
    """Production wall: fast TAO retrieval + sparse KPR final ReID gate.

    Geometry/calibration/Qwen are deliberately absent. NvDCF owns local tracks,
    the lightweight TAO embedding generates multi-frame candidates, and KPR is a
    sparse final authority before two peer-camera tracks may share a Global ID.

    KPR itself runs in an isolated worker process and is started lazily on the
    first real merge candidate. A native/PyTorch failure therefore cannot abort
    the DeepStream camera process, and a CUDA worker can fall back to CPU while
    the wall keeps running.
    """

    def __init__(self) -> None:
        previous_room_map = os.environ.get("CAMERA_V2_REID_ROOM_MAP", "").strip()
        if previous_room_map and previous_room_map != PRODUCTION_ROOM_MAP:
            print(
                "CAMERA_REID correcting stale room map "
                f"old={previous_room_map} new={PRODUCTION_ROOM_MAP}",
                flush=True,
            )
        os.environ["CAMERA_V2_REID_ROOM_MAP"] = PRODUCTION_ROOM_MAP

        os.environ.setdefault("CAMERA_V2_REID_PEER_MIN_REID", "0.36")
        os.environ.setdefault("CAMERA_V2_REID_PEER_CONFIRM_REID", "0.42")
        os.environ.setdefault("CAMERA_V2_REID_SAME_ROOM", "0.54")
        os.environ.setdefault("CAMERA_V2_REID_COVISIBLE", "0.52")
        os.environ.setdefault("CAMERA_V2_REID_CONFIRM_VOTES", "4")
        os.environ.setdefault("CAMERA_V2_KPR", "1")
        os.environ.setdefault("CAMERA_V2_KPR_REQUIRED", "1")

        self.kpr_reid: KPRPairVerifier | None = None
        self.adaptive_reid: KPRGuardedAdaptiveTrackletReID | None = None
        super().__init__()

        if self.reid_mode == "external":
            # Recreate the identity manager only after the production room map is
            # forced above.  This prevents stale shell environment from leaking
            # into the actual Global-ID topology.
            self.global_reid = StableGlobalReIDManager()
            self.kpr_reid = KPRPairVerifier()
            self.adaptive_reid = KPRGuardedAdaptiveTrackletReID(self.global_reid, self.kpr_reid)
            print(
                "CAMERA_ADAPTIVE_REID ready "
                "fast_candidate=tao-resnet50 final_authority=kpr-part-based "
                "kpr_sparse=1 kpr_fresh_votes=2 kpr_process_isolated=1 kpr_lazy_start=1 "
                "kpr_cuda_fallback_cpu=1 visibility_weighted_parts=1 "
                "frozen_model=1 online_training=0 single_controller=1 sticky_local_id=1 "
                "diverse_bank=1 fresh_both_cameras_votes=1 tracklet_multi_frame=1 "
                "camera_pair_adaptive=1 mutual_best=1 temporal_votes=1 "
                "peer_identity_lease=1 hysteresis=1 late_reassoc=1 "
                "geometry=0 calibration=0 auto_seat=0 qwen=0 same_camera_unique=1 "
                f"room_map={self.global_reid.room_map}",
                flush=True,
            )

    def run(self) -> int:
        try:
            return super().run()
        finally:
            if self.kpr_reid is not None:
                self.kpr_reid.close()

    def _remember_kpr_visuals(self, cid, frame, detections, match_boxes=None) -> None:
        verifier = self.kpr_reid
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
                (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
                for r in detections
            ]

        pairs = []
        for di, det_box in enumerate(match_boxes):
            for ti, track in enumerate(tracks):
                score = self._association_score(det_box, self._track_box(track))
                if score is not None:
                    pairs.append((score, di, ti))
        pairs.sort(reverse=True)
        used_dets, used_tracks = set(), set()
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
            bw, bh = max(1.0, dx2 - dx1), max(1.0, dy2 - dy1)
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
        self._remember_kpr_visuals(cid, frame, detections, match_boxes)

    def _consume_external_reid(self) -> None:
        worker = self.external_reid
        if worker is None:
            return
        rows = worker.drain(24)
        if not rows:
            if worker.error:
                self.reid_error = worker.error
            if self.kpr_reid is not None:
                with self.reid_lock:
                    self.kpr_reid.poll(time.monotonic())
            return
        now = time.monotonic()
        with self.reid_lock:
            if self.adaptive_reid is not None:
                self.adaptive_reid.observe_rows(rows, now)
            self.global_reid.observe(rows, now)
            if self.adaptive_reid is not None:
                self.adaptive_reid.reconcile(now)
            self.reid_vectors_seen += len(rows)
            self.reid_last_batch = len(rows)
        self.reid_error = worker.error

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        adaptive = self.adaptive_reid
        if adaptive is not None:
            row = adaptive.snapshot()
            thresholds = row.get("thresholds", {})
            threshold_text = ",".join(
                f"{pair}:{float(value):.3f}" for pair, value in thresholds.items()
            ) or "none"
            print(
                "CAMERA_ADAPTIVE_REID "
                f"banks={row['banks']} samples={row['bank_samples']} dup_skip={row['duplicate_skip']} "
                f"replace={row['bank_replace']} scans={row['scans']} comparisons={row['comparisons']} "
                f"mutual={row['mutual']} vote_pairs={row['vote_pairs']} merges={row['adaptive_merges']} "
                f"corrections={row['corrections']} peer_locks={row.get('peer_locks_active', 0)} "
                f"lock_blocks={row.get('peer_lock_blocks', 0)} lock_releases={row.get('lock_releases', 0)} "
                f"lock_corrections={row.get('lock_corrections', 0)} fresh_vote_skip={row.get('fresh_vote_skip', 0)} "
                f"kpr_gate_attempts={row.get('kpr_gate_attempts', 0)} kpr_gate_pending={row.get('kpr_gate_pending', 0)} "
                f"kpr_gate_blocked={row.get('kpr_gate_blocked', 0)} kpr_gate_approved={row.get('kpr_gate_approved', 0)} "
                f"samecam_collisions={row['samecam_collisions']} samecam_repairs={row['samecam_repairs']} "
                f"pair={row['last_camera_pair']} score={float(row['last_pair_score']):.3f} "
                f"margin={float(row['last_pair_margin']):.3f} thr={float(row['last_threshold']):.3f} "
                f"release={float(row.get('release_floor', -1.0)):.3f} adaptive_thresholds={threshold_text}",
                flush=True,
            )
        if self.kpr_reid is not None:
            q = self.kpr_reid.snapshot()
            print(
                "CAMERA_KPR_REID "
                f"enabled={int(bool(q['enabled']))} required={int(bool(q['required']))} ready={int(bool(q['ready']))} "
                f"backend={q['backend']} worker_pid={q.get('worker_pid', 0)} worker_exit={q.get('worker_exit', 'none')} "
                f"fallbacks={q.get('fallbacks', 0)} visual_tracks={q['visual_tracks']} "
                f"requests={q['requests']} responses={q['responses']} pending={q['pending']} "
                f"approved={q['approved']} blocked={q['blocked']} same={q['same']} "
                f"different={q['different']} uncertain={q['uncertain']} score={float(q['score']):.3f} "
                f"distance={float(q['distance']):.3f} parts={q['visible_parts']} latency_ms={float(q['latency_ms']):.0f} "
                f"failed={q['failed']} dropped={q['dropped']} wait_fresh={q['wait_fresh']} no_visual={q['no_visual']} "
                f"error={q['error'] or 'none'}",
                flush=True,
            )
        return keep


def main() -> int:
    os.environ.setdefault("CAMERA_V2_REID", "1")
    os.environ.setdefault("CAMERA_V2_REID_BACKEND", "external")
    os.environ.setdefault("CAMERA_V2_KPR", "1")
    os.environ.setdefault("CAMERA_V2_KPR_REQUIRED", "1")
    return CameraPersonTrackingAdaptiveHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
