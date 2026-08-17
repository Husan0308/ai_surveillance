from __future__ import annotations

import os
import time

from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .position_aware_reid import PositionAwareAdaptiveTrackletReID
from .stable_global_reid import StableGlobalReIDManager


class CameraPersonTrackingAdaptiveHeatmap(CameraPersonTrackingHeatmap):
    """Production camera wall with frozen ReID + sticky position-aware adaptation.

    No Qwen and no online neural-network training are used here. The embedding
    model stays frozen. New local tracks are sticky private anchors; one cross-camera
    controller performs multi-frame mutual-best association. For mostly-seated office
    workers it additionally learns camera-local stationary seat anchors and learns
    peer-camera seat correspondence only from already-confirmed identity leases.
    """

    def __init__(self) -> None:
        os.environ.setdefault(
            "CAMERA_V2_REID_ROOM_MAP",
            "0:0,3:0,1:1,4:1,2:2,5:2",
        )
        # Base same-room gates remain conservative. Cross-camera association is
        # decided by multi-frame adaptive tracklets, not by single-frame profiles.
        os.environ.setdefault("CAMERA_V2_REID_PEER_MIN_REID", "0.36")
        os.environ.setdefault("CAMERA_V2_REID_PEER_CONFIRM_REID", "0.42")
        os.environ.setdefault("CAMERA_V2_REID_SAME_ROOM", "0.54")
        os.environ.setdefault("CAMERA_V2_REID_COVISIBLE", "0.52")
        os.environ.setdefault("CAMERA_V2_REID_CONFIRM_VOTES", "4")

        # Position layer defaults are intentionally conservative: geometry is
        # learned only while a worker is stationary, and a peer seat link is taught
        # only by an already-confirmed ReID identity lease.
        os.environ.setdefault("CAMERA_V2_POS_WINDOW", "8")
        os.environ.setdefault("CAMERA_V2_POS_MIN_SAMPLES", "4")
        os.environ.setdefault("CAMERA_V2_POS_STATIONARY_SPREAD", "0.040")
        os.environ.setdefault("CAMERA_V2_POS_SEAT_RADIUS", "0.085")
        os.environ.setdefault("CAMERA_V2_POS_LINK_VOTES", "3")
        os.environ.setdefault("CAMERA_V2_POS_LINK_MIN_REID", "0.52")
        os.environ.setdefault("CAMERA_V2_POS_MATCH_BOOST", "0.11")

        self.adaptive_reid: PositionAwareAdaptiveTrackletReID | None = None
        super().__init__()

        if self.reid_mode == "external":
            # No frames have been consumed yet, so replacing the empty base manager
            # here is safe. From this point there is exactly one cross-camera writer.
            self.global_reid = StableGlobalReIDManager()
            self.adaptive_reid = PositionAwareAdaptiveTrackletReID(
                self.global_reid,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )
            print(
                "CAMERA_ADAPTIVE_REID ready "
                "frozen_model=1 online_training=0 single_controller=1 sticky_local_id=1 "
                "diverse_bank=1 stationary_bootstrap=1 fresh_both_cameras_votes=1 "
                "tracklet_multi_frame=1 camera_pair_adaptive=1 mutual_best=1 "
                "temporal_votes=1 peer_identity_lease=1 hysteresis=1 late_reassoc=1 "
                "auto_seat_geometry=1 learned_peer_seats=1 position_veto=1 "
                "same_camera_unique=1 qwen=0 "
                f"room_map={self.global_reid.room_map}",
                flush=True,
            )

    def _consume_external_reid(self) -> None:
        worker = self.external_reid
        if worker is None:
            return
        rows = worker.drain(24)
        if not rows:
            if worker.error:
                self.reid_error = worker.error
            return

        now = time.monotonic()
        with self.reid_lock:
            adaptive = self.adaptive_reid
            if adaptive is not None:
                adaptive.observe_rows(rows, now)

            # Sticky manager creates/updates local anchors only. It does not perform
            # frame-level cross-camera reassignment, so it cannot fight the adaptive
            # controller and make labels oscillate around a threshold.
            self.global_reid.observe(rows, now)

            if adaptive is not None:
                adaptive.reconcile(now)
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
                f"banks={row['banks']} samples={row['bank_samples']} "
                f"dup_skip={row['duplicate_skip']} replace={row['bank_replace']} "
                f"scans={row['scans']} comparisons={row['comparisons']} "
                f"mutual={row['mutual']} vote_pairs={row['vote_pairs']} "
                f"merges={row['adaptive_merges']} corrections={row['corrections']} "
                f"peer_locks={row.get('peer_locks_active', 0)} "
                f"lock_blocks={row.get('peer_lock_blocks', 0)} "
                f"lock_releases={row.get('lock_releases', 0)} "
                f"lock_corrections={row.get('lock_corrections', 0)} "
                f"fresh_vote_skip={row.get('fresh_vote_skip', 0)} "
                f"pos_tracks={row.get('position_tracks', 0)} "
                f"stationary={row.get('stationary_tracks', 0)} "
                f"seat_anchors={row.get('seat_anchors', 0)} "
                f"seat_links={row.get('seat_links', 0)} "
                f"pos_match={row.get('position_matches', 0)} "
                f"pos_veto={row.get('position_vetoes', 0)} "
                f"seat_conflicts={row.get('seat_link_conflicts', 0)} "
                f"samecam_collisions={row['samecam_collisions']} "
                f"samecam_repairs={row['samecam_repairs']} "
                f"pair={row['last_camera_pair']} "
                f"score={float(row['last_pair_score']):.3f} "
                f"margin={float(row['last_pair_margin']):.3f} "
                f"thr={float(row['last_threshold']):.3f} "
                f"release={float(row.get('release_floor', -1.0)):.3f} "
                f"adaptive_thresholds={threshold_text}",
                flush=True,
            )
        return keep


def main() -> int:
    os.environ.setdefault("CAMERA_V2_REID", "1")
    os.environ.setdefault("CAMERA_V2_REID_BACKEND", "external")
    return CameraPersonTrackingAdaptiveHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
