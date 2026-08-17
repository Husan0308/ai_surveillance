from __future__ import annotations

import os
import time

from .manual_geometry_reid import ManualGeometryAdaptiveTrackletReID
from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .stable_global_reid import StableGlobalReIDManager


class CameraPersonTrackingAdaptiveHeatmap(CameraPersonTrackingHeatmap):
    """Production camera wall with frozen ReID + manual calibrated room geometry.

    No Qwen, no online neural-network training and no automatic seat calibration are
    used here. The user supplies explicit image->room homography correspondences.
    Local NvDCF identities stay sticky; one tracklet controller fuses appearance
    with same-time calibrated room positions for cross-camera Global IDs.
    """

    def __init__(self) -> None:
        os.environ.setdefault(
            "CAMERA_V2_REID_ROOM_MAP",
            "0:0,3:0,1:1,4:1,2:2,5:2",
        )
        os.environ.setdefault("CAMERA_V2_REID_PEER_MIN_REID", "0.36")
        os.environ.setdefault("CAMERA_V2_REID_PEER_CONFIRM_REID", "0.42")
        os.environ.setdefault("CAMERA_V2_REID_SAME_ROOM", "0.54")
        os.environ.setdefault("CAMERA_V2_REID_COVISIBLE", "0.52")
        os.environ.setdefault("CAMERA_V2_REID_CONFIRM_VOTES", "4")

        # Geometry defaults. Distances themselves live in the manual JSON per room.
        os.environ.setdefault("CAMERA_V2_WORLD_WINDOW", "16")
        os.environ.setdefault("CAMERA_V2_WORLD_TTL", "5.0")
        os.environ.setdefault("CAMERA_V2_GEOMETRY_WEIGHT", "0.38")
        os.environ.setdefault("CAMERA_V2_GEOMETRY_MIN_REID", "0.24")
        os.environ.setdefault("CAMERA_V2_WORLD_FOOT_LIFT", "0.04")
        os.environ.setdefault("CAMERA_V2_CALIB_MAX_RMSE_M", "0.25")

        self.adaptive_reid: ManualGeometryAdaptiveTrackletReID | None = None
        super().__init__()

        if self.reid_mode == "external":
            self.global_reid = StableGlobalReIDManager()
            self.adaptive_reid = ManualGeometryAdaptiveTrackletReID(
                self.global_reid,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )
            calib = self.adaptive_reid.calibration.snapshot()
            print(
                "CAMERA_ADAPTIVE_REID ready "
                "frozen_model=1 online_training=0 single_controller=1 sticky_local_id=1 "
                "diverse_bank=1 fresh_both_cameras_votes=1 tracklet_multi_frame=1 "
                "camera_pair_adaptive=1 mutual_best=1 temporal_votes=1 "
                "peer_identity_lease=1 hysteresis=1 late_reassoc=1 "
                "manual_homography=1 auto_calibration=0 same_time_world_gate=1 "
                "same_camera_unique=1 qwen=0 "
                f"calibrated_cameras={calib['ready_cameras']} "
                f"calibration_sources={calib['camera_sources']} "
                f"room_map={self.global_reid.room_map}",
                flush=True,
            )
            if calib["errors"]:
                print(
                    "CAMERA_CALIBRATION warning=" + " | ".join(calib["errors"]),
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

            # Sticky manager owns local anchors only; calibrated tracklet controller
            # is the sole cross-camera writer, preventing frame-level ID oscillation.
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
            errors = row.get("calibration_errors", [])
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
                f"calib={row.get('calibration_ready_cameras', 0)}/6 "
                f"world_tracks={row.get('world_tracks', 0)} "
                f"geo_pairs={row.get('geometry_pairs', 0)} "
                f"geo_match={row.get('geometry_matches', 0)} "
                f"geo_veto={row.get('geometry_vetoes', 0)} "
                f"world_rmse={float(row.get('world_rmse_m', -1.0)):.3f}m "
                f"world_common={row.get('world_common', 0)} "
                f"samecam_collisions={row['samecam_collisions']} "
                f"samecam_repairs={row['samecam_repairs']} "
                f"pair={row['last_camera_pair']} "
                f"score={float(row['last_pair_score']):.3f} "
                f"margin={float(row['last_pair_margin']):.3f} "
                f"thr={float(row['last_threshold']):.3f} "
                f"release={float(row.get('release_floor', -1.0)):.3f} "
                f"adaptive_thresholds={threshold_text} "
                f"calib_error={'none' if not errors else errors[0]}",
                flush=True,
            )
        return keep


def main() -> int:
    os.environ.setdefault("CAMERA_V2_REID", "1")
    os.environ.setdefault("CAMERA_V2_REID_BACKEND", "external")
    return CameraPersonTrackingAdaptiveHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
