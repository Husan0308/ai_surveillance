from __future__ import annotations

import os
import time

from .adaptive_reid_live import LiveAdaptiveTrackletReID
from .person_tracking_heatmap import CameraPersonTrackingHeatmap


class CameraPersonTrackingAdaptiveHeatmap(CameraPersonTrackingHeatmap):
    """Production camera wall with frozen ReID + online tracklet adaptation.

    No Qwen and no online neural-network training are used here. The embedding
    model remains frozen; only diverse tracklet memory, camera-pair thresholds and
    temporal one-to-one decisions adapt while the office runs.
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

        self.adaptive_reid: LiveAdaptiveTrackletReID | None = None
        super().__init__()
        if self.reid_mode == "external":
            self.adaptive_reid = LiveAdaptiveTrackletReID(self.global_reid)
            print(
                "CAMERA_ADAPTIVE_REID ready "
                "frozen_model=1 online_training=0 diverse_bank=1 stationary_bootstrap=1 "
                "fresh_evidence_votes=1 tracklet_multi_frame=1 camera_pair_adaptive=1 "
                "mutual_best=1 temporal_votes=1 late_reassoc=1 same_camera_unique=1 qwen=0 "
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
                f"samecam_collisions={row['samecam_collisions']} "
                f"samecam_repairs={row['samecam_repairs']} "
                f"pair={row['last_camera_pair']} "
                f"score={float(row['last_pair_score']):.3f} "
                f"margin={float(row['last_pair_margin']):.3f} "
                f"thr={float(row['last_threshold']):.3f} "
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
