from __future__ import annotations

import os
import signal

from .step4_reid_pair_runtime_v1 import ROOT, V11Step4ReIDPairRuntimeV1
from .step4_reid_same_room_matcher_v1 import SameRoomMatcherConfigV1
from .step4_reid_same_room_shadow_cached_v3 import V11SameRoomMatcherShadowWorkerCachedV3
from .step4_reid_scheduler_v1 import ReIDResultV1


def _optional_threshold(name: str) -> float | None:
    raw = os.environ.get(name, "off").strip().lower()
    if raw in ("", "off", "none", "disabled"):
        return None
    return float(raw)


class V11Step4ReIDSameRoomRuntimeV1(V11Step4ReIDPairRuntimeV1):
    """Step 4 diagnostics-only reciprocal matcher layered on frozen prior steps."""

    def __init__(self) -> None:
        super().__init__()
        self.quality_run_sec = max(
            0.0,
            float(os.environ.get("V11_STEP4_MATCH_RUN_SEC", str(self.quality_run_sec))),
        )
        tsv_setting = os.environ.get(
            "V11_STEP4_MATCH_TSV",
            str(ROOT / "artifacts/reid/step4_same_room_matches_v1.tsv"),
        ).strip()
        self.match_tsv_path = (
            None if tsv_setting.lower() in ("", "0", "off", "none") else tsv_setting
        )
        config = SameRoomMatcherConfigV1(
            recent_age_sec=float(os.environ.get("V11_STEP4_MATCH_RECENT_SEC", "12.0")),
            min_robust_score=_optional_threshold("V11_STEP4_MATCH_MIN_ROBUST_SCORE"),
            min_row_margin=_optional_threshold("V11_STEP4_MATCH_MIN_ROW_MARGIN"),
            min_column_margin=_optional_threshold("V11_STEP4_MATCH_MIN_COLUMN_MARGIN"),
        )
        affinity_setting = os.environ.get("V11_STEP4_MATCH_CPU", "").strip().lower()
        affinity_cpu = (
            None
            if affinity_setting in ("", "off", "none", "disabled")
            else int(affinity_setting)
        )

        # Step 4 live acceptance intentionally validates only the first same-room
        # pair. Entrance and Main Rooms remain untouched until their dedicated
        # later steps. The matcher implementation itself stays generic.
        match_camera_rooms = {
            camera_id: self.pair_camera_rooms[camera_id]
            for camera_id in ("CAM-01", "CAM-04")
            if camera_id in self.pair_camera_rooms
        }
        if set(match_camera_rooms) != {"CAM-01", "CAM-04"}:
            raise ValueError("Step4 Devs matcher requires CAM-01 and CAM-04 room metadata")
        if len(set(match_camera_rooms.values())) != 1:
            raise ValueError("CAM-01 and CAM-04 must belong to the same room")

        self.match_worker = V11SameRoomMatcherShadowWorkerCachedV3(
            self.reid_gallery.gallery_views,
            match_camera_rooms,
            tsv_path=self.match_tsv_path,
            config=config,
            max_tracks_per_camera=8,
            min_cycle_interval_sec=2.0,
            affinity_cpu=affinity_cpu,
        )
        # Step3 and Step4 used to be independently notified by the same ReID
        # completion. Their separate 2-second cadence clocks could drift, causing
        # Step4 to see a new gallery fingerprint before Step3 had published the
        # exact score for it. Make Step3 the producer clock: the matcher wakes only
        # after a Step3 score cycle has completed publishing cache evidence.
        self.pair_worker.set_scores_published_callback(self.match_worker.notify)

        self.match_closed = False
        thresholds = (
            f"robust={config.min_robust_score if config.min_robust_score is not None else 'off'},"
            f"row_margin={config.min_row_margin if config.min_row_margin is not None else 'off'},"
            f"column_margin={config.min_column_margin if config.min_column_margin is not None else 'off'}"
        )
        print(
            "CAMERA_V11_STEP4_REID_SAME_ROOM_MATCHER_V1_ARCH "
            "mode=shadow diagnostic_no_merge=1 same_room_only=1 different_camera_only=1 "
            "evidence=step3_robust_score step3_formula_changed=0 step3_status_required=VALID "
            "reciprocal_before_proposal=1 assignment=scipy_linear_sum_assignment_maximize "
            "assignment_eligible_only=1 one_to_one=1 deterministic=1 "
            "worker=one async=1 dirty_slot=latest-only cadence=2.0s camera_queue=0 "
            "phase_delay=100ms evidence_cache=shared-step3-pair-score-v3 lookup_only=1 "
            "trigger=step3_evidence_cycle_complete "
            "camera_display_block=0 tracker_mutation=0 local_track_id_mutation=0 "
            "global_id=0 room_id=0 face=0 handoff=0 hysteresis=0 identity_state=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP4_REID_SAME_ROOM_MATCHER_V1_CONFIG "
            f"recent_age_sec={config.recent_age_sec:.3f} thresholds={thresholds} "
            f"matcher_affinity_cpu={affinity_cpu if affinity_cpu is not None else 'off'} "
            "live_threshold_tuning=0 max_tracks_per_camera=8 "
            "rooms=Devs:CAM-01+CAM-04,Entrance:CAM-02+CAM-05,Main_Rooms:CAM-03+CAM-06 "
            "priority=CAM-01+CAM-04 other_rooms_production=0 active_matcher_pairs=CAM-01+CAM-04 "
            f"tsv={self.match_tsv_path or 'disabled'} raw_embeddings_tsv=0",
            flush=True,
        )

    def _on_reid_result(self, result: ReIDResultV1) -> None:
        # super() updates the gallery and wakes the Step3 pair scorer. Step4 is
        # intentionally *not* woken here; it is notified by the Step3 completion
        # callback above, after exact pair evidence has been published.
        super()._on_reid_result(result)

    def _print_match_stats(self) -> None:
        row = self.match_worker.snapshot()
        print(
            "CAMERA_V11_STEP4_REID_SAME_ROOM_MATCHER_V1 "
            f"cycles={row['cycles']} "
            f"matrices_built={row['matrices_built']} "
            f"pairs_considered={row['pairs_considered']} "
            f"pairs_valid={row['pairs_valid']} "
            f"pairs_insufficient={row['pairs_insufficient']} "
            f"nonreciprocal={row['pairs_nonreciprocal']} "
            f"low_margin={row['pairs_low_margin']} "
            f"low_score={row['pairs_low_score']} "
            f"assignment_conflicts={row['assignment_conflicts']} "
            f"proposals={row['proposals']} "
            f"unique_proposals={row['unique_proposals']} "
            f"proposal_changes={row['proposal_changes']} "
            f"stale={row['pairs_stale']} "
            f"invalid={row['pairs_invalid']} "
            f"cache_hits={row['evidence_cache_hits']} "
            f"cache_misses={row['evidence_cache_misses']} "
            f"match_p50={row['match_p50_ms']:.3f}ms "
            f"match_p95={row['match_p95_ms']:.3f}ms "
            f"worker_errors={row['worker_errors']}",
            flush=True,
        )

    def _print_stats(self) -> None:
        super()._print_stats()
        self._print_match_stats()

    def run(self) -> int:
        try:
            self.match_worker.start()
            return super().run()
        finally:
            self._close_match_worker()

    def _close_match_worker(self) -> None:
        if self.match_closed:
            return
        self.match_closed = True
        self.match_worker.close(timeout_sec=3.0)
        self._print_match_stats()

    def close(self) -> None:
        self._close_match_worker()
        super().close()


def main() -> int:
    service = V11Step4ReIDSameRoomRuntimeV1()

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
