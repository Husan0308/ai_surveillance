from __future__ import annotations

import os
import signal

from .step4_reid_camera_tracklet_v1 import (
    CameraTrackletConfigV1,
    CameraTrackletContinuityV1,
)
from .step4_reid_pair_runtime_v1 import ROOT
from .step4_reid_same_room_runtime_v1 import V11Step4ReIDSameRoomRuntimeV1
from .step5_global_shadow_worker_v1 import V11GlobalShadowWorkerV1
from .step5_same_room_shadow_tap_v1 import V11SameRoomMatcherShadowWorkerStep5TapV1


class V11Step5GlobalShadowRuntimeV1(V11Step4ReIDSameRoomRuntimeV1):
    """Step5 shadow Global ID lifecycle layered on Step4 + camera continuity."""

    def __init__(self) -> None:
        super().__init__()
        tsv_setting = os.environ.get(
            "V11_STEP5_GLOBAL_TSV",
            str(ROOT / "artifacts/reid/step5_global_shadow_v1.tsv"),
        ).strip()
        self.global_tsv_path = (
            None if tsv_setting.lower() in ("", "0", "off", "none") else tsv_setting
        )
        self.global_shadow_worker = V11GlobalShadowWorkerV1(
            tsv_path=self.global_tsv_path,
            queue_capacity=int(os.environ.get("V11_STEP5_GLOBAL_QUEUE_CAPACITY", "256")),
            confirm_observations=int(
                os.environ.get("V11_STEP5_GLOBAL_CONFIRM_OBSERVATIONS", "3")
            ),
            confirm_consecutive=int(
                os.environ.get("V11_STEP5_GLOBAL_CONFIRM_CONSECUTIVE", "3")
            ),
            expire_provisional_after_missed_cycles=int(
                os.environ.get("V11_STEP5_GLOBAL_EXPIRE_MISSED_CYCLES", "6")
            ),
            successor_confirm_observations=int(
                os.environ.get("V11_STEP5_SUCCESSOR_CONFIRM_OBSERVATIONS", "2")
            ),
            successor_max_gap_cycles=int(
                os.environ.get("V11_STEP5_SUCCESSOR_MAX_GAP_CYCLES", "2")
            ),
        )

        self.camera_tracklet_continuity = CameraTrackletContinuityV1(
            self.reid_gallery.gallery_views,
            self._camera_tracklet_active_snapshot,
            config=CameraTrackletConfigV1(),
        )

        old = self.match_worker
        self.match_worker = V11SameRoomMatcherShadowWorkerStep5TapV1(
            self.reid_gallery.gallery_views,
            old.camera_rooms,
            tsv_path=old.tsv_path,
            config=old.config,
            max_tracks_per_camera=old.max_tracks_per_camera,
            min_cycle_interval_sec=old.min_cycle_interval_sec,
            phase_delay_sec=old.phase_delay_sec,
            affinity_cpu=old.affinity_cpu,
            global_shadow_worker=self.global_shadow_worker,
            camera_tracklet_continuity=self.camera_tracklet_continuity,
        )
        self.pair_worker.set_scores_published_callback(self.match_worker.notify)
        self.global_shadow_closed = False
        print(
            "CAMERA_V11_STEP4_CAMERA_TRACKLET_V1_ARCH "
            "scope=CAM-01+CAM-04 mode=shadow same_camera_reid=1 recent_lost=1 "
            "simultaneous_stitch=0 active_overlap_defer=1 bounded_fallback=1 "
            "reciprocal=1 margin=1 confirm_cycles=2 tracker_mutation=0 "
            "step4_matcher_mutation=0 production_id_mutation=0",
            flush=True,
        )
        cfg = self.camera_tracklet_continuity.config
        print(
            "CAMERA_V11_STEP4_CAMERA_TRACKLET_V1_CONFIG "
            f"recent_gap_sec={cfg.recent_gap_sec:.2f} "
            f"max_overlap_sec={cfg.max_overlap_sec:.2f} "
            f"visually_active_sec={cfg.visually_active_sec:.2f} "
            f"min_samples={cfg.min_samples} "
            f"min_robust_score={cfg.min_robust_score:.3f} "
            f"min_margin={cfg.min_margin:.3f} "
            f"min_support_ge_065={cfg.min_support_ge_065} "
            f"confirm_cycles={cfg.confirm_cycles} "
            f"active_overlap_grace_cycles={cfg.active_overlap_grace_cycles}",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1_ARCH "
            "mode=shadow input=step4.5_camera_tracklet_MATCH_PROPOSED "
            "reciprocal_required=1 assigned_required=1 "
            "states=PROVISIONAL+CONFIRMED_SHADOW+EXPIRED_SHADOW "
            "confirm_observations=3 confirm_consecutive=3 "
            "identity_owner=person track_successor_reuse=1 current_member_per_camera=1 "
            "historical_aliases=1 same_cycle_overlap_reject=1 "
            "active_same_camera_member_reject=1 cross_owner_ambiguity=1 "
            "successor_evidence_required=1 cross_global_merge=0 "
            "hysteresis=0 production_global_id=0 room_id=0 tracker_mutation=0 "
            "face=0 handoff=0 identity_accuracy_proven=0 "
            "queue=bounded async=1 matcher_blocking_state_work=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1_CONFIG "
            f"tsv={self.global_tsv_path or 'disabled'} queue_capacity="
            f"{self.global_shadow_worker._queue.maxsize} "
            "confirm_observations=3 confirm_consecutive=3 "
            f"expire_provisional_after_missed_cycles="
            f"{self.global_shadow_worker.machine.expire_provisional_after_missed_cycles} "
            f"successor_confirm_observations="
            f"{self.global_shadow_worker.machine.successor_confirm_observations} "
            f"successor_max_gap_cycles="
            f"{self.global_shadow_worker.machine.successor_max_gap_cycles}",
            flush=True,
        )

    def _camera_tracklet_active_snapshot(self):
        return {camera: frozenset(tracks) for camera, tracks in self.active_track_ids.items()}

    def _print_camera_tracklet_stats(self) -> None:
        row = self.camera_tracklet_continuity.snapshot()
        print(
            "CAMERA_V11_STEP4_CAMERA_TRACKLET_V1 "
            f"stable_ids={row['stable_ids']} "
            f"raw_mapped={row['raw_mapped']} "
            f"stitched={row['stitched_total']} "
            f"pending_votes={row['pending_votes']} "
            f"deferred_allocations={row['deferred_allocations']} "
            f"pending_total={row['pending_total']} "
            f"suppressed={row['suppressed_total']} "
            f"low_score={row['low_score_total']} "
            f"low_margin={row['low_margin_total']} "
            f"nonreciprocal={row['nonreciprocal_total']} "
            f"overlap_reject={row['overlap_reject_total']} "
            f"active_overlap_deferred={row['active_overlap_deferred_total']} "
            f"active_overlap_fallback={row['active_overlap_fallback_allocated_total']} "
            f"same_camera_active_ct_collision={row['same_camera_active_ct_collision']} "
            f"successor_reid_score={row['successor_reid_score']:.6f} "
            f"successor_margin={row['successor_margin']:.6f} "
            f"successor_rejected_active_predecessor="
            f"{row['successor_rejected_active_predecessor']} "
            f"insufficient={row['insufficient_total']} "
            f"refresh_p50={row['refresh_p50_ms']:.3f}ms "
            f"refresh_p95={row['refresh_p95_ms']:.3f}ms",
            flush=True,
        )

    def _print_global_shadow_stats(self) -> None:
        row = self.global_shadow_worker.snapshot()
        print(
            "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1 "
            f"created={row['global_shadow_created']} "
            f"provisional={row['global_shadow_provisional']} "
            f"confirmed={row['global_shadow_confirmed']} "
            f"observations={row['global_shadow_observations']} "
            f"conflicts={row['global_shadow_conflicts']} "
            f"ambiguities={row['global_shadow_ambiguities']} "
            f"successor_attaches={row['global_shadow_successor_attaches']} "
            f"global_same_camera_member_reject="
            f"{row['global_same_camera_member_reject']} "
            f"expired={row['global_shadow_expired']} "
            f"active={row['global_shadow_active']} "
            f"member_tracks={row['global_shadow_member_tracks']} "
            f"queue_pending={row['queue_pending']} "
            f"queue_dropped={row['queue_dropped']} "
            f"events_written={row['events_written']} "
            f"state_p50={row['state_p50_ms']:.3f}ms "
            f"state_p95={row['state_p95_ms']:.3f}ms "
            f"worker_errors={row['worker_errors']}",
            flush=True,
        )

    def _print_stats(self) -> None:
        super()._print_stats()
        self._print_camera_tracklet_stats()
        self._print_global_shadow_stats()

    def run(self) -> int:
        self.global_shadow_worker.start()
        try:
            return super().run()
        finally:
            self._close_global_shadow_worker()

    def _close_global_shadow_worker(self) -> None:
        if self.global_shadow_closed:
            return
        self.global_shadow_closed = True
        self.global_shadow_worker.close(timeout_sec=3.0)
        self._print_camera_tracklet_stats()
        self._print_global_shadow_stats()

    def close(self) -> None:
        super().close()
        self._close_global_shadow_worker()


def main() -> int:
    service = V11Step5GlobalShadowRuntimeV1()

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
