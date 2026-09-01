from __future__ import annotations

import os
import signal

from .step4_reid_pair_runtime_v1 import ROOT
from .step4_reid_same_room_runtime_v1 import V11Step4ReIDSameRoomRuntimeV1
from .step5_global_shadow_worker_v1 import V11GlobalShadowWorkerV1
from .step5_same_room_shadow_tap_v1 import V11SameRoomMatcherShadowWorkerStep5TapV1


class V11Step5GlobalShadowRuntimeV1(V11Step4ReIDSameRoomRuntimeV1):
    """Step5 shadow Global ID lifecycle layered on the passing Step4 matcher."""

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
        )
        # Step4 bound the pair-score completion callback to the matcher object that
        # existed during super().__init__(). Step5 replaces that matcher with the
        # tapping subclass, so rebind the producer callback to the worker that will
        # actually be started. Otherwise only close() wakes the new matcher.
        self.pair_worker.set_scores_published_callback(self.match_worker.notify)
        self.global_shadow_closed = False
        print(
            "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1_ARCH "
            "mode=shadow input=step4_MATCH_PROPOSED reciprocal_required=1 assigned_required=1 "
            "states=PROVISIONAL+CONFIRMED_SHADOW+EXPIRED_SHADOW "
            "confirm_observations=3 confirm_consecutive=3 different_pair_reuse=0 "
            "conflict_resolution=0 hysteresis=0 production_global_id=0 room_id=0 "
            "tracker_mutation=0 face=0 handoff=0 identity_accuracy_proven=0 "
            "queue=bounded async=1 matcher_blocking_state_work=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1_CONFIG "
            f"tsv={self.global_tsv_path or 'disabled'} queue_capacity="
            f"{self.global_shadow_worker._queue.maxsize} "
            "confirm_observations=3 confirm_consecutive=3 "
            f"expire_provisional_after_missed_cycles="
            f"{self.global_shadow_worker.machine.expire_provisional_after_missed_cycles}",
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
