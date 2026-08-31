from __future__ import annotations

import os
import signal

from .step4_reid_pair_runtime_v1 import ROOT
from .step5_global_shadow_runtime_v1 import V11Step5GlobalShadowRuntimeV1
from .step5_same_room_shadow_tap_v1 import V11SameRoomMatcherShadowWorkerStep5TapV1
from .step6_step5_worker_tap_v1 import V11GlobalShadowWorkerStep6TapV1


class V11Step6GlobalShadowRuntimeV1(V11Step5GlobalShadowRuntimeV1):
    """Step6 temporal/conflict hysteresis layered on the passing Step5 shadow ID."""

    def __init__(self) -> None:
        super().__init__()

        old_global = self.global_shadow_worker
        old_match = self.match_worker
        verify_tsv_setting = os.environ.get(
            "V11_STEP6_VERIFY_TSV",
            str(ROOT / "artifacts/reid/step6_global_verify_v1.tsv"),
        ).strip()
        verify_tsv_path = (
            None
            if verify_tsv_setting.lower() in ("", "0", "off", "none")
            else verify_tsv_setting
        )

        self.global_shadow_worker = V11GlobalShadowWorkerStep6TapV1(
            tsv_path=old_global.tsv_path,
            queue_capacity=old_global._queue.maxsize,
            confirm_observations=old_global.machine.confirm_observations,
            confirm_consecutive=old_global.machine.confirm_consecutive,
            expire_provisional_after_missed_cycles=(
                old_global.machine.expire_provisional_after_missed_cycles
            ),
            verify_tsv_path=verify_tsv_path,
            verify_clean_observations=int(
                os.environ.get("V11_STEP6_VERIFY_CLEAN_OBSERVATIONS", "3")
            ),
            recover_clean_observations=int(
                os.environ.get("V11_STEP6_RECOVER_CLEAN_OBSERVATIONS", "3")
            ),
            persistent_conflict_observations=int(
                os.environ.get("V11_STEP6_PERSISTENT_CONFLICT_OBSERVATIONS", "3")
            ),
        )

        self.match_worker = V11SameRoomMatcherShadowWorkerStep5TapV1(
            self.reid_gallery.gallery_views,
            old_match.camera_rooms,
            tsv_path=old_match.tsv_path,
            config=old_match.config,
            max_tracks_per_camera=old_match.max_tracks_per_camera,
            min_cycle_interval_sec=old_match.min_cycle_interval_sec,
            phase_delay_sec=old_match.phase_delay_sec,
            affinity_cpu=old_match.affinity_cpu,
            global_shadow_worker=self.global_shadow_worker,
        )
        self.global_shadow_closed = False
        self.verify_tsv_path = verify_tsv_path

        print(
            "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1_ARCH "
            "mode=shadow input=step5_shadow_events same_room_pair=CAM-01+CAM-04 "
            "verification=time_consistency+conflict_hysteresis "
            "verify_clean_observations=3 recover_clean_observations=3 "
            "persistent_conflict_observations=3 conflict_action=hold_no_reassign "
            "geometry=disabled_requires_common_world_calibration raw_pixel_geometry=forbidden "
            "step5_owner_mutation=0 production_global_id=0 room_id=0 tracker_mutation=0 "
            "face=0 handoff=0 identity_accuracy_proven=0 "
            "execution=step5_async_worker matcher_blocking_verify_work=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1_CONFIG "
            f"verify_tsv={self.verify_tsv_path or 'disabled'} "
            f"verify_clean_observations={self.global_shadow_worker.verifier.verify_clean_observations} "
            f"recover_clean_observations={self.global_shadow_worker.verifier.recover_clean_observations} "
            f"persistent_conflict_observations="
            f"{self.global_shadow_worker.verifier.persistent_conflict_observations} "
            "reid_threshold_added=0 geometry_threshold_added=0",
            flush=True,
        )

    def _print_step6_stats(self) -> None:
        row = self.global_shadow_worker.snapshot()
        print(
            "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1 "
            f"records_created={row['verify_records_created']} "
            f"pending={row['verify_pending']} "
            f"verified={row['verify_verified']} "
            f"hold={row['verify_hold']} "
            f"expired={row['verify_expired']} "
            f"verified_total={row['verify_verified_total']} "
            f"hold_events={row['verify_hold_events']} "
            f"recovered_total={row['verify_recovered_total']} "
            f"persistent_conflicts={row['verify_persistent_conflicts']} "
            f"verify_events={row['verify_events_total']} "
            f"events_written={row['verify_events_written']} "
            f"verify_p50={row['verify_p50_ms']:.3f}ms "
            f"verify_p95={row['verify_p95_ms']:.3f}ms "
            f"verify_worker_errors={row['verify_worker_errors']} "
            "geometry_enabled=0 production_global_id=0 room_id=0 face=0 handoff=0",
            flush=True,
        )

    def _print_stats(self) -> None:
        super()._print_stats()
        self._print_step6_stats()

    def _close_global_shadow_worker(self) -> None:
        was_closed = self.global_shadow_closed
        super()._close_global_shadow_worker()
        if not was_closed:
            self._print_step6_stats()


def main() -> int:
    service = V11Step6GlobalShadowRuntimeV1()

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
