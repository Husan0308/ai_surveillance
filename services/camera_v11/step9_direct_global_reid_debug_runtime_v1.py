from __future__ import annotations

import os
import signal
import time

from .step8_two_person_debug_runtime_v1 import V11Step8TwoPersonDebugRuntimeV1
from .step9_direct_global_reid_v1 import (
    DirectGlobalReIDConfigV1,
    DirectGlobalReIDResolverV1,
)


class V11Step9DirectGlobalReIDDebugRuntimeV1(V11Step8TwoPersonDebugRuntimeV1):
    """Debug runtime whose visible identity label is owned by direct ReID memory.

    Existing Step5/Step6 shadow machinery remains observational only.  The label
    shown in the CAM-01/CAM-04 preview comes from DirectGlobalReIDResolverV1,
    which associates local tracker IDs directly to persistent appearance memory.
    """

    def __init__(self) -> None:
        super().__init__()
        self.direct_global_reid = DirectGlobalReIDResolverV1(
            DirectGlobalReIDConfigV1(
                min_robust_score=float(os.environ.get("V11_DIRECT_REID_MIN_ROBUST", "0.74")),
                min_top3_mean=float(os.environ.get("V11_DIRECT_REID_MIN_TOP3", "0.78")),
                min_median_best=float(os.environ.get("V11_DIRECT_REID_MIN_MEDIAN_BEST", "0.70")),
                min_support_ge_070=int(os.environ.get("V11_DIRECT_REID_MIN_SUPPORT_070", "3")),
                min_margin=float(os.environ.get("V11_DIRECT_REID_MIN_MARGIN", "0.06")),
                confirm_evidence=int(os.environ.get("V11_DIRECT_REID_CONFIRM_EVIDENCE", "2")),
                new_identity_evidence=int(os.environ.get("V11_DIRECT_REID_NEW_ID_EVIDENCE", "2")),
                global_memory_sec=float(os.environ.get("V11_DIRECT_REID_MEMORY_SEC", "1800")),
            )
        )
        self._direct_last_resolve_mono = 0.0
        self._direct_resolve_interval_sec = 0.20
        self._direct_last_decision_count = 0
        print(
            "CAMERA_V11_DIRECT_GLOBAL_REID_V1_ARCH "
            "input=local_reid_gallery identity_owner=persistent_person_memory "
            "pair_state_machine=not_identity_owner floor_calibration_required=0 "
            "manual_label_calibration_required=0 one_active_member_per_camera=1 "
            "no_forced_match=1 repeated_evidence=1 global_gallery=bounded8 "
            "local_track_switch_reassociate=1 cross_global_merge=0",
            flush=True,
        )
        cfg = self.direct_global_reid.config
        print(
            "CAMERA_V11_DIRECT_GLOBAL_REID_V1_CONFIG "
            f"min_samples={cfg.min_track_samples} min_robust={cfg.min_robust_score:.3f} "
            f"min_top3={cfg.min_top3_mean:.3f} min_median_best={cfg.min_median_best:.3f} "
            f"min_support_ge_070={cfg.min_support_ge_070} min_margin={cfg.min_margin:.3f} "
            f"confirm_evidence={cfg.confirm_evidence} new_identity_evidence={cfg.new_identity_evidence} "
            f"memory_sec={cfg.global_memory_sec:.0f}",
            flush=True,
        )

    def _resolve_direct_identity(self) -> None:
        now = time.monotonic()
        if now - self._direct_last_resolve_mono < self._direct_resolve_interval_sec:
            return
        self._direct_last_resolve_mono = now
        active = {
            camera: frozenset(str(track) for track in tracks)
            for camera, tracks in self.active_track_ids.items()
        }
        decisions = self.direct_global_reid.resolve(
            self.reid_gallery.gallery_views(),
            active,
            now_ns=time.monotonic_ns(),
        )
        for row in decisions:
            if row.decision in {"paired_create", "single_camera_create", "reassociate", "same_camera_collision_split"}:
                score = "none" if row.robust_score is None else f"{row.robust_score:.4f}"
                margin = "none" if row.margin is None else f"{row.margin:.4f}"
                print(
                    "CAMERA_V11_DIRECT_GLOBAL_REID_DECISION "
                    f"camera={row.camera_id} track={row.local_track_id} "
                    f"global_id={row.global_id or 'PENDING'} decision={row.decision} "
                    f"robust={score} margin={margin}",
                    flush=True,
                )

    def _on_reid_result(self, result) -> None:
        super()._on_reid_result(result)
        self._resolve_direct_identity()

    def _quality_track_update(
        self, camera_id: str, track_ids: tuple[str, ...], captured_ns: int
    ) -> None:
        super()._quality_track_update(camera_id, track_ids, captured_ns)
        self._resolve_direct_identity()

    def _identity_labels(self, camera_id: str, raw_track_id: str) -> tuple[str, str, str]:
        # Do not expose CT/GSH as the identity decision in this runtime.  The
        # second label slot is intentionally marked REID so the preview clearly
        # shows that GID is coming from the new direct resolver.
        try:
            global_id = self.direct_global_reid.global_for_track(camera_id, raw_track_id)
        except AttributeError:
            global_id = None
        return "REID", global_id or "GID-pending", "DIRECT"

    def _print_stats(self) -> None:
        super()._print_stats()
        if not hasattr(self, "direct_global_reid"):
            return
        row = self.direct_global_reid.snapshot()
        print(
            "CAMERA_V11_DIRECT_GLOBAL_REID_V1 "
            f"global_ids={row['global_ids']} active_members={row['active_members']} "
            f"created={row['created_total']} paired_create={row['paired_create_total']} "
            f"reassociated={row['reassociated_total']} no_match={row['no_match_total']} "
            f"low_score={row['low_score_total']} low_margin={row['low_margin_total']} "
            f"same_camera_reject={row['same_camera_reject_total']} "
            f"collision_repairs={row['active_collision_repairs']}",
            flush=True,
        )
        for record in self.direct_global_reid.records():
            members = ",".join(
                f"{camera}:{track}" for camera, track in record["current_members"].items()
            ) or "-"
            print(
                "CAMERA_V11_DIRECT_GLOBAL_REID_RECORD "
                f"global_id={record['global_id']} members={members} "
                f"aliases={len(record['aliases'])} samples={record['samples']}",
                flush=True,
            )


def main() -> int:
    service = V11Step9DirectGlobalReIDDebugRuntimeV1()

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
