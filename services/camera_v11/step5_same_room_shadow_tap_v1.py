from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

from .step4_reid_camera_tracklet_v1 import CameraTrackletContinuityV1
from .step4_reid_same_room_matcher_v1 import MATCH_PROPOSED, SameRoomPairDiagnosticV1
from .step4_reid_same_room_shadow_cached_v3 import V11SameRoomMatcherShadowWorkerCachedV3
from .step5_global_shadow_worker_v1 import V11GlobalShadowWorkerV1


@dataclass(frozen=True)
class Step5IdentityGateConfigV2:
    """Conservative identity gate between Step4 mechanics and Step5 ownership.

    Step4 remains a diagnostic/mechanics matcher.  A reciprocal maximum alone is
    not proof that two people are the same identity: in a crowded view the best
    edge may still be a near-tie between two different people.  Step5 therefore
    requires absolute appearance evidence plus row/column separation before a
    proposal is allowed to mutate Global Shadow state.
    """

    min_robust_score: float = 0.72
    min_row_margin: float = 0.05
    min_column_margin: float = 0.05
    min_support_ge_065: int = 3

    def __post_init__(self) -> None:
        for name in ("min_robust_score", "min_row_margin", "min_column_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_robust_score > 1.0:
            raise ValueError("min_robust_score must be <= 1")
        if self.min_support_ge_065 < 1:
            raise ValueError("min_support_ge_065 must be positive")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None or not raw.strip() else raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None or not raw.strip() else raw)


def step5_identity_gate_config_from_env_v2() -> Step5IdentityGateConfigV2:
    return Step5IdentityGateConfigV2(
        min_robust_score=_env_float("V11_STEP5_IDENTITY_MIN_ROBUST_SCORE", 0.72),
        min_row_margin=_env_float("V11_STEP5_IDENTITY_MIN_ROW_MARGIN", 0.05),
        min_column_margin=_env_float("V11_STEP5_IDENTITY_MIN_COLUMN_MARGIN", 0.05),
        min_support_ge_065=_env_int("V11_STEP5_IDENTITY_MIN_SUPPORT_GE_065", 3),
    )


def proposal_passes_step5_identity_gate_v2(
    row: SameRoomPairDiagnosticV1,
    config: Step5IdentityGateConfigV2,
) -> tuple[bool, str]:
    """Return whether a Step4 pair is strong enough to mutate global identity.

    A missing margin is acceptable only when there is no competitor on that side
    (row_second/column_second is None).  When competitors exist, lack of a margin
    is treated as ambiguous.  This explicitly supports NO-MATCH rather than
    forcing the highest-scoring edge into a GSH.
    """

    if row.status != MATCH_PROPOSED or not row.reciprocal or not row.assigned:
        return False, "not_proposed"
    score = row.robust_score
    if score is None or not math.isfinite(float(score)):
        return False, "invalid_score"
    if float(score) < config.min_robust_score:
        return False, "low_robust_score"
    if int(row.support_ge_065) < config.min_support_ge_065:
        return False, "low_support_ge_065"

    if row.row_second is not None:
        if row.row_margin is None or not math.isfinite(float(row.row_margin)):
            return False, "missing_row_margin"
        if float(row.row_margin) < config.min_row_margin:
            return False, "low_row_margin"
    if row.column_second is not None:
        if row.column_margin is None or not math.isfinite(float(row.column_margin)):
            return False, "missing_column_margin"
        if float(row.column_margin) < config.min_column_margin:
            return False, "low_column_margin"
    return True, "accepted"


class V11SameRoomMatcherShadowWorkerStep5TapV1(
    V11SameRoomMatcherShadowWorkerCachedV3
):
    """Step4 matcher with bounded Step4.5 continuity + conservative Step5 tap.

    Step4 still scores raw local T-IDs. Before a MATCH_PROPOSED row reaches Step5,
    CAM-01/CAM-04 raw fragments are translated to stable camera-tracklet IDs by
    the same-camera continuity layer. A not-yet-proven fragment is suppressed.

    Crucially, Step4's reciprocal/assignment result is not by itself an identity
    decision.  The Step5 identity gate additionally requires absolute ReID score,
    high-similarity support, and row/column margins when competitors exist. Weak
    or ambiguous pairs become NO-MATCH and never acquire/modify a GSH.
    """

    def __init__(
        self,
        *args,
        global_shadow_worker: V11GlobalShadowWorkerV1,
        camera_tracklet_continuity: CameraTrackletContinuityV1 | None = None,
        identity_gate_config: Step5IdentityGateConfigV2 | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.global_shadow_worker = global_shadow_worker
        self.camera_tracklet_continuity = camera_tracklet_continuity
        self.identity_gate_config = identity_gate_config or step5_identity_gate_config_from_env_v2()
        self.identity_gate_accepted = 0
        self.identity_gate_rejected = 0
        self.identity_gate_reasons: dict[str, int] = {}
        cfg = self.identity_gate_config
        print(
            "CAMERA_V11_STEP5_IDENTITY_GATE_V2 "
            f"min_robust_score={cfg.min_robust_score:.3f} "
            f"min_row_margin={cfg.min_row_margin:.3f} "
            f"min_column_margin={cfg.min_column_margin:.3f} "
            f"min_support_ge_065={cfg.min_support_ge_065} "
            "no_match_on_ambiguity=1 forced_match=0",
            flush=True,
        )

    def _match_cycle(self) -> None:
        next_cycle = int(self.cycles) + 1
        if self.camera_tracklet_continuity is not None:
            self.camera_tracklet_continuity.refresh(next_cycle, time.monotonic_ns())
        self.global_shadow_worker.enqueue_cycle_start(next_cycle, time.time_ns())
        try:
            super()._match_cycle()
        finally:
            self.global_shadow_worker.enqueue_cycle_end(next_cycle, time.time_ns())

    def _record_gate(self, accepted: bool, reason: str) -> None:
        if accepted:
            self.identity_gate_accepted += 1
            return
        self.identity_gate_rejected += 1
        self.identity_gate_reasons[reason] = self.identity_gate_reasons.get(reason, 0) + 1

    def identity_gate_snapshot(self) -> dict[str, object]:
        return {
            "accepted": self.identity_gate_accepted,
            "rejected": self.identity_gate_rejected,
            "reasons": dict(sorted(self.identity_gate_reasons.items())),
            "config": self.identity_gate_config,
        }

    def _tsv_row(self, timestamp_ns, cycle, row, stability):
        result = super()._tsv_row(timestamp_ns, cycle, row, stability)
        if row.status == MATCH_PROPOSED and row.reciprocal and row.assigned:
            accepted, reason = proposal_passes_step5_identity_gate_v2(
                row, self.identity_gate_config
            )
            self._record_gate(accepted, reason)
            if not accepted:
                return result

            proposal = row
            active_camera_members: tuple[tuple[str, str], ...] = ()
            if self.camera_tracklet_continuity is not None:
                proposal = self.camera_tracklet_continuity.canonicalize_proposal(row)
                active_camera_members = tuple(
                    self.camera_tracklet_continuity.active_camera_members()
                )
            if proposal is not None:
                self.global_shadow_worker.enqueue_proposal(
                    cycle,
                    timestamp_ns,
                    proposal,
                    active_camera_members=active_camera_members,
                )
        return result
