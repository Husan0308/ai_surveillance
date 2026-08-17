from __future__ import annotations

import time

from .kpr_reid_verifier import KPRPairVerifier
from .stable_adaptive_reid import StableAdaptiveTrackletReID


class KPRGuardedAdaptiveTrackletReID(StableAdaptiveTrackletReID):
    """Stable tracklet association with KPR as the final merge authority.

    TAO/tracklet evidence is still used to cheaply discover a mutual-best pair.
    It cannot commit a cross-camera Global-ID merge by itself. The first attempted
    merge schedules KPR; the pair is merged only after repeated fresh KPR part-based
    confirmations. This keeps the expensive transformer off the hot 20-FPS path.
    """

    def __init__(self, manager, verifier: KPRPairVerifier) -> None:
        self.kpr = verifier
        super().__init__(manager)
        self.stats.setdefault("kpr_gate_attempts", 0)
        self.stats.setdefault("kpr_gate_pending", 0)
        self.stats.setdefault("kpr_gate_blocked", 0)
        self.stats.setdefault("kpr_gate_unavailable", 0)
        self.stats.setdefault("kpr_gate_approved", 0)

    def _merge(self, a, b, now: float) -> bool:
        self.stats["kpr_gate_attempts"] += 1
        state = self.kpr.authorization(a, b, now)
        if state == "approved":
            merged = super()._merge(a, b, now)
            if merged:
                self.stats["kpr_gate_approved"] += 1
            return merged
        if state == "blocked":
            self.stats["kpr_gate_blocked"] += 1
            return False
        if state == "unavailable":
            self.stats["kpr_gate_unavailable"] += 1
            if not self.kpr.required:
                return super()._merge(a, b, now)
            return False
        self.stats["kpr_gate_pending"] += 1
        return False

    def _audit_peer_locks(self, now: float) -> None:
        # Once a pair is merged, the existing hysteresis/late-correction logic owns
        # the lease. KPR is a merge gate, not a frame-by-frame switcher. If a lease
        # is later released because repeated fresh appearance evidence contradicts
        # it, clear the old KPR authorization so a future reunion needs fresh KPR.
        before = dict(self.peer_owner)
        super()._audit_peer_locks(now)
        for a, b in before.items():
            if before.get(b) != a:
                continue
            if self.peer_owner.get(a) != b or self.peer_owner.get(b) != a:
                self.kpr.forget_pair(a, b)

    def reconcile(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.kpr.poll(now)
        super().reconcile(now)

    def snapshot(self) -> dict:
        row = super().snapshot()
        row.update(
            {
                "kpr_gate_attempts": self.stats["kpr_gate_attempts"],
                "kpr_gate_pending": self.stats["kpr_gate_pending"],
                "kpr_gate_blocked": self.stats["kpr_gate_blocked"],
                "kpr_gate_unavailable": self.stats["kpr_gate_unavailable"],
                "kpr_gate_approved": self.stats["kpr_gate_approved"],
            }
        )
        return row
