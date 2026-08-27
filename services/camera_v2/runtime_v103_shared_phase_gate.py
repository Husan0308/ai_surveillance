from __future__ import annotations

import time
from collections import deque

from .runtime_v101_tracker_batch_audit import PascalTrackerBatchAuditRuntime


class PascalTrackerSharedPhaseRuntime(PascalTrackerBatchAuditRuntime):
    """V10.3: phase-align per-camera tracker admission before nvstreammux.

    V10.1/V10.2 showed the six independent 10 Hz gates arrive at tracker_mux in
    different phases.  Lowering batch-size from 6 to 4 did not fix the tail: the
    mux still emitted ~50% partial batches and gate->mux p95 stayed ~130-145 ms.

    Keep V10.0's correct six-source mux target and 40 ms timeout.  Change only
    tracker admission semantics: instead of each camera advancing from its own
    previous PTS, all cameras use the same monotonic 10 Hz slot.  At most the
    first frame from each camera in a global slot is accepted.  With 20 FPS
    inputs this gives every source one fresh candidate in the same 100 ms epoch
    without blocking any source or enabling timestamp synchronization.
    """

    def __init__(self) -> None:
        self._v103_last_slot: dict[str, int] = {}
        self._v103_phase_ms: dict[str, deque[float]] = {}
        self._v103_accepts = 0
        self._v103_drops = 0
        super().__init__()
        for camera in self.cameras:
            self._v103_phase_ms[camera.camera_id] = deque(maxlen=2048)

        batch_size = None
        timeout_us = None
        try:
            batch_size = int(self.tracker_mux.get_property("batch-size"))
        except Exception:
            pass
        try:
            timeout_us = int(self.tracker_mux.get_property("batched-push-timeout"))
        except Exception:
            pass
        print(
            "CAMERA_V103_ARCH only_change=tracker-gate-independent-to-shared-monotonic-phase "
            f"tracker_batch_size={batch_size} target={len(self.cameras)} "
            f"tracker_timeout_us={timeout_us} tracker_hz={self.track_fps:.1f} "
            "sync_inputs=0 pts_cross_source_not_used_for_gate=1",
            flush=True,
        )
        if batch_size is not None and batch_size != len(self.cameras):
            raise RuntimeError(
                f"V10.3 expected tracker_mux batch-size={len(self.cameras)}, got {batch_size}"
            )
        if timeout_us is not None and timeout_us != 40_000:
            raise RuntimeError(
                f"V10.3 expected tracker_mux batched-push-timeout=40000, got {timeout_us}"
            )

    def _tracker_rate_probe(self, _pad, info, cid: str):
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.DROP
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.DROP

        now = time.monotonic()
        period_s = 1.0 / max(1e-6, float(self.track_fps))
        slot = int(now / period_s)
        slot_start = slot * period_s
        accept = False

        with self.track_gate_lock:
            if self._v103_last_slot.get(cid) != slot:
                self._v103_last_slot[cid] = slot
                self.track_buffers_passed += 1
                self._v103_accepts += 1
                accept = True
            else:
                self._v103_drops += 1

        if not accept:
            return self.Gst.PadProbeReturn.DROP

        phase_ms = max(0.0, (now - slot_start) * 1000.0)
        phases = self._v103_phase_ms.get(cid)
        if phases is not None:
            phases.append(phase_ms)

        # Preserve V9.9 exact-frame stage accounting because we bypass its old
        # independent per-source gate implementation here.
        if buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts = int(buffer.pts)
            if self._valid_pts(pts):
                source_id = int(self.camera_index[cid])
                with self._v99_lock:
                    self._v99_gate_mono[(source_id, pts)] = now
                    source_pts = self.stats[cid].last_pts_ns
                    if source_pts is not None:
                        lag = self._delta_ms(int(source_pts), pts)
                        if lag is not None and lag >= -5.0:
                            self._v99_source_gate_ms[source_id].append(max(0.0, lag))
                    self._v99_prune(now)

        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        parts = []
        all_phases: list[float] = []
        for cid in sorted(self._v103_phase_ms):
            vals = list(self._v103_phase_ms[cid])
            all_phases.extend(vals)
            if vals:
                parts.append(
                    f"{cid}:{self._pct(vals, 0.50):.0f}/{self._pct(vals, 0.95):.0f}ms"
                )
        print(
            "CAMERA_V103_PHASE "
            f"accepts={self._v103_accepts} drops={self._v103_drops} "
            f"phase_p50={self._pct(all_phases, 0.50):.0f}ms "
            f"phase_p95={self._pct(all_phases, 0.95):.0f}ms "
            f"per_camera={'/'.join(parts)}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalTrackerSharedPhaseRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
