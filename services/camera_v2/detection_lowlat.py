from __future__ import annotations

from .detection_only_pose_v4 import DetectionOnlyLowLatencyV4


class DetectionLowLatency(DetectionOnlyLowLatencyV4):
    """Canonical low-latency runtime with the corrected pose-gate API contract."""

    def _pose_filter(self, cid: str, rows, frame):
        boxes = [
            (tuple(float(v) for v in coords), float(score))
            for coords, score in rows
        ]
        # pose_gate_v3 uses trusted_boxes=; older detection-only code used the
        # obsolete existing_boxes= spelling and therefore never exercised the
        # real pose path while it inherited the old scheduler.
        with self._pose_call_lock:
            return self.pose_gate.filter(
                cid,
                frame,
                boxes,
                trusted_boxes=None,
            )


def main() -> int:
    return DetectionLowLatency().run()


if __name__ == "__main__":
    raise SystemExit(main())
