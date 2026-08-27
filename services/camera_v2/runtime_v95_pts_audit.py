from __future__ import annotations

import threading
import time
from collections import deque

from .pts_bridge import FramePtsBridge
from .runtime_v94_xmap import PascalXMapRuntime

_GST_CLOCK_TIME_NONE = (1 << 64) - 1


class PascalPtsAuditRuntime(PascalXMapRuntime):
    """V9.5: measure the real frame-time gap before changing bbox behavior again.

    V9.4 fixed detector X mapping.  V9.5 intentionally keeps every V9.4 runtime
    parameter and tracking/display behavior unchanged.  It reads NvDsFrameMeta.buf_pts
    from tracker-mux and display-mux batches so we can distinguish:

      * source -> display backlog,
      * source -> NvDCF backlog,
      * display frame PTS -> latest NvDCF frame PTS skew,
      * detector capture PTS -> tracker frame PTS when a correction is consumed,
      * actual per-camera NvDCF PTS cadence (not global batch cadence).

    All measurements are diagnostic only; no timestamps or boxes are rewritten.
    """

    def __init__(self) -> None:
        self._v95_lock = threading.RLock()
        self._v95_pts = None
        self._v95_tracker_latest: dict[int, tuple[int, int]] = {}
        self._v95_tracker_dt_ms: dict[int, deque[float]] = {}
        self._v95_display_tracker_ms: dict[int, deque[float]] = {}
        self._v95_source_display_ms: dict[int, deque[float]] = {}
        self._v95_source_tracker_ms: dict[int, deque[float]] = {}
        self._v95_detector_inject_ms: dict[int, deque[float]] = {}
        self._v95_gate_events: dict[str, deque[tuple[float, int]]] = {}
        self._v95_seq_pts: dict[tuple[str, int], int] = {}
        super().__init__()
        self._v95_pts = FramePtsBridge()
        for source_id in self.index_camera:
            self._v95_tracker_dt_ms[source_id] = deque(maxlen=1024)
            self._v95_display_tracker_ms[source_id] = deque(maxlen=2048)
            self._v95_source_display_ms[source_id] = deque(maxlen=2048)
            self._v95_source_tracker_ms[source_id] = deque(maxlen=2048)
            self._v95_detector_inject_ms[source_id] = deque(maxlen=512)
        for cid in self.camera_index:
            self._v95_gate_events[cid] = deque(maxlen=16)
        print(
            "CAMERA_V95_ARCH behavior_change=0 pts_audit=1 "
            "clock=NvDsFrameMeta.buf_pts per_camera=1 "
            "measure=source-display,source-tracker,display-tracker,detector-inject,tracker-cadence",
            flush=True,
        )

    @staticmethod
    def _valid_pts(pts: int) -> bool:
        value = int(pts)
        return 0 <= value < _GST_CLOCK_TIME_NONE

    @staticmethod
    def _delta_ms(newer: int, older: int) -> float | None:
        if not PascalPtsAuditRuntime._valid_pts(newer) or not PascalPtsAuditRuntime._valid_pts(older):
            return None
        delta = (int(newer) - int(older)) / 1_000_000.0
        if -2000.0 <= delta <= 5000.0:
            return float(delta)
        return None

    def _copy_pts(self, buffer) -> list[dict]:
        bridge = self._v95_pts
        if bridge is None or buffer is None:
            return []
        try:
            return bridge.copy(buffer, max_rows=max(16, len(self.cameras) * 2))
        except Exception as exc:
            print(
                f"CAMERA_V95_PTS warning={type(exc).__name__}:{exc}",
                flush=True,
            )
            return []

    def _tracker_probe(self, pad, info):
        ret = super()._tracker_probe(pad, info)
        buffer = info.get_buffer()
        rows = self._copy_pts(buffer)
        with self._v95_lock:
            for row in rows:
                source_id = int(row["source_id"])
                pts = int(row["buf_pts"])
                frame_num = int(row["frame_num"])
                if source_id not in self._v95_tracker_dt_ms or not self._valid_pts(pts):
                    continue
                previous = self._v95_tracker_latest.get(source_id)
                if previous is not None and pts > previous[0]:
                    dt_ms = (pts - previous[0]) / 1_000_000.0
                    if 1.0 <= dt_ms <= 1000.0:
                        self._v95_tracker_dt_ms[source_id].append(float(dt_ms))
                self._v95_tracker_latest[source_id] = (pts, frame_num)

                cid = self.index_camera.get(source_id)
                if cid is not None:
                    source_pts = self.stats[cid].last_pts_ns
                    if source_pts is not None:
                        lag = self._delta_ms(int(source_pts), pts)
                        if lag is not None and lag >= -5.0:
                            self._v95_source_tracker_ms[source_id].append(max(0.0, lag))
        return ret

    def _display_overlay_probe(self, pad, info):
        buffer = info.get_buffer()
        rows = self._copy_pts(buffer)
        with self._v95_lock:
            for row in rows:
                source_id = int(row["source_id"])
                display_pts = int(row["buf_pts"])
                if source_id not in self._v95_display_tracker_ms or not self._valid_pts(display_pts):
                    continue

                latest = self._v95_tracker_latest.get(source_id)
                if latest is not None:
                    skew = self._delta_ms(display_pts, latest[0])
                    if skew is not None:
                        self._v95_display_tracker_ms[source_id].append(skew)

                cid = self.index_camera.get(source_id)
                if cid is not None:
                    source_pts = self.stats[cid].last_pts_ns
                    if source_pts is not None:
                        backlog = self._delta_ms(int(source_pts), display_pts)
                        if backlog is not None and backlog >= -5.0:
                            self._v95_source_display_ms[source_id].append(max(0.0, backlog))
        return super()._display_overlay_probe(pad, info)

    def _detector_gate_probe(self, pad, info, cid: str):
        ret = super()._detector_gate_probe(pad, info, cid)
        if ret == self.Gst.PadProbeReturn.OK:
            buffer = info.get_buffer()
            if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
                pts = int(buffer.pts)
                if self._valid_pts(pts):
                    with self._v95_lock:
                        self._v95_gate_events[cid].append((time.monotonic(), pts))
        return ret

    def _publish_detector(self, cid: str, captured: float, boxes) -> None:
        super()._publish_detector(cid, captured, boxes)
        with self.pending_lock:
            pending = self.pending.get(cid)
        if pending is None:
            return
        seq = int(pending[0])
        with self._v95_lock:
            events = self._v95_gate_events.get(cid)
            if not events:
                return
            closest = min(events, key=lambda item: abs(float(item[0]) - float(captured)))
            if abs(float(closest[0]) - float(captured)) <= 0.25:
                self._v95_seq_pts[(cid, seq)] = int(closest[1])
            while events and float(captured) - float(events[0][0]) > 1.0:
                events.popleft()

    def _inject_detector_probe(self, pad, info):
        buffer = info.get_buffer()
        frame_rows = self._copy_pts(buffer)
        current_pts = {
            int(row["source_id"]): int(row["buf_pts"])
            for row in frame_rows
            if self._valid_pts(int(row["buf_pts"]))
        }
        before = dict(self.injected_seq)
        ret = super()._inject_detector_probe(pad, info)
        after = dict(self.injected_seq)

        with self._v95_lock:
            for cid, source_id in self.camera_index.items():
                new_seq = int(after.get(cid, 0))
                if new_seq <= int(before.get(cid, 0)):
                    continue
                detector_pts = self._v95_seq_pts.pop((cid, new_seq), None)
                tracker_pts = current_pts.get(int(source_id))
                if detector_pts is None or tracker_pts is None:
                    continue
                skew = self._delta_ms(tracker_pts, detector_pts)
                if skew is not None and skew >= -5.0:
                    self._v95_detector_inject_ms[int(source_id)].append(max(0.0, skew))
        return ret

    def _pct(self, values, p: float) -> float:
        return self._percentile_v93(list(values), p)

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._v95_lock:
            snapshots = []
            for source_id, cid in sorted(self.index_camera.items()):
                tracker_dt = list(self._v95_tracker_dt_ms.get(source_id, ()))
                display_tracker = list(self._v95_display_tracker_ms.get(source_id, ()))
                source_display = list(self._v95_source_display_ms.get(source_id, ()))
                source_tracker = list(self._v95_source_tracker_ms.get(source_id, ()))
                det_inject = list(self._v95_detector_inject_ms.get(source_id, ()))
                snapshots.append(
                    (
                        cid,
                        tracker_dt,
                        display_tracker,
                        source_display,
                        source_tracker,
                        det_inject,
                    )
                )

        for cid, tracker_dt, d_t, s_d, s_t, det in snapshots:
            dt_p50 = self._pct(tracker_dt, 0.50)
            dt_p95 = self._pct(tracker_dt, 0.95)
            tracker_pts_hz = 1000.0 / dt_p50 if dt_p50 > 0.0 else 0.0
            print(
                "CAMERA_V95_PTS "
                f"camera={cid} tracker_pts_hz={tracker_pts_hz:.2f} "
                f"tracker_dt_p50={dt_p50:.0f}ms tracker_dt_p95={dt_p95:.0f}ms "
                f"display_minus_tracker_p50={self._pct(d_t, 0.50):.0f}ms "
                f"display_minus_tracker_p95={self._pct(d_t, 0.95):.0f}ms "
                f"source_minus_display_p95={self._pct(s_d, 0.95):.0f}ms "
                f"source_minus_tracker_p95={self._pct(s_t, 0.95):.0f}ms "
                f"detector_to_inject_p50={self._pct(det, 0.50):.0f}ms "
                f"detector_to_inject_p95={self._pct(det, 0.95):.0f}ms "
                f"samples=dt:{len(tracker_dt)},dt_skew:{len(d_t)},det:{len(det)}",
                flush=True,
            )
        return keep


def main() -> int:
    return PascalPtsAuditRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
