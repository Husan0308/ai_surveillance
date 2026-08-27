from __future__ import annotations

import threading
import time
from collections import deque

from .runtime_v104_postmux_gate import PascalTrackerPostMuxGateRuntime


class PascalTrackerMuxArrivalAuditRuntime(PascalTrackerPostMuxGateRuntime):
    """V10.5: diagnostic-only per-source tracker_mux arrival/freshness audit.

    V10.4 improved raw batch fill with continuous pre-mux inputs, but raw batches
    are still ~37% partial and selected tracker frames remain stale.  Keep V10.4
    behavior unchanged and answer the next narrow question before any tuning:

      * are individual sources actually arriving at tracker_mux with large gaps?
      * or does nvstreammux emit an older queued frame even though a newer frame
        from that same source has already arrived?

    All comparisons are per-source.  Cross-camera PTS values are never compared.
    """

    def __init__(self) -> None:
        self._v105_lock = threading.RLock()
        self._v105_last_arrival_mono: dict[int, float] = {}
        self._v105_arrival_dt_ms: dict[int, deque[float]] = {}
        self._v105_input_count: dict[int, int] = {}
        self._v105_latest_input_pts: dict[int, int] = {}
        self._v105_mux_latest_lag_ms: dict[int, deque[float]] = {}
        self._v105_mux_rows: dict[int, int] = {}
        super().__init__()
        for source_id in self.index_camera:
            self._v105_arrival_dt_ms[source_id] = deque(maxlen=4096)
            self._v105_mux_latest_lag_ms[source_id] = deque(maxlen=4096)
            self._v105_input_count[source_id] = 0
            self._v105_mux_rows[source_id] = 0
        print(
            "CAMERA_V105_ARCH behavior_change=0 only_change=per-source-mux-arrival-audit "
            "measure=input-interarrival,same-source-latest-vs-mux-frame cross-source-pts=never",
            flush=True,
        )

    def _tracker_rate_probe(self, pad, info, cid: str):
        ret = super()._tracker_rate_probe(pad, info, cid)
        if ret != self.Gst.PadProbeReturn.OK:
            return ret
        buffer = info.get_buffer()
        if buffer is None or buffer.pts == self.Gst.CLOCK_TIME_NONE:
            return ret
        pts = int(buffer.pts)
        if not self._valid_pts(pts):
            return ret

        source_id = int(self.camera_index[cid])
        now = time.monotonic()
        with self._v105_lock:
            previous = self._v105_last_arrival_mono.get(source_id)
            if previous is not None:
                dt_ms = (now - previous) * 1000.0
                if 0.0 <= dt_ms <= 5000.0:
                    self._v105_arrival_dt_ms[source_id].append(float(dt_ms))
            self._v105_last_arrival_mono[source_id] = now
            self._v105_latest_input_pts[source_id] = pts
            self._v105_input_count[source_id] = self._v105_input_count.get(source_id, 0) + 1
        return ret

    def _inject_detector_probe(self, pad, info):
        buffer = info.get_buffer()
        rows = self._copy_pts(buffer)
        with self._v105_lock:
            for row in rows:
                source_id = int(row["source_id"])
                row_pts = int(row["buf_pts"])
                latest_pts = self._v105_latest_input_pts.get(source_id)
                if latest_pts is None or not self._valid_pts(row_pts):
                    continue
                # Same-source PTS domain only.  Positive lag means a fresher
                # frame from this camera had already reached the mux sink path.
                lag_ms = self._delta_ms(int(latest_pts), row_pts)
                if lag_ms is not None and lag_ms >= -5.0:
                    self._v105_mux_latest_lag_ms[source_id].append(max(0.0, lag_ms))
                    self._v105_mux_rows[source_id] = self._v105_mux_rows.get(source_id, 0) + 1
        return super()._inject_detector_probe(pad, info)

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._v105_lock:
            snapshot = []
            counts = dict(self._v105_input_count)
            for source_id, cid in sorted(self.index_camera.items()):
                snapshot.append(
                    (
                        cid,
                        list(self._v105_arrival_dt_ms.get(source_id, ())),
                        list(self._v105_mux_latest_lag_ms.get(source_id, ())),
                        int(counts.get(source_id, 0)),
                        int(self._v105_mux_rows.get(source_id, 0)),
                    )
                )

        max_count = max((item[3] for item in snapshot), default=0)
        for cid, arrivals, lag, count, mux_rows in snapshot:
            ratio = 100.0 * count / max(1, max_count)
            print(
                "CAMERA_V105_SOURCE "
                f"camera={cid} input_count={count} input_vs_max={ratio:.1f}% "
                f"arrival_p50={self._pct(arrivals, 0.50):.0f}ms "
                f"arrival_p95={self._pct(arrivals, 0.95):.0f}ms "
                f"arrival_p99={self._pct(arrivals, 0.99):.0f}ms "
                f"mux_latest_lag_p50={self._pct(lag, 0.50):.0f}ms "
                f"mux_latest_lag_p95={self._pct(lag, 0.95):.0f}ms "
                f"samples=arrival:{len(arrivals)},mux:{mux_rows}",
                flush=True,
            )
        return keep


def main() -> int:
    return PascalTrackerMuxArrivalAuditRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
