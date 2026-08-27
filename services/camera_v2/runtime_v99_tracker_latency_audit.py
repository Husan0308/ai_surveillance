from __future__ import annotations

import threading
import time
from collections import deque

from .runtime_v98_tracker_pool import PascalTrackerPoolRuntime


class PascalTrackerLatencyAuditRuntime(PascalTrackerPoolRuntime):
    """V9.9: diagnostic-only localization of tracker freshness loss.

    V9.8 showed tracker_mux pool depth is not the cause.  Keep every V9.8
    behavior unchanged and measure three boundaries using the exact frame PTS:

      accepted tracker branch buffer -> tracker_mux output -> NvDCF output.

    The wall-time measurements are keyed by (source_id, buf_pts), so they do not
    confuse per-camera partial batches or compare unrelated frames.
    """

    def __init__(self) -> None:
        self._v99_lock = threading.RLock()
        self._v99_gate_mono: dict[tuple[int, int], float] = {}
        self._v99_mux_mono: dict[tuple[int, int], float] = {}
        self._v99_source_gate_ms: dict[int, deque[float]] = {}
        self._v99_gate_mux_ms: dict[int, deque[float]] = {}
        self._v99_mux_nvdcf_ms: dict[int, deque[float]] = {}
        self._v99_source_mux_ms: dict[int, deque[float]] = {}
        super().__init__()
        for source_id in self.index_camera:
            self._v99_source_gate_ms[source_id] = deque(maxlen=2048)
            self._v99_gate_mux_ms[source_id] = deque(maxlen=2048)
            self._v99_mux_nvdcf_ms[source_id] = deque(maxlen=2048)
            self._v99_source_mux_ms[source_id] = deque(maxlen=2048)
        print(
            "CAMERA_V99_ARCH behavior_change=0 only_change=tracker-stage-PTS-audit "
            "measure=source-gate,gate-mux,mux-NvDCF,source-mux exact_pts_match=1",
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
        with self._v99_lock:
            self._v99_gate_mono[(source_id, pts)] = now
            source_pts = self.stats[cid].last_pts_ns
            if source_pts is not None:
                lag = self._delta_ms(int(source_pts), pts)
                if lag is not None and lag >= -5.0:
                    self._v99_source_gate_ms[source_id].append(max(0.0, lag))
            self._v99_prune(now)
        return ret

    def _inject_detector_probe(self, pad, info):
        buffer = info.get_buffer()
        rows = self._copy_pts(buffer)
        now = time.monotonic()
        with self._v99_lock:
            for row in rows:
                source_id = int(row["source_id"])
                pts = int(row["buf_pts"])
                if not self._valid_pts(pts):
                    continue
                key = (source_id, pts)
                gate_mono = self._v99_gate_mono.get(key)
                if gate_mono is not None:
                    self._v99_gate_mux_ms[source_id].append(max(0.0, (now - gate_mono) * 1000.0))
                self._v99_mux_mono[key] = now
                cid = self.index_camera.get(source_id)
                if cid is not None:
                    source_pts = self.stats[cid].last_pts_ns
                    if source_pts is not None:
                        lag = self._delta_ms(int(source_pts), pts)
                        if lag is not None and lag >= -5.0:
                            self._v99_source_mux_ms[source_id].append(max(0.0, lag))
            self._v99_prune(now)
        return super()._inject_detector_probe(pad, info)

    def _tracker_probe(self, pad, info):
        buffer = info.get_buffer()
        rows = self._copy_pts(buffer)
        now = time.monotonic()
        with self._v99_lock:
            for row in rows:
                source_id = int(row["source_id"])
                pts = int(row["buf_pts"])
                key = (source_id, pts)
                mux_mono = self._v99_mux_mono.get(key)
                if mux_mono is not None:
                    self._v99_mux_nvdcf_ms[source_id].append(max(0.0, (now - mux_mono) * 1000.0))
            self._v99_prune(now)
        return super()._tracker_probe(pad, info)

    def _v99_prune(self, now: float) -> None:
        # Bound diagnostic dictionaries; normal tracker latency is far below this.
        cutoff = now - 5.0
        stale_gate = [key for key, value in self._v99_gate_mono.items() if value < cutoff]
        stale_mux = [key for key, value in self._v99_mux_mono.items() if value < cutoff]
        for key in stale_gate:
            self._v99_gate_mono.pop(key, None)
        for key in stale_mux:
            self._v99_mux_mono.pop(key, None)

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._v99_lock:
            snapshots = [
                (
                    source_id,
                    self.index_camera[source_id],
                    list(self._v99_source_gate_ms[source_id]),
                    list(self._v99_gate_mux_ms[source_id]),
                    list(self._v99_source_mux_ms[source_id]),
                    list(self._v99_mux_nvdcf_ms[source_id]),
                )
                for source_id in sorted(self.index_camera)
            ]
        for _source_id, cid, s_g, g_m, s_m, m_n in snapshots:
            print(
                "CAMERA_V99_TRACK_STAGE "
                f"camera={cid} "
                f"source_gate_p50={self._pct(s_g, 0.50):.0f}ms source_gate_p95={self._pct(s_g, 0.95):.0f}ms "
                f"gate_mux_p50={self._pct(g_m, 0.50):.0f}ms gate_mux_p95={self._pct(g_m, 0.95):.0f}ms "
                f"source_mux_p95={self._pct(s_m, 0.95):.0f}ms "
                f"mux_nvdcf_p50={self._pct(m_n, 0.50):.0f}ms mux_nvdcf_p95={self._pct(m_n, 0.95):.0f}ms "
                f"samples=sg:{len(s_g)},gm:{len(g_m)},mn:{len(m_n)}",
                flush=True,
            )
        return keep


def main() -> int:
    return PascalTrackerLatencyAuditRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
