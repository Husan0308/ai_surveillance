from __future__ import annotations

import threading
import time
from collections import Counter, deque

from .runtime_v101_tracker_batch_audit import PascalTrackerBatchAuditRuntime


class PascalTrackerPostMuxGateRuntime(PascalTrackerBatchAuditRuntime):
    """V10.4: feed fresh 20 FPS camera frames into nvstreammux, throttle after mux.

    V10.3 proved a shared pre-mux 10 Hz slot is still too sparse: accepted frame
    arrivals span most of the 100 ms slot, so a 40 ms mux timeout cannot reliably
    collect all six sources.  The mux therefore emits mostly partial batches and
    the tracker consumes stale frames.

    Keep display/detector/NvDCF/bbox behavior unchanged.  The tracker branch now
    allows every fresh source frame to reach tracker_mux.  tracker_mux can form a
    six-source batch from continuously arriving 20 FPS inputs, then a single
    post-mux gate admits at most one batch per 10 Hz slot to NvDCF.  A 60 ms mux
    timeout covers one 20 FPS frame period plus modest live-source jitter; full
    batches still leave immediately, so this is not a fixed 60 ms delay.
    """

    def __init__(self) -> None:
        self._v104_lock = threading.RLock()
        self._v104_raw_sizes: deque[int] = deque(maxlen=4096)
        self._v104_raw_unique: deque[int] = deque(maxlen=4096)
        self._v104_raw_dt_ms: deque[float] = deque(maxlen=4096)
        self._v104_raw_hits: Counter[int] = Counter()
        self._v104_raw_batches = 0
        self._v104_last_raw_mono: float | None = None
        self._v104_last_post_slot: int | None = None
        self._v104_selected = 0
        self._v104_dropped = 0
        self._v104_input_passes = 0
        super().__init__()

        # V10.0's constructor verifies 40 ms before returning.  Change the
        # property only after that guard has run and before PLAYING state.
        self._set_if(self.tracker_mux, "batched-push-timeout", 60_000)
        timeout_us = None
        batch_size = None
        try:
            timeout_us = int(self.tracker_mux.get_property("batched-push-timeout"))
        except Exception:
            pass
        try:
            batch_size = int(self.tracker_mux.get_property("batch-size"))
        except Exception:
            pass
        print(
            "CAMERA_V104_ARCH only_change=tracker-throttle-pre-mux-to-post-mux "
            f"tracker_batch_size={batch_size} target={len(self.cameras)} "
            f"tracker_timeout_us={timeout_us} tracker_hz={self.track_fps:.1f} "
            f"source_hz={self.source_fps} pre_mux_gate=off post_mux_gate=on sync_inputs=0",
            flush=True,
        )
        if batch_size is not None and batch_size != len(self.cameras):
            raise RuntimeError(
                f"V10.4 expected tracker_mux batch-size={len(self.cameras)}, got {batch_size}"
            )
        if timeout_us is not None and timeout_us != 60_000:
            raise RuntimeError(
                f"V10.4 expected tracker_mux batched-push-timeout=60000, got {timeout_us}"
            )

    def _tracker_rate_probe(self, _pad, info, cid: str):
        """Do not sparsify each source before nvstreammux.

        Preserve V9.9's exact-PTS upstream timestamp so selected post-mux frames
        still have meaningful source->mux latency accounting.
        """
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.DROP
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.DROP

        now = time.monotonic()
        with self.track_gate_lock:
            self.track_buffers_passed += 1
            self._v104_input_passes += 1

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

    def _inject_detector_probe(self, pad, info):
        """Audit every mux output, but only pass one batch per 10 Hz slot."""
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.DROP

        now = time.monotonic()
        rows = self._copy_pts(buffer)
        valid = [row for row in rows if self._valid_pts(int(row["buf_pts"]))]
        source_ids = [int(row["source_id"]) for row in valid]
        unique = set(source_ids)

        with self._v104_lock:
            self._v104_raw_batches += 1
            self._v104_raw_sizes.append(len(valid))
            self._v104_raw_unique.append(len(unique))
            for sid in unique:
                self._v104_raw_hits[sid] += 1
            if self._v104_last_raw_mono is not None:
                dt = (now - self._v104_last_raw_mono) * 1000.0
                if 0.0 <= dt <= 5000.0:
                    self._v104_raw_dt_ms.append(float(dt))
            self._v104_last_raw_mono = now

            period_s = 1.0 / max(1e-6, float(self.track_fps))
            slot = int(now / period_s)
            if self._v104_last_post_slot == slot:
                self._v104_dropped += 1
                return self.Gst.PadProbeReturn.DROP
            self._v104_last_post_slot = slot
            self._v104_selected += 1

        # Selected batches keep all V10.1/V9.9 instrumentation and detector
        # metadata injection, then continue into NvDCF.
        return super()._inject_detector_probe(pad, info)

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self._v104_lock:
            sizes = list(self._v104_raw_sizes)
            unique_sizes = list(self._v104_raw_unique)
            dts = list(self._v104_raw_dt_ms)
            hits = dict(self._v104_raw_hits)
            batches = int(self._v104_raw_batches)
            selected = int(self._v104_selected)
            dropped = int(self._v104_dropped)
            input_passes = int(self._v104_input_passes)

        if not sizes:
            print("CAMERA_V104_MUX samples=0", flush=True)
            return keep

        full = sum(1 for x in sizes if x >= len(self.cameras))
        full_pct = 100.0 * full / len(sizes)
        source_parts = []
        for source_id in sorted(self.index_camera):
            cid = self.index_camera[source_id]
            ratio = 100.0 * hits.get(source_id, 0) / max(1, len(sizes))
            source_parts.append(f"{cid}:{ratio:.0f}%")

        print(
            "CAMERA_V104_MUX "
            f"samples={batches} target={len(self.cameras)} input_passes={input_passes} "
            f"size_p50={self._pct(sizes, 0.50):.0f} size_p95={self._pct(sizes, 0.95):.0f} "
            f"unique_p50={self._pct(unique_sizes, 0.50):.0f} unique_p95={self._pct(unique_sizes, 0.95):.0f} "
            f"full_pct={full_pct:.1f} raw_dt_p50={self._pct(dts, 0.50):.0f}ms "
            f"raw_dt_p95={self._pct(dts, 0.95):.0f}ms selected={selected} dropped={dropped} "
            f"source_hit={'/'.join(source_parts)}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalTrackerPostMuxGateRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
