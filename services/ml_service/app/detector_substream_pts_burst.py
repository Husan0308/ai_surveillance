from __future__ import annotations

import signal
import time

from services.ml_service.app.detector_substream import DetectorSubstreamService
from services.ml_service.app.detector_substream_burst import DetectorSubstreamBurstService
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W, TRT86DetectorClient


class DetectorSubstreamPtsBurstService(DetectorSubstreamBurstService):
    """PTS-paced sparse capture with a tiny burst-safe pending queue.

    V7 proved that CAM-02's media timeline can sustain the requested cadence even
    when TCP delivery is bursty, but its single latest slot could overwrite accepted
    frames before TensorRT consumed them. V9 added a bounded pending deque, but kept
    a wall-clock boolean demand; a demand that stayed armed across a delivery gap
    coalesced the next 500 ms deadline and CAM-02 still fell below 2 Hz.

    V10 combines the two useful pieces: gate cadence is driven by decoded-buffer PTS
    (media time), while accepted frames enter V9's bounded per-camera pending deque.
    Thus a TCP burst may deliver two media-time-spaced accepted frames close together
    in wall time without losing either. Only PTS-selected frames pass the gate before
    nvvideoconvert, so GPU preprocessing remains sparse and bounded.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate_start_wall = {camera.camera_id: 0.0 for camera in self.cameras}
        self.gate_next_pts_ns: dict[str, int | None] = {
            camera.camera_id: None for camera in self.cameras
        }
        self.gate_last_pts_ns: dict[str, int | None] = {
            camera.camera_id: None for camera in self.cameras
        }
        self.gate_period_ns = max(1, int(round(self.target_period * 1_000_000_000.0)))
        self.gate_pts_passes = {camera.camera_id: 0 for camera in self.cameras}
        self.gate_wall_fallback_passes = {camera.camera_id: 0 for camera in self.cameras}
        self.gate_pts_resets = {camera.camera_id: 0 for camera in self.cameras}

    def _enable_paced_gate(self) -> None:
        # Keep only the initial wall-time phase staggering. Once each camera starts,
        # its cadence follows the decoded buffer PTS/media timeline, not callback
        # arrival time. No demand thread is started in this mode.
        base = time.monotonic() + 0.05
        phase = self.target_period / max(1, len(self.cameras))
        with self.capture_condition:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                start = base + index * phase
                self.gate_start_wall[cid] = start
                self.gate_next_wall[cid] = start
                self.gate_next_pts_ns[cid] = None
                self.gate_last_pts_ns[cid] = None
                self.capture_requested[cid] = False
            self.gate_enabled = True

        print(
            "ML_DETECTOR_PTS_BURST_GATE "
            f"target={self.target_hz:.2f}Hz/cam phase_spacing={phase * 1000.0:.1f}ms "
            f"gate_before_convert=1 pending_depth={self.pending_depth} blocking_capture_wait=0 "
            "pace_clock=buffer-pts fallback=monotonic backlog=bounded-pending",
            flush=True,
        )

    def _wall_fallback_gate(self, cid: str, now: float):
        due = self.gate_next_wall[cid]
        if now + 1e-9 < due:
            return self.Gst.PadProbeReturn.DROP
        late = max(0.0, now - due)
        steps = max(1, int(late // self.target_period) + 1)
        self.gate_next_wall[cid] = due + steps * self.target_period
        self.gate_passed[cid] += 1
        self.gate_wall_fallback_passes[cid] += 1
        return self.Gst.PadProbeReturn.OK

    def _capture_gate_probe(self, _pad, info, cid: str):
        if not self.gate_enabled:
            return self.Gst.PadProbeReturn.DROP

        now = time.monotonic()
        if now + 1e-9 < self.gate_start_wall[cid]:
            return self.Gst.PadProbeReturn.DROP

        buffer = info.get_buffer()
        pts_ns: int | None = None
        if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE:
            pts_ns = int(buffer.pts)

        if pts_ns is None:
            return self._wall_fallback_gate(cid, now)

        last_pts = self.gate_last_pts_ns[cid]
        if last_pts is not None and pts_ns < last_pts:
            # Reconnect/discontinuity: rebase on this camera's new media timeline.
            self.gate_next_pts_ns[cid] = None
            self.gate_pts_resets[cid] += 1
        self.gate_last_pts_ns[cid] = pts_ns

        due_pts = self.gate_next_pts_ns[cid]
        if due_pts is None:
            self.gate_next_pts_ns[cid] = pts_ns + self.gate_period_ns
            self.gate_passed[cid] += 1
            self.gate_pts_passes[cid] += 1
            return self.Gst.PadProbeReturn.OK

        if pts_ns < due_pts:
            return self.Gst.PadProbeReturn.DROP

        # Advance by whole media-time periods. If delivery is bursty, later buffers
        # in the same wall-time burst can satisfy later PTS deadlines; V9's bounded
        # pending deque keeps those accepted frames distinct until TRT consumes them.
        late_ns = max(0, pts_ns - due_pts)
        steps = max(1, int(late_ns // self.gate_period_ns) + 1)
        self.gate_next_pts_ns[cid] = due_pts + steps * self.gate_period_ns
        self.gate_passed[cid] += 1
        self.gate_pts_passes[cid] += 1
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> None:
        now = time.monotonic()
        elapsed = max(1e-6, now - self.stats_at)
        gate_rows = []
        replaced_rows = []
        pending_rows = []

        for camera in self.cameras:
            cid = camera.camera_id

            count = self.gate_passed[cid]
            delta = count - self.gate_passed_last[cid]
            self.gate_passed_last[cid] = count
            gate_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")

            count = self.pending_replaced[cid]
            delta = count - self.pending_replaced_last[cid]
            self.pending_replaced_last[cid] = count
            replaced_rows.append(f"{cid}:{delta}")
            pending_rows.append(f"{cid}:{len(self.pending_frames[cid])}")

        DetectorSubstreamService._print_stats(self)
        print(
            "ML_DETECTOR_PTS_BURST_STATS "
            f"gate=[{' '.join(gate_rows)}] replaced=[{' '.join(replaced_rows)}] "
            f"pending=[{' '.join(pending_rows)}] phase_spacing={self.phase_spacing_ms:.1f}ms "
            f"stale_drops={self.stale_drops} max_input_age={self.max_input_age_ms:.0f}ms "
            f"pending_depth={self.pending_depth} "
            f"pts_passes={sum(self.gate_pts_passes.values())} "
            f"wall_fallback={sum(self.gate_wall_fallback_passes.values())} "
            f"pts_resets={sum(self.gate_pts_resets.values())} "
            "capture_block_p95=0.0ms pace_clock=buffer-pts backlog=bounded-pending",
            flush=True,
        )

    def run(self) -> int:
        print(
            "ML_DETECTOR_PROFILE "
            f"source=Hikvision-substream-direct cameras={len(self.cameras)} "
            f"capture={INPUT_W}x{CONTENT_H} target={self.target_hz:.2f}Hz/cam "
            f"rtsp={self.rtsp_latency_ms}ms extra_surfaces={self.extra_surfaces} "
            f"conf={self.conf:.2f} max_det={self.max_det}",
            flush=True,
        )
        print(
            "ML_DETECTOR_BOUNDARY camera_service=independent main_stream=0 camera_shm=0 "
            "tracker=0 api=0 ui=0 substream_nvdec=1 sparse_convert=1 "
            f"scheduler=pts-burst-ready-first pending_depth={self.pending_depth} "
            "blocking_capture_wait=0",
            flush=True,
        )

        self._start_sources()
        self.detector = TRT86DetectorClient()
        self._enable_paced_gate()

        while not self.stop_requested:
            self._poll_bus()
            item = self._take_oldest_ready()
            if item is None:
                if time.monotonic() - self.stats_at >= 5.0:
                    self._print_stats()
                time.sleep(0.001)
                continue

            _index, cid, seq, captured_ns, frame = item
            input_age = max(0.0, (time.monotonic_ns() - captured_ns) / 1_000_000.0)
            if input_age > self.max_input_age_ms:
                self.stale_drops += 1
                continue

            self.input_age_ms.append(input_age)
            result = self.detector.infer(frame, self.conf, self.max_det)
            result_age = max(0.0, (time.monotonic_ns() - captured_ns) / 1_000_000.0)
            self.processed[cid] += 1
            self.box_counts[cid] += len(result.boxes)
            self.infer_ms.append(result.roundtrip_ms)
            self.result_age_ms.append(result_age)

            n = sum(self.processed.values())
            if n <= 3 or n % 20 == 0:
                best = max((row[4] for row in result.boxes), default=0.0)
                print(
                    "ML_DETECTOR_TRT "
                    f"n={n} camera={cid} frame_seq={seq} capture_wait=0.0ms "
                    f"input_age={input_age:.1f}ms roundtrip={result.roundtrip_ms:.1f}ms "
                    f"prep={result.prep_ms:.1f}ms trt={result.trt_ms:.1f}ms "
                    f"result_age={result_age:.1f}ms boxes={len(result.boxes)} best={best:.3f}",
                    flush=True,
                )

            if time.monotonic() - self.stats_at >= 5.0:
                self._print_stats()

        return 0


def main() -> int:
    service = DetectorSubstreamPtsBurstService()

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
