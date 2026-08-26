from __future__ import annotations

import os
import signal
import threading
import time
from collections import deque

from services.ml_service.app.detector_substream import DetectorSubstreamService
from services.ml_service.app.detector_substream_paced import DetectorSubstreamPacedService
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W, TRT86DetectorClient


class DetectorSubstreamBurstService(DetectorSubstreamPacedService):
    """Burst-tolerant sparse capture for fixed/locked camera configurations.

    V8 allowed only one unprocessed frame per camera. With a bursty 10 FPS RTSP
    source, a frame could arrive late, then wait behind the other five cameras'
    TensorRT work. If that wait crossed the next 500 ms deadline, V8 coalesced the
    next capture request and the camera fell below 2 Hz even though the source had
    enough frames.

    V9 separates capture cadence from inference consumption. Each camera may keep
    a tiny application-side pending deque (default depth 2). A new wall-clock
    demand is blocked only by an already-armed RTSP capture request, not by an
    older frame waiting for TensorRT. If the tiny deque is full, the oldest frame
    is replaced by the newest one. The GStreamer path remains sparse: only demanded
    frames pass the pre-nvvideoconvert gate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pending_depth = max(
            2, min(4, int(os.environ.get("ML_SUBSTREAM_PENDING_DEPTH", "2")))
        )
        self.pending_frames = {
            camera.camera_id: deque(maxlen=self.pending_depth) for camera in self.cameras
        }
        self.last_enqueued_seq = {camera.camera_id: 0 for camera in self.cameras}
        self.pending_replaced = {camera.camera_id: 0 for camera in self.cameras}
        self.pending_replaced_last = {camera.camera_id: 0 for camera in self.cameras}

    def _enable_paced_gate(self) -> None:
        base = time.monotonic() + 0.05
        phase = self.target_period / max(1, len(self.cameras))
        with self.capture_condition:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                self.gate_next_wall[cid] = base + index * phase
                self.capture_requested[cid] = False
            self.gate_enabled = True

        self.demand_thread = threading.Thread(
            target=self._demand_loop,
            name="ml-substream-burst-demand-scheduler",
            daemon=True,
        )
        self.demand_thread.start()
        print(
            "ML_DETECTOR_BURST_GATE "
            f"target={self.target_hz:.2f}Hz/cam phase_spacing={phase * 1000.0:.1f}ms "
            f"demand_poll={self.demand_poll_ms:.1f}ms gate_before_convert=1 "
            f"pending_depth={self.pending_depth} blocking_capture_wait=0 "
            "pace_clock=wall-demand-latched backlog=latest-replace",
            flush=True,
        )

    def _demand_loop(self) -> None:
        poll_sec = self.demand_poll_ms / 1000.0
        while not self.stop_requested:
            now = time.monotonic()
            with self.capture_condition:
                for camera in self.cameras:
                    cid = camera.camera_id
                    due = self.gate_next_wall[cid]
                    if now + 1e-9 < due:
                        continue

                    late = max(0.0, now - due)
                    steps = max(1, int(late // self.target_period) + 1)
                    self.gate_next_wall[cid] = due + steps * self.target_period

                    # Critical V9 difference: an older frame waiting for TensorRT
                    # does not suppress the next 2 Hz capture demand. Only a demand
                    # that is already armed and still waiting for an RTSP frame can
                    # coalesce another deadline.
                    if self.capture_requested[cid]:
                        self.gate_coalesced[cid] += steps
                        continue

                    self.capture_requested[cid] = True
                    self.gate_demands[cid] += 1
                    if steps > 1:
                        self.gate_coalesced[cid] += steps - 1

            time.sleep(poll_sec)

    def _on_sample(self, sink, cid: str):
        flow = DetectorSubstreamService._on_sample(self, sink, cid)
        with self.capture_condition:
            slot = self.capture_slots[cid]
            if slot.frame is None or slot.seq <= self.last_enqueued_seq[cid]:
                return flow

            queue = self.pending_frames[cid]
            if len(queue) >= self.pending_depth:
                queue.popleft()
                self.pending_replaced[cid] += 1

            # The numpy frame object is newly allocated for every appsink sample;
            # retaining this reference is safe because the base slot is replaced,
            # not mutated, on the next sample.
            queue.append((slot.captured_ns, slot.seq, slot.frame))
            self.last_enqueued_seq[cid] = slot.seq
            self.capture_condition.notify_all()
        return flow

    def _take_oldest_ready(self):
        ready = None
        with self.capture_condition:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                queue = self.pending_frames[cid]
                if not queue:
                    continue
                captured_ns, seq, frame = queue[0]
                candidate = (captured_ns, index, cid, seq, frame)
                if ready is None or candidate[:2] < ready[:2]:
                    ready = candidate

            if ready is None:
                return None

            captured_ns, index, cid, seq, frame = ready
            self.pending_frames[cid].popleft()
            self.processed_seq[cid] = seq
            return index, cid, seq, captured_ns, frame.copy()

    def _print_stats(self) -> None:
        now = time.monotonic()
        elapsed = max(1e-6, now - self.stats_at)
        demand_rows = []
        gate_rows = []
        coalesced_rows = []
        replaced_rows = []
        pending_rows = []

        for camera in self.cameras:
            cid = camera.camera_id

            count = self.gate_demands[cid]
            delta = count - self.gate_demands_last[cid]
            self.gate_demands_last[cid] = count
            demand_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")

            count = self.gate_passed[cid]
            delta = count - self.gate_passed_last[cid]
            self.gate_passed_last[cid] = count
            gate_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")

            count = self.gate_coalesced[cid]
            delta = count - self.gate_coalesced_last[cid]
            self.gate_coalesced_last[cid] = count
            coalesced_rows.append(f"{cid}:{delta}")

            count = self.pending_replaced[cid]
            delta = count - self.pending_replaced_last[cid]
            self.pending_replaced_last[cid] = count
            replaced_rows.append(f"{cid}:{delta}")
            pending_rows.append(f"{cid}:{len(self.pending_frames[cid])}")

        DetectorSubstreamService._print_stats(self)
        print(
            "ML_DETECTOR_BURST_STATS "
            f"demand=[{' '.join(demand_rows)}] gate=[{' '.join(gate_rows)}] "
            f"coalesced=[{' '.join(coalesced_rows)}] replaced=[{' '.join(replaced_rows)}] "
            f"pending=[{' '.join(pending_rows)}] phase_spacing={self.phase_spacing_ms:.1f}ms "
            f"stale_drops={self.stale_drops} max_input_age={self.max_input_age_ms:.0f}ms "
            f"pending_depth={self.pending_depth} capture_block_p95=0.0ms "
            "pace_clock=wall-demand-latched backlog=latest-replace",
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
            f"scheduler=burst-buffered-ready-first pending_depth={self.pending_depth} "
            "blocking_capture_wait=0",
            flush=True,
        )

        self._start_sources()
        self.detector = TRT86DetectorClient()
        self._enable_paced_gate()

        try:
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
        finally:
            self.stop_requested = True
            self._stop_demand_scheduler()

        return 0


def main() -> int:
    service = DetectorSubstreamBurstService()

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
