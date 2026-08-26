from __future__ import annotations

import os
import signal
import threading
import time

from services.ml_service.app.detector_substream import DetectorSubstreamService
from services.ml_service.app.detector_substream_prequeue_demand import (
    DetectorSubstreamPrequeueDemandService,
)


class DetectorSubstreamPrequeueTokenService(DetectorSubstreamPrequeueDemandService):
    """Bounded wall-clock token bucket in front of the sparse GPU conversion path.

    CAM-02 arrives in bursts: several decoded frames may arrive together after a
    gap longer than the 500 ms detector period. A one-bit demand latch loses those
    elapsed deadlines by coalescing them. This service keeps a tiny bounded token
    bucket per camera instead. One token is generated per wall-clock detector
    deadline, and one arriving decoded frame consumes one token.

    Capacity is deliberately small (default 2): it can recover one missed 500 ms
    deadline from a short RTSP burst without allowing an unbounded catch-up storm.
    The gate remains on input_q:sink, before the leaky queue and nvvideoconvert.
    Only token-bearing frames enter the GPU conversion path.
    """

    def __init__(self) -> None:
        self.token_capacity = max(
            1, min(4, int(os.environ.get("ML_SUBSTREAM_TOKEN_CAPACITY", "2")))
        )
        super().__init__()
        self.gate_tokens = {camera.camera_id: 0 for camera in self.cameras}
        self.gate_scheduled = {camera.camera_id: 0 for camera in self.cameras}
        self.gate_scheduled_last = {camera.camera_id: 0 for camera in self.cameras}
        self.gate_token_overflow = {camera.camera_id: 0 for camera in self.cameras}
        self.gate_token_overflow_last = {camera.camera_id: 0 for camera in self.cameras}

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)

        # After the prequeue gate only sparse accepted frames remain. Allow a short
        # two-frame catch-up burst to survive both queues instead of immediately
        # collapsing back to a one-frame leaky queue.
        input_q = self.pipeline.get_by_name(f"ml_sub_input_q_{index}")
        output_q = self.pipeline.get_by_name(f"ml_sub_output_q_{index}")
        sink = self.pipeline.get_by_name(f"ml_sub_sink_{index}")
        if input_q is not None:
            self._set_if(input_q, "max-size-buffers", self.token_capacity)
        if output_q is not None:
            self._set_if(output_q, "max-size-buffers", self.token_capacity)
        if sink is not None:
            self._set_if(sink, "max-buffers", self.token_capacity)

        print(
            f"ML_SUBSTREAM_TOKEN_QUEUE {camera.camera_id} "
            f"accepted_queue_depth={self.token_capacity} gate_before_queue=1",
            flush=True,
        )

    def _enable_paced_gate(self) -> None:
        base = time.monotonic() + 0.05
        phase = self.target_period / max(1, len(self.cameras))
        with self.capture_condition:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                self.gate_next_wall[cid] = base + index * phase
                self.capture_requested[cid] = False
                self.gate_tokens[cid] = 0
            self.gate_enabled = True

        self.demand_thread = threading.Thread(
            target=self._demand_loop,
            name="ml-substream-token-bucket-scheduler",
            daemon=True,
        )
        self.demand_thread.start()
        print(
            "ML_DETECTOR_TOKEN_GATE "
            f"target={self.target_hz:.2f}Hz/cam phase_spacing={phase * 1000.0:.1f}ms "
            f"demand_poll={self.demand_poll_ms:.1f}ms token_capacity={self.token_capacity} "
            f"gate_before_convert=1 pending_depth={self.pending_depth} "
            "blocking_capture_wait=0 pace_clock=wall-token-bucket backlog=bounded-catchup",
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
                    self.gate_scheduled[cid] += steps

                    room = max(0, self.token_capacity - self.gate_tokens[cid])
                    credited = min(steps, room)
                    overflow = steps - credited
                    if credited:
                        self.gate_tokens[cid] += credited
                        self.gate_demands[cid] += credited
                    if overflow:
                        self.gate_token_overflow[cid] += overflow
                        # Keep the old counter meaningful for compatibility: these
                        # are deadlines that could not be retained by the bucket.
                        self.gate_coalesced[cid] += overflow

            time.sleep(poll_sec)

    def _capture_gate_probe(self, _pad, _info, cid: str):
        if not self.gate_enabled:
            return self.Gst.PadProbeReturn.DROP

        with self.capture_condition:
            if self.gate_tokens[cid] <= 0:
                return self.Gst.PadProbeReturn.DROP
            self.gate_tokens[cid] -= 1
            self.gate_passed[cid] += 1
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> None:
        now = time.monotonic()
        elapsed = max(1e-6, now - self.stats_at)
        scheduled_rows = []
        credit_rows = []
        gate_rows = []
        overflow_rows = []
        token_rows = []
        replaced_rows = []
        pending_rows = []

        for camera in self.cameras:
            cid = camera.camera_id

            count = self.gate_scheduled[cid]
            delta = count - self.gate_scheduled_last[cid]
            self.gate_scheduled_last[cid] = count
            scheduled_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")

            count = self.gate_demands[cid]
            delta = count - self.gate_demands_last[cid]
            self.gate_demands_last[cid] = count
            credit_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")

            count = self.gate_passed[cid]
            delta = count - self.gate_passed_last[cid]
            self.gate_passed_last[cid] = count
            gate_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")

            count = self.gate_token_overflow[cid]
            delta = count - self.gate_token_overflow_last[cid]
            self.gate_token_overflow_last[cid] = count
            overflow_rows.append(f"{cid}:{delta}")
            token_rows.append(f"{cid}:{self.gate_tokens[cid]}")

            count = self.pending_replaced[cid]
            delta = count - self.pending_replaced_last[cid]
            self.pending_replaced_last[cid] = count
            replaced_rows.append(f"{cid}:{delta}")
            pending_rows.append(f"{cid}:{len(self.pending_frames[cid])}")

        DetectorSubstreamService._print_stats(self)
        print(
            "ML_DETECTOR_TOKEN_STATS "
            f"scheduled=[{' '.join(scheduled_rows)}] credit=[{' '.join(credit_rows)}] "
            f"gate=[{' '.join(gate_rows)}] overflow=[{' '.join(overflow_rows)}] "
            f"tokens=[{' '.join(token_rows)}] replaced=[{' '.join(replaced_rows)}] "
            f"pending=[{' '.join(pending_rows)}] phase_spacing={self.phase_spacing_ms:.1f}ms "
            f"stale_drops={self.stale_drops} max_input_age={self.max_input_age_ms:.0f}ms "
            f"pending_depth={self.pending_depth} token_capacity={self.token_capacity} "
            "capture_block_p95=0.0ms pace_clock=wall-token-bucket backlog=bounded-catchup",
            flush=True,
        )


def main() -> int:
    service = DetectorSubstreamPrequeueTokenService()

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
