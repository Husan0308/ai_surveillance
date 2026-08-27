from __future__ import annotations

import signal
import time

from .step2_production_fp32_v12 import V11Step2ProductionFP32V12


class V11Step2ProductionFP32V13(V11Step2ProductionFP32V12):
    """V12 plus bounded demand credits for bursty CAM-02 wall cadence.

    Credits are capture opportunities, not frames. At most two elapsed 2 Hz
    deadlines may be retained, while GStreamer still owns exactly one pending
    sample. A TCP burst consumes credits with current decoded frames; if two arrive
    before the detector pulls, appsink drops the older sample and keeps the latest.
    """

    def __init__(self) -> None:
        self.credit_capacity = 2
        self.credits: dict[str, int] = {}
        self.credit_overflow: dict[str, int] = {}
        super().__init__()
        for camera in self.cameras:
            cid = camera.camera_id
            self.credits[cid] = 0
            self.credit_overflow[cid] = 0
        print(
            "CAMERA_V11_STEP2_V13_CREDIT capacity=2 unit=capture-deadline "
            "frame_queue=0 appsink_pending=1 burst_policy=latest-overwrite",
            flush=True,
        )

    def _enable_demands(self) -> None:
        base = time.monotonic() + 0.05
        phase = (1.0 / self.target_hz) / len(self.cameras)
        with self.lock:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                self.next_due[cid] = base + index * phase
                self.requested[cid] = False
                self.credits[cid] = 0
            self.gate_enabled = True
        self.demand_thread = __import__("threading").Thread(
            target=self._demand_loop,
            name="camera-v11-step2-credit-scheduler",
            daemon=True,
        )
        self.demand_thread.start()
        print(
            "CAMERA_V11_STEP2_V13_DEMAND "
            f"target={self.target_hz:.2f}Hz/camera phase={phase * 1000.0:.1f}ms "
            f"credit_capacity={self.credit_capacity} frame_pending_max=1",
            flush=True,
        )

    def _demand_loop(self) -> None:
        period = 1.0 / self.target_hz
        while not self.stop_requested:
            now = time.monotonic()
            with self.lock:
                if not self.gate_enabled:
                    break
                for camera in self.cameras:
                    cid = camera.camera_id
                    due = self.next_due[cid]
                    if now < due:
                        continue
                    elapsed = max(0.0, now - due)
                    steps = max(1, int(elapsed // period) + 1)
                    self.next_due[cid] = due + steps * period
                    self.demands[cid] += steps
                    room = max(0, self.credit_capacity - self.credits[cid])
                    credited = min(room, steps)
                    self.credits[cid] += credited
                    self.credit_overflow[cid] += steps - credited
            time.sleep(0.001)

    def _capture_gate_probe(self, _pad, info, cid: str):
        with self.lock:
            stat = self.stats[cid]
            if not self.gate_enabled or self.credits.get(cid, 0) <= 0:
                stat.gate_drops += 1
                return self.Gst.PadProbeReturn.DROP
            self.credits[cid] -= 1
            stat.accepted += 1
            self.accepted_seq[cid] += 1
            self.accepted_ns[cid] = time.monotonic_ns()
            buffer = info.get_buffer()
            self.accepted_pts_ns[cid] = (
                int(buffer.pts)
                if buffer is not None and buffer.pts != self.Gst.CLOCK_TIME_NONE
                else -1
            )
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> None:
        super()._print_stats()
        with self.lock:
            rows = [
                f"{camera.camera_id}:credit={self.credits[camera.camera_id]},"
                f"overflow={self.credit_overflow[camera.camera_id]}"
                for camera in self.cameras
            ]
        print(
            "CAMERA_V11_STEP2_V13_CREDIT_STATS "
            + " | ".join(rows)
            + f" capacity={self.credit_capacity} frame_pending_max=1",
            flush=True,
        )


def main() -> int:
    service = V11Step2ProductionFP32V13()

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
