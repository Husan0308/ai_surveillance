from __future__ import annotations

import signal
import threading
import time

import numpy as np

from .step2_production_fp32 import V11Step2ProductionFP32
from .step2_trt86 import Step2TRT86Client


class V11Step2ProductionFP32V12(V11Step2ProductionFP32):
    """Asynchronous one-slot scheduler for bursty detector substreams.

    The streaming probes only update a demand bit and timestamps. GStreamer owns
    the one pending sample per camera (`max-buffers=1, drop=true`); Python owns no
    frame queue. The consumer scans cameras round-robin and never waits for a
    particular camera, so CAM-02 gaps cannot stall the other five cameras.
    """

    def __init__(self) -> None:
        self.gate_enabled = False
        self.converted_ns: dict[str, int] = {}
        self.last_processed_seq: dict[str, int] = {}
        self.overwritten: dict[str, int] = {}
        self.demands: dict[str, int] = {}
        self.coalesced: dict[str, int] = {}
        self.next_due: dict[str, float] = {}
        self.demand_thread: threading.Thread | None = None
        super().__init__()
        for camera in self.cameras:
            cid = camera.camera_id
            self.converted_ns.setdefault(cid, 0)
            self.last_processed_seq.setdefault(cid, 0)
            self.overwritten.setdefault(cid, 0)
            self.demands.setdefault(cid, 0)
            self.coalesced.setdefault(cid, 0)
            self.next_due.setdefault(cid, 0.0)
        print(
            "CAMERA_V11_STEP2_V12_SCHEDULER mode=async-ready-round-robin "
            "streaming_callback=timestamp-only blocking_camera_wait=0 "
            "pending_owner=appsink pending_per_camera=1 overwrite_old=1 schedule_debt=0",
            flush=True,
        )

    def _build_camera(self, index, camera) -> None:
        super()._build_camera(index, camera)
        cid = camera.camera_id
        self.converted_ns.setdefault(cid, 0)
        output_src = self.output_queues[cid].get_static_pad("src")
        output_src.add_probe(self.Gst.PadProbeType.BUFFER, self._converted_probe, cid)

    def _converted_probe(self, _pad, _info, cid: str):
        # Intentionally lightweight: no mapping, allocation, conversion, or I/O.
        with self.lock:
            self.converted_ns[cid] = time.monotonic_ns()
        return self.Gst.PadProbeReturn.OK

    def _enable_demands(self) -> None:
        base = time.monotonic() + 0.05
        phase = (1.0 / self.target_hz) / len(self.cameras)
        with self.lock:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                self.next_due[cid] = base + index * phase
                self.requested[cid] = False
            self.gate_enabled = True
        self.demand_thread = threading.Thread(
            target=self._demand_loop,
            name="camera-v11-step2-demand-latches",
            daemon=True,
        )
        self.demand_thread.start()
        print(
            "CAMERA_V11_STEP2_V12_DEMAND "
            f"target={self.target_hz:.2f}Hz/camera phase={phase * 1000.0:.1f}ms "
            "clock=monotonic outstanding_max=1 missed_deadline=coalesce",
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
                    if self.requested[cid]:
                        self.coalesced[cid] += steps
                        continue
                    self.requested[cid] = True
                    self.demands[cid] += 1
                    if steps > 1:
                        self.coalesced[cid] += steps - 1
            time.sleep(0.001)

    def _capture_gate_probe(self, pad, info, cid: str):
        if not self.gate_enabled:
            with self.lock:
                self.stats[cid].gate_drops += 1
            return self.Gst.PadProbeReturn.DROP
        return super()._capture_gate_probe(pad, info, cid)

    def _copy_sample(self, cid: str, sample):
        copy_started = time.perf_counter()
        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        if (width, height) != (672, 378):
            raise RuntimeError(f"{cid}: detector sample is {width}x{height}")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError(f"{cid}: detector sample map failed")
        try:
            tight = width * 4
            size = int(getattr(mapped, "size", len(mapped.data)))
            stride = size // height if size % height == 0 else tight
            if stride < tight or size < stride * height:
                raise RuntimeError(f"{cid}: invalid BGRx stride={stride} size={size}")
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=stride * height)
            bgrx = raw.reshape(height, stride)[:, :tight].reshape(height, width, 4)
            target = self.detector.content if self.detector is not None else self._local_content
            np.copyto(target, bgrx[:, :, :3], casting="no")
        finally:
            buffer.unmap(mapped)
        return (time.perf_counter() - copy_started) * 1000.0

    def _take_ready(self, start_index: int):
        count = len(self.cameras)
        for offset in range(count):
            index = (start_index + offset) % count
            cid = self.cameras[index].camera_id
            sample = self.sinks[cid].emit("try-pull-sample", 0)
            if sample is None:
                continue
            taken_ns = time.monotonic_ns()
            with self.lock:
                sequence = self.accepted_seq[cid]
                accepted_ns = self.accepted_ns[cid]
                converted_ns = self.converted_ns.get(cid, taken_ns)
                previous = self.last_processed_seq[cid]
                if sequence > previous + 1:
                    self.overwritten[cid] += sequence - previous - 1
                self.last_processed_seq[cid] = sequence
            conversion_ms = max(0.0, (converted_ns - accepted_ns) / 1_000_000.0)
            schedule_wait_ms = max(0.0, (taken_ns - converted_ns) / 1_000_000.0)
            return index, cid, sequence, accepted_ns, sample, conversion_ms, schedule_wait_ms
        return None

    def _print_stats(self) -> None:
        super()._print_stats()
        with self.lock:
            rows = [
                f"{camera.camera_id}:demand={self.demands[camera.camera_id]},"
                f"accepted={self.accepted_seq[camera.camera_id]},"
                f"processed={self.stats[camera.camera_id].processed},"
                f"overwritten={self.overwritten[camera.camera_id]},"
                f"coalesced={self.coalesced[camera.camera_id]},"
                f"pending={int(self.accepted_seq[camera.camera_id] > self.last_processed_seq[camera.camera_id])}"
                for camera in self.cameras
            ]
        print(
            "CAMERA_V11_STEP2_V12_LATEST "
            + " | ".join(rows)
            + " pending_max=1 retry_old=0 fairness=round-robin",
            flush=True,
        )

    def run(self) -> int:
        needs_trt = self.mode in {"synthetic-trt", "full"}
        if needs_trt:
            self.detector = Step2TRT86Client()
            self._warmup()
        if self.mode == "synthetic-trt":
            return super().run()

        if self.mode in {"extraction", "preprocessing"}:
            self._local_content.fill(114)
        self._start_ingest()
        self._enable_demands()
        scan_index = 0

        try:
            while not self.stop_requested:
                self._poll_bus()
                item = self._take_ready(scan_index)
                if item is None:
                    if time.monotonic() - self.report_at >= 5.0:
                        self._print_stats()
                    time.sleep(0.001)
                    continue
                index, cid, _sequence, accepted_ns, sample, conversion_ms, schedule_wait_ms = item
                scan_index = (index + 1) % len(self.cameras)
                self.stage_values["schedule_wait"].append(schedule_wait_ms)
                self.stage_values["capture_wait"].append(0.0)
                self.stage_values["nvmm_resize_bgrx"].append(conversion_ms)
                self.stage_values["map_copy"].append(self._copy_sample(cid, sample))

                age_ms = max(0.0, (time.monotonic_ns() - accepted_ns) / 1_000_000.0)
                if age_ms > self.max_input_age_ms:
                    with self.lock:
                        self.stats[cid].stale_drops += 1
                    continue

                if self.mode == "preprocessing":
                    self.stage_values["preprocess"].append(self._local_preprocess())
                elif needs_trt:
                    started = time.perf_counter()
                    result = self.detector.infer_preloaded(self.conf, self.max_det)
                    roundtrip = (time.perf_counter() - started) * 1000.0
                    self.stage_values["ipc_roundtrip"].append(roundtrip)
                    worker_total = float(result.stages.get("total_ms", 0.0))
                    self.stage_values["cpu_ipc_block"].append(max(0.0, roundtrip - worker_total))
                    self._append_worker_stages(result.stages)

                result_age = max(0.0, (time.monotonic_ns() - accepted_ns) / 1_000_000.0)
                self.stage_values["result_age"].append(result_age)
                with self.lock:
                    self.stats[cid].processed += 1
                if time.monotonic() - self.report_at >= 5.0:
                    self._print_stats()
        finally:
            with self.lock:
                self.gate_enabled = False
            thread = self.demand_thread
            if thread is not None:
                thread.join(timeout=1.0)
                self.demand_thread = None
        return 0


def main() -> int:
    service = V11Step2ProductionFP32V12()

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
