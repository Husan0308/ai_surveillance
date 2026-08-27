from __future__ import annotations

import signal
import time
from collections import deque

from .step2_production_fp32 import _pct
from .step2_production_fp32_v18 import V11Step2ProductionFP32V18
from .step2_trt86 import Step2TRT86Client
from .step3_tracker_v1 import V11PerCameraTrackerV1


class V11Step3TrackingV1(V11Step2ProductionFP32V18):
    """Frozen Step2 detector plus CPU-only per-camera local tracking metadata."""

    def __init__(self) -> None:
        super().__init__()
        if self.mode != "full":
            raise RuntimeError("Step3 V1 supports V11_STEP2_MODE=full only")
        camera_ids = [camera.camera_id for camera in self.cameras]
        self.tracker = V11PerCameraTrackerV1(camera_ids)
        self.stage_values["tracker"] = deque(maxlen=2048)
        self.track_updates = {cid: 0 for cid in camera_ids}
        self.track_created = {cid: 0 for cid in camera_ids}
        self.track_recovered = {cid: 0 for cid in camera_ids}
        self.track_removed = {cid: 0 for cid in camera_ids}
        self.latest_track_ids: dict[str, tuple[str, ...]] = {cid: () for cid in camera_ids}
        self.track_duplicate_errors = 0
        self.track_prefix_errors = 0
        print(
            "CAMERA_V11_STEP3_ARCH base=step2-frozen-v12 tracker=cpu-time-aware-bytetrack-style "
            "scope=per-camera appearance=0 reid=0 face=0 display_topology_changed=0 "
            "tracker_frame_queue=0 tracker_gpu=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP3_POLICY low=0.18 high=0.30 new=0.30 match=0.22 "
            "low_match=0.18 confirm_hits=2 shadow=0.9s max_lost=2.5s clock=capture-monotonic",
            flush=True,
        )

    def _consume_tracking(self, cid: str, boxes: list[list[float]], captured_ns: int) -> None:
        update = self.tracker.update(cid, boxes, captured_ns)
        self.stage_values["tracker"].append(float(update.step_ms))
        ids = tuple(snapshot.track_id for snapshot in update.snapshots)
        if len(ids) != len(set(ids)):
            self.track_duplicate_errors += 1
        prefix = f"{cid}-T"
        self.track_prefix_errors += sum(1 for track_id in ids if not track_id.startswith(prefix))
        self.track_updates[cid] += 1
        self.track_created[cid] += int(update.created)
        self.track_recovered[cid] += int(update.recovered)
        self.track_removed[cid] += int(update.removed)
        self.latest_track_ids[cid] = ids

    def _print_stats(self) -> None:
        super()._print_stats()
        tracker_values = self.stage_values["tracker"]
        rows = []
        for camera in self.cameras:
            cid = camera.camera_id
            ids = self.latest_track_ids[cid]
            rows.append(
                f"{cid}:updates={self.track_updates[cid]},created={self.track_created[cid]},"
                f"recovered={self.track_recovered[cid]},removed={self.track_removed[cid]},"
                f"visible={len(ids)},ids={','.join(ids) if ids else '-'}"
            )
        print(
            "CAMERA_V11_STEP3_TRACKER "
            + " | ".join(rows)
            + f" tracker_p50={_pct(tracker_values, 0.50):.3f}ms"
            + f" tracker_p95={_pct(tracker_values, 0.95):.3f}ms"
            + f" duplicate_errors={self.track_duplicate_errors}"
            + f" prefix_errors={self.track_prefix_errors}"
            + " appearance=0 reid=0 face=0",
            flush=True,
        )

    def run(self) -> int:
        self.detector = Step2TRT86Client()
        self._warmup()
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

                started = time.perf_counter()
                result = self.detector.infer_preloaded(self.conf, self.max_det)
                roundtrip = (time.perf_counter() - started) * 1000.0
                self.stage_values["ipc_roundtrip"].append(roundtrip)
                worker_total = float(result.stages.get("total_ms", 0.0))
                self.stage_values["cpu_ipc_block"].append(max(0.0, roundtrip - worker_total))
                self._append_worker_stages(result.stages)

                self._consume_tracking(cid, result.boxes, accepted_ns)

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
    service = V11Step3TrackingV1()

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
