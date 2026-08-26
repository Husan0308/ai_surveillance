from __future__ import annotations

import os
import signal
import time
from collections import deque

from services.ml_service.app.detector_substream import DetectorSubstreamService, _percentile
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W, TRT86DetectorClient


class DetectorSubstreamLiveService(DetectorSubstreamService):
    """Live-source lifecycle + bounded one-frame capture pipeline.

    At most one next camera is armed while TensorRT works on the current camera.
    A prearmed camera is never allowed to head-of-line block the detector for the
    full RTSP capture timeout: after inference it gets only a small completion
    budget. If it misses that budget the request is cancelled and the scheduler
    advances to another due camera. There is still no historical frame queue.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prearm_horizon_ms = max(
            0.0, min(300.0, float(os.environ.get("ML_SUBSTREAM_PREARM_HORIZON_MS", "180")))
        )
        self.prearm_block_budget_ms = max(
            0.0, min(100.0, float(os.environ.get("ML_SUBSTREAM_PREARM_BLOCK_MS", "40")))
        )
        self.capture_latency_ms: deque[float] = deque(maxlen=240)
        self.capture_block_ms: deque[float] = deque(maxlen=240)
        self.prearm_count = 0
        self.prearm_hits = 0
        self.prearm_skips = 0

    @staticmethod
    def _state_name(value) -> str:
        return str(getattr(value, "value_nick", value))

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        sink = self.pipeline.get_by_name(f"ml_sub_sink_{index}")
        if sink is None:
            raise RuntimeError(f"{camera.camera_id}: appsink missing after graph build")

        # Live RTSP sources do not preroll in PAUSED. Do not let appsink hold the
        # parent pipeline in an async preroll transition while sources are locked.
        self._set_if(sink, "async", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "processing-deadline", 0)
        self._set_if(sink, "max-lateness", -1)

    def _start_sources(self) -> None:
        for source in self.sources.values():
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)

        transition = self.pipeline.set_state(self.Gst.State.PLAYING)
        state_ret, current, pending = self.pipeline.get_state(0)
        print(
            "ML_SUBSTREAM_STATE "
            f"pipeline_target=PLAYING set_state={self._state_name(transition)} "
            f"query={self._state_name(state_ret)} current={self._state_name(current)} "
            f"pending={self._state_name(pending)} sink_async=0",
            flush=True,
        )

        for camera in self.cameras:
            source = self.sources[camera.camera_id]
            source.set_locked_state(False)
            synced = bool(source.sync_state_with_parent())
            state_ret, current, pending = source.get_state(0)
            print(
                "ML_SUBSTREAM_STATE "
                f"{camera.camera_id} sync_parent={int(synced)} "
                f"query={self._state_name(state_ret)} current={self._state_name(current)} "
                f"pending={self._state_name(pending)}",
                flush=True,
            )
            time.sleep(self.startup_stagger_sec)

    def _arm_capture(self, cid: str) -> tuple[int, float]:
        with self.capture_condition:
            baseline = self.capture_slots[cid].seq
            self.capture_requested[cid] = True
        return baseline, time.monotonic()

    def _cancel_capture(self, cid: str) -> None:
        with self.capture_condition:
            self.capture_requested[cid] = False

    def _await_armed_capture(
        self,
        cid: str,
        baseline: int,
        armed_at: float,
        *,
        block_budget_ms: float | None = None,
    ):
        block_started = time.monotonic()
        hard_deadline = armed_at + self.capture_timeout_ms / 1000.0
        deadline = hard_deadline
        if block_budget_ms is not None:
            deadline = min(deadline, block_started + max(0.0, block_budget_ms) / 1000.0)

        with self.capture_condition:
            while not self.stop_requested:
                slot = self.capture_slots[cid]
                if slot.seq > baseline and slot.frame is not None:
                    now = time.monotonic()
                    capture_latency = (now - armed_at) * 1000.0
                    blocked = (now - block_started) * 1000.0
                    return slot.seq, slot.captured_ns, slot.frame.copy(), capture_latency, blocked, False

                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    self.capture_requested[cid] = False
                    hard_timeout = now >= hard_deadline
                    if hard_timeout:
                        self.capture_timeouts[cid] += 1
                    capture_latency = (now - armed_at) * 1000.0
                    blocked = (now - block_started) * 1000.0
                    return baseline, 0, None, capture_latency, blocked, not hard_timeout
                self.capture_condition.wait(timeout=min(remaining, 0.01))
        return baseline, 0, None, 0.0, 0.0, False

    def _select_due(self, cursor: int, horizon_ms: float = 0.0):
        if not self.cameras:
            return None
        now = time.monotonic()
        horizon = max(0.0, horizon_ms) / 1000.0
        for offset in range(len(self.cameras)):
            index = (cursor + offset) % len(self.cameras)
            cid = self.cameras[index].camera_id
            if self.next_due[cid] <= now + horizon:
                return index, cid
        return None

    def _print_stats(self) -> None:
        super()._print_stats()
        hit_rate = 100.0 * self.prearm_hits / max(1, self.prearm_count)
        print(
            "ML_DETECTOR_PIPELINE_STATS "
            f"prearm={self.prearm_count} hit_rate={hit_rate:.1f}% skips={self.prearm_skips} "
            f"capture_latency_p95={_percentile(self.capture_latency_ms, 0.95):.1f}ms "
            f"capture_block_p95={_percentile(self.capture_block_ms, 0.95):.1f}ms "
            f"horizon={self.prearm_horizon_ms:.0f}ms "
            f"block_budget={self.prearm_block_budget_ms:.0f}ms queue_depth=1",
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
            "tracker=0 api=0 ui=0 substream_nvdec=1 JIT_convert=1 "
            "capture_prearm=1 head_of_line_block=bounded queue_depth=1",
            flush=True,
        )
        self._start_sources()
        self.detector = TRT86DetectorClient()

        cursor = 0
        # cid, baseline, armed_at, was_prearmed
        pending_capture: tuple[str, int, float, bool] | None = None

        while not self.stop_requested:
            self._poll_bus()

            if pending_capture is None:
                selected = self._select_due(cursor, 0.0)
                if selected is None:
                    if time.monotonic() - self.stats_at >= 5.0:
                        self._print_stats()
                    time.sleep(0.002)
                    continue
                index, cid = selected
                baseline, armed_at = self._arm_capture(cid)
                pending_capture = (cid, baseline, armed_at, False)
                cursor = (index + 1) % len(self.cameras)

            cid, baseline, armed_at, was_prearmed = pending_capture
            pending_capture = None
            budget = self.prearm_block_budget_ms if was_prearmed else None
            seq, captured_ns, frame, capture_latency, capture_block, soft_skip = self._await_armed_capture(
                cid,
                baseline,
                armed_at,
                block_budget_ms=budget,
            )
            self.capture_latency_ms.append(capture_latency)
            self.capture_block_ms.append(capture_block)
            self.capture_wait_ms.append(capture_block)

            if frame is None:
                if soft_skip:
                    self.prearm_skips += 1
                    # The camera is still due. Move on instead of letting one slow
                    # substream stall all other cameras; the round-robin cursor will
                    # retry it on a later pass.
                    self.next_due[cid] = min(self.next_due[cid], time.monotonic())
                else:
                    # A genuine source timeout gets a short retry delay, not a full
                    # target period, so recovery is prompt without a busy loop.
                    self.next_due[cid] = time.monotonic() + 0.05
                continue

            # Advance cadence only after a successful fresh capture.
            self.next_due[cid] = max(self.next_due[cid] + self.target_period, armed_at + self.target_period)

            selected = self._select_due(cursor, self.prearm_horizon_ms)
            if selected is not None:
                index, next_cid = selected
                next_baseline, next_armed_at = self._arm_capture(next_cid)
                pending_capture = (next_cid, next_baseline, next_armed_at, True)
                cursor = (index + 1) % len(self.cameras)
                self.prearm_count += 1

            input_age = max(0.0, (time.monotonic_ns() - captured_ns) / 1_000_000.0)
            self.input_age_ms.append(input_age)
            result = self.detector.infer(frame, self.conf, self.max_det)
            result_age = max(0.0, (time.monotonic_ns() - captured_ns) / 1_000_000.0)
            self.processed[cid] += 1
            self.box_counts[cid] += len(result.boxes)
            self.infer_ms.append(result.roundtrip_ms)
            self.result_age_ms.append(result_age)

            if pending_capture is not None:
                next_slot = self.capture_slots[pending_capture[0]]
                if next_slot.seq > pending_capture[1]:
                    self.prearm_hits += 1

            n = sum(self.processed.values())
            if n <= 3 or n % 20 == 0:
                best = max((row[4] for row in result.boxes), default=0.0)
                print(
                    "ML_DETECTOR_TRT "
                    f"n={n} camera={cid} frame_seq={seq} capture_wait={capture_block:.1f}ms "
                    f"capture_latency={capture_latency:.1f}ms input_age={input_age:.1f}ms "
                    f"roundtrip={result.roundtrip_ms:.1f}ms prep={result.prep_ms:.1f}ms "
                    f"trt={result.trt_ms:.1f}ms result_age={result_age:.1f}ms "
                    f"boxes={len(result.boxes)} best={best:.3f}",
                    flush=True,
                )
            if time.monotonic() - self.stats_at >= 5.0:
                self._print_stats()

        return 0


def main() -> int:
    service = DetectorSubstreamLiveService()

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
