from __future__ import annotations

import os
import signal
import time

from services.ml_service.app.detector_substream import DetectorSubstreamService, _percentile
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W, TRT86DetectorClient


class DetectorSubstreamPacedService(DetectorSubstreamService):
    """Sparse paced substream producer + nonblocking TensorRT consumer.

    Each camera passes only target_hz frames through nvvideoconvert. Camera phases
    are evenly staggered across the target period so six conversions do not burst
    together. Appsinks publish one latest frame per camera; the detector consumes
    whichever fresh slot is ready and never waits on a particular camera.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate_enabled = False
        self.gate_next = {camera.camera_id: 0.0 for camera in self.cameras}
        self.gate_passed = {camera.camera_id: 0 for camera in self.cameras}
        self.gate_passed_last = {camera.camera_id: 0 for camera in self.cameras}
        self.processed_seq = {camera.camera_id: 0 for camera in self.cameras}
        self.max_input_age_ms = max(
            50.0, min(500.0, float(os.environ.get("ML_SUBSTREAM_MAX_INPUT_AGE_MS", "180")))
        )
        self.stale_drops = 0
        self.phase_spacing_ms = 1000.0 * self.target_period / max(1, len(self.cameras))

    @staticmethod
    def _state_name(value) -> str:
        return str(getattr(value, "value_nick", value))

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        sink = self.pipeline.get_by_name(f"ml_sub_sink_{index}")
        if sink is None:
            raise RuntimeError(f"{camera.camera_id}: appsink missing after graph build")
        # Live RTSP sources do not preroll in PAUSED. Keep the capture sink out of
        # the async preroll/clock path; samples are handled immediately by signal.
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

    def _enable_paced_gate(self) -> None:
        # Start slightly in the future, then distribute the six camera phases over
        # one target period. At 2 Hz/camera this is ~83.3 ms between conversions.
        base = time.monotonic() + 0.05
        phase = self.target_period / max(1, len(self.cameras))
        for index, camera in enumerate(self.cameras):
            self.gate_next[camera.camera_id] = base + index * phase
        self.gate_enabled = True
        print(
            "ML_DETECTOR_PACED_GATE "
            f"target={self.target_hz:.2f}Hz/cam phase_spacing={phase * 1000.0:.1f}ms "
            "gate_before_convert=1 latest_slot=1 blocking_capture_wait=0",
            flush=True,
        )

    def _capture_gate_probe(self, _pad, _info, cid: str):
        if not self.gate_enabled:
            return self.Gst.PadProbeReturn.DROP

        now = time.monotonic()
        due = self.gate_next[cid]
        if now + 1e-9 < due:
            return self.Gst.PadProbeReturn.DROP

        # Fixed-deadline cadence: late source frames do not shift future deadlines
        # and no catch-up burst is emitted. Exactly this buffer is allowed through.
        late = max(0.0, now - due)
        steps = max(1, int(late // self.target_period) + 1)
        self.gate_next[cid] = due + steps * self.target_period
        self.gate_passed[cid] += 1
        return self.Gst.PadProbeReturn.OK

    def _take_oldest_ready(self):
        ready = None
        with self.capture_condition:
            for index, camera in enumerate(self.cameras):
                cid = camera.camera_id
                slot = self.capture_slots[cid]
                if slot.seq <= self.processed_seq[cid] or slot.frame is None:
                    continue
                candidate = (slot.captured_ns, index, cid, slot.seq, slot.frame)
                if ready is None or candidate[:2] < ready[:2]:
                    ready = candidate
            if ready is None:
                return None
            captured_ns, index, cid, seq, frame = ready
            self.processed_seq[cid] = seq
            return index, cid, seq, captured_ns, frame.copy()

    def _print_stats(self) -> None:
        now = time.monotonic()
        elapsed = max(1e-6, now - self.stats_at)
        gate_rows = []
        for camera in self.cameras:
            cid = camera.camera_id
            count = self.gate_passed[cid]
            delta = count - self.gate_passed_last[cid]
            self.gate_passed_last[cid] = count
            gate_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")
        super()._print_stats()
        print(
            "ML_DETECTOR_PACED_STATS "
            f"gate=[{' '.join(gate_rows)}] phase_spacing={self.phase_spacing_ms:.1f}ms "
            f"stale_drops={self.stale_drops} max_input_age={self.max_input_age_ms:.0f}ms "
            "capture_block_p95=0.0ms queue_depth=latest-per-camera",
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
            "scheduler=paced-ready-first blocking_capture_wait=0",
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
    service = DetectorSubstreamPacedService()

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
