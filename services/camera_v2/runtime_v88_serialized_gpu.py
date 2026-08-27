from __future__ import annotations

import os
import queue as pyqueue
import threading
import time

from .runtime_v85_nvdcf_relief import PascalNvDCFReliefRuntime


class PascalSerializedGpuOwnershipRuntime(PascalNvDCFReliefRuntime):
    """V8.8: short detector reservations to avoid Pascal CUDA context time-slicing."""

    def __init__(self) -> None:
        self.v88_detector_reserved = threading.Event()
        self.v88_quiet_before_ms = max(
            0.0,
            min(80.0, float(os.environ.get("CAMERA_V88_QUIET_BEFORE_MS", "30"))),
        )
        self.v88_tracker_gate_drops = 0
        self.v88_mux_gate_drops = 0
        self.v88_reservations = 0
        self.v88_reservation_ms_ema = 0.0
        self.v88_reservation_alpha = 0.20
        super().__init__()
        print(
            "CAMERA_V88_ARCH "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"quiet_before={self.v88_quiet_before_ms:.0f}ms detector=batch1-v84 "
            "gpu_ownership=short-serialized display_gated=0 bbox_policy=unchanged "
            "mps=0 tracker_quality=v85",
            flush=True,
        )

    def _tracker_rate_probe(self, pad, info, cid: str):
        if self.v88_detector_reserved.is_set():
            self.v88_tracker_gate_drops += 1
            return self.Gst.PadProbeReturn.DROP
        return super()._tracker_rate_probe(pad, info, cid)

    def _inject_detector_probe(self, pad, info):
        if self.v88_detector_reserved.is_set():
            self.v88_mux_gate_drops += 1
            return self.Gst.PadProbeReturn.DROP
        return super()._inject_detector_probe(pad, info)

    def _detector_scheduler_v84(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=60.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "V88 TRT86 batch1 startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "V88 TRT86 batch1 startup failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_V88_DETECT_READY "
            f"model={ready.get('model')} backend={ready.get('backend')} batch=1 "
            f"global={self.v84_global_hz:.2f}Hz per_camera={self.detect_hz:.2f}Hz "
            f"quiet_before={self.v88_quiet_before_ms:.0f}ms queue_depth=1",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        versions = {cid: 0 for cid in ids}
        rr = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            selected = None
            for offset in range(len(ids)):
                cid = ids[(rr + offset) % len(ids)]
                if self.stats[cid].frames > 0:
                    selected = cid
                    rr = (rr + offset + 1) % len(ids)
                    break
            if selected is None:
                self.det_stop.wait(0.03)
                continue

            cid = selected
            self._request_capture(cid)
            row = self.mailbox.wait_new(
                cid,
                versions[cid],
                timeout=self.v84_capture_timeout,
            )
            self._clear_capture(cid)
            if row is None:
                self.v84_capture_miss += 1
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.003)
                continue

            version, captured_at, frame = row
            versions[cid] = version
            reservation_started = time.monotonic()
            self.v88_detector_reserved.set()
            self.v88_reservations += 1
            try:
                if self.v88_quiet_before_ms > 0.0:
                    self.det_stop.wait(self.v88_quiet_before_ms / 1000.0)
                if self.det_stop.is_set():
                    return
                self.job_q.put(
                    {"cameras": [cid], "frames": [frame], "captured": [captured_at]},
                    timeout=0.10,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "V88 TRT86 batch1 result timeout"
                self.det_stop.wait(0.03)
                continue
            finally:
                self.v88_detector_reserved.clear()
                held_ms = max(0.0, (time.monotonic() - reservation_started) * 1000.0)
                if self.v88_reservation_ms_ema <= 0.0:
                    self.v88_reservation_ms_ema = held_ms
                else:
                    a = self.v88_reservation_alpha
                    self.v88_reservation_ms_ema = (
                        self.v88_reservation_ms_ema * (1.0 - a) + held_ms * a
                    )

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "V88 TRT86 batch1 fatal")
                return
            if result.get("type") != "result":
                continue

            completed = time.monotonic()
            gpu_ms = float(result.get("batch_ms") or 0.0)
            roundtrip_ms = float(result.get("total_ms") or gpu_ms)
            self.v84_gpu_ms_ema = self._ema_v84(
                self.v84_gpu_ms_ema, gpu_ms, self.v84_ema_alpha
            )
            self.v84_roundtrip_ms_ema = self._ema_v84(
                self.v84_roundtrip_ms_ema, roundtrip_ms, self.v84_ema_alpha
            )
            self._adapt_v84()

            if self.v84_last_complete > 0.0:
                self.v84_intervals.append(completed - self.v84_last_complete)
            self.v84_last_complete = completed
            self.v84_calls += 1
            self.v84_per_camera_calls[cid] += 1

            detector_rows = result.get("boxes", {}).get(cid, [])
            boxes = self._map_detector_rows(detector_rows)
            self._publish_detector(cid, captured_at, boxes)
            age_ms = max(0.0, (completed - captured_at) * 1000.0)
            self.v84_result_age_ms = age_ms

            with self.det_lock:
                self.det_counts[cid] = len(boxes)
                self.detector_times[cid].append(completed)
                self.det_calls += 1
                self.det_inputs += 1
                self.det_batch_ms = gpu_ms
                self.det_result_age_ms = age_ms
                self.det_error = ""

            if self.v84_calls <= 8 or self.v84_calls % 20 == 0:
                duty = self.v84_global_hz * max(0.0, self.v84_gpu_ms_ema) / 1000.0
                print(
                    "CAMERA_V88_DETECT "
                    f"call={self.v84_calls} camera={cid} gpu={gpu_ms:.1f}ms "
                    f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
                    f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
                    f"age={age_ms:.0f}ms global_hz={self.v84_global_hz:.2f} "
                    f"duty_est={duty:.2f} reservation_ema={self.v88_reservation_ms_ema:.1f}ms "
                    f"tracker_gate_drops={self.v88_tracker_gate_drops} "
                    f"mux_gate_drops={self.v88_mux_gate_drops} boxes={len(boxes)}",
                    flush=True,
                )

            desired = 1.0 / max(0.1, self.v84_global_hz)
            elapsed = time.monotonic() - cycle_started
            self.det_stop.wait(max(0.001, desired - elapsed))

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_V88_STATS "
            f"reservations={self.v88_reservations} "
            f"reservation_ema={self.v88_reservation_ms_ema:.1f}ms "
            f"quiet_before={self.v88_quiet_before_ms:.0f}ms "
            f"tracker_gate_drops={self.v88_tracker_gate_drops} "
            f"mux_gate_drops={self.v88_mux_gate_drops} "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
            f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
            f"tracked_now={self.tracked_now} tracker_batches={self.tracker_batches} "
            "display_gated=0 bbox_policy=unchanged",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalSerializedGpuOwnershipRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
