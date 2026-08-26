from __future__ import annotations

import os
import queue as pyqueue
import time

from .person_tracking_pascal_trt86 import CameraPersonTrackingPascalTRT86
from .pose_gate import PoseGateClient


class CameraPersonTrackingTRT86PoseGate(CameraPersonTrackingPascalTRT86):
    """TRT8.6 person detector + crop-only pose validation + NvDCF.

    Strong YOLO person detections are accepted directly. Ambiguous candidates are
    validated by a low-rate pose worker before detector metadata reaches NvDCF.
    The clean camera-wall branch remains untouched; this is an ML feature branch.
    """

    def __init__(self) -> None:
        self.pose_gate: PoseGateClient | None = None
        self._gate_logs = 0
        super().__init__()
        self.pose_gate = PoseGateClient()
        print(
            "CAMERA_ML_ARCH "
            "primary=YOLO26s/TRT8.6 pose_gate=crop-only/CPU "
            "tracker=NvDCF global_id=off reid=off face=off",
            flush=True,
        )

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        assert self.pose_gate is not None

        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO TRT86 worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO TRT86 worker failed")
            return

        with self.det_lock:
            self.det_ready = True

        all_ids = [camera.camera_id for camera in self.cameras]
        allowed = self._active_camera_set()
        ids = [cid for cid in all_ids if cid in allowed]
        if not ids:
            raise RuntimeError("CAMERA_V2_DETECT_ACTIVE_CAMERAS selected no cameras")

        print(
            "CAMERA_ML_READY "
            f"model={ready.get('model')} input=672x384 micro_batch=1 "
            f"raw_conf={os.environ.get('CAMERA_V2_DETECT_CONF')} "
            f"target={self.detector_target_hz:.2f}Hz/cam "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"active={','.join(ids)} backend={ready.get('backend')} "
            "flow=TRT86->pose-gate->NvDCF capture=jit-latest-no-prefetch",
            flush=True,
        )

        groups = [[cid] for cid in ids]
        versions = {cid: 0 for cid in ids}
        group_index = 0
        age_log_n = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1

            # One fresh frame is requested only when the detector is ready for it.
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=0.8)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                    timeout_count = self.capture_timeouts
                if timeout_count <= 3 or timeout_count % 20 == 0:
                    print(
                        "CAMERA_ML_CAPTURE_TIMEOUT "
                        f"count={timeout_count} waiting={','.join(group)}",
                        flush=True,
                    )
                self.det_stop.wait(0.025)
                continue

            frames = []
            captured = []
            frame_by_cid = {}
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                captured.append(captured_t)
                frames.append(frame)
                frame_by_cid[cid] = frame
            self._clear_requests()

            try:
                self.job_q.put(
                    {"cameras": group, "frames": frames, "captured": captured},
                    timeout=0.3,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO TRT86 result timeout"
                self.det_stop.wait(0.05)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO TRT86 fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO TRT86 batch error")
                self.det_stop.wait(0.10)
                continue
            if result.get("type") != "result":
                continue

            counts: dict[str, int] = {}
            ages_ms: list[float] = []

            for cid, captured_t in zip(result["cameras"], result["captured"]):
                raw_rows = list(result["boxes"].get(cid, []))
                gated_rows, gate = self.pose_gate.filter(
                    cid,
                    frame_by_cid[cid],
                    raw_rows,
                )
                # Expand/de-duplicate only after pose has judged the original
                # detector geometry. Crop validation should never see expanded UI
                # margins because they can accidentally include a nearby person.
                detections = self._dedup_and_expand(gated_rows)
                prepared = self.latency_compensator.prepare(cid, captured_t, detections)
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)

                # Freshness must include pose validation time, not only TensorRT.
                # Otherwise an expensive pose crop can be reported as fresh and
                # then be discarded later by the metadata-injection age check.
                completed_t = time.monotonic()
                age_ms = max(0.0, (completed_t - captured_t) * 1000.0)
                ages_ms.append(age_ms)
                self.detector_times[cid].append(completed_t)

                self._gate_logs += 1
                if (
                    self._gate_logs <= 12
                    or self._gate_logs % 30 == 0
                    or gate.pose_reject > 0
                    or gate.fallback > 0
                ):
                    print(
                        "CAMERA_ML_GATE "
                        f"cid={cid} raw={gate.raw} direct={gate.direct} "
                        f"pose_checked={gate.pose_checked} pose_accept={gate.pose_accept} "
                        f"pose_reject={gate.pose_reject} low_reject={gate.low_reject} "
                        f"overflow={gate.overflow} final={gate.final} "
                        f"pose_ms={gate.pose_ms:.1f} fallback={gate.fallback}",
                        flush=True,
                    )

            self._update_freshness_budget(ages_ms)
            batch_ms = float(result.get("batch_ms") or 0.0)

            age_log_n += 1
            if ages_ms and (age_log_n <= 3 or age_log_n % 20 == 0):
                print(
                    "CAMERA_ML_FRESHNESS "
                    f"n={age_log_n} result_age={max(ages_ms):.1f}ms "
                    f"budget={self.max_detector_result_age_ms:.1f}ms "
                    f"trt_batch={batch_ms:.1f}ms",
                    flush=True,
                )

            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(group)
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                if ages_ms:
                    self.detector_result_age_ms = max(ages_ms)
                self.det_error = ""
                target_hz = self.detector_target_hz

            # The target is per-camera. Serial B1 TRT means each full six-camera
            # round must share the detector budget rather than burst all cameras.
            desired_call_interval = 1.0 / max(0.1, target_hz * len(groups))
            elapsed = time.monotonic() - cycle_started
            idle = max(self.detector_min_idle, desired_call_interval - elapsed)
            self.det_stop.wait(idle)

    def run(self) -> int:
        try:
            return super().run()
        finally:
            if self.pose_gate is not None:
                self.pose_gate.close()


def main() -> int:
    return CameraPersonTrackingTRT86PoseGate().run()


if __name__ == "__main__":
    raise SystemExit(main())
