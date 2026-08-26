from __future__ import annotations

import os
import queue as pyqueue
import time

from .detection_only_pose_v4 import DetectionOnlyLowLatencyV4


class DetectionLowLatency(DetectionOnlyLowLatencyV4):
    """Canonical low-latency runtime with fresh-capture scheduling."""

    def _pose_filter(self, cid: str, rows, frame):
        boxes = [
            (tuple(float(v) for v in coords), float(score))
            for coords, score in rows
        ]
        # pose_gate_v3 uses trusted_boxes=; older detection-only code used the
        # obsolete existing_boxes= spelling and therefore never exercised the
        # real pose path while it inherited the old scheduler.
        with self._pose_call_lock:
            return self.pose_gate.filter(
                cid,
                frame,
                boxes,
                trusted_boxes=None,
            )

    def _scheduler(self) -> None:
        """Primary S scheduler that never queues an already-stale camera frame.

        The shared GPU slot is acquired first. Only then is the per-camera gate
        opened, so the captured BGR frame is the freshest frame available just
        before TRT inference. Rescue may delay detector cadence, but can never add
        its runtime to the age of a frame that was captured before the wait.
        """
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "primary TRT86 startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = str(ready.get("error") or "primary TRT86 failed")
            return
        with self.det_lock:
            self.det_ready = True

        all_ids = [camera.camera_id for camera in self.cameras]
        configured = [
            x.strip()
            for x in os.environ.get("CAMERA_V2_DETECT_ACTIVE_CAMERAS", "").split(",")
            if x.strip()
        ]
        allowed = set(configured)
        ids = [cid for cid in all_ids if not configured or cid in allowed]
        if not ids:
            with self.det_lock:
                self.det_error = "primary selected no cameras"
            return

        versions = {cid: 0 for cid in ids}
        start = time.monotonic()
        period = 1.0 / max(0.01, self.current_primary_hz)
        due = {
            cid: start + (i * period / max(1, len(ids)))
            for i, cid in enumerate(ids)
        }
        print(
            "CAMERA_LOWLAT_READY "
            f"primary=YOLO26s/672x384 cameras={ids} "
            f"stagger={period / max(1, len(ids)):.3f}s rescue={self.rescue_camera} "
            "capture_after_gpu_slot=1",
            flush=True,
        )

        while not self.det_stop.is_set():
            cid = min(ids, key=lambda x: due[x])
            now = time.monotonic()
            if due[cid] > now:
                if self.det_stop.wait(min(0.20, due[cid] - now)):
                    break
                continue

            captured_t = None
            frame = None
            result = None
            try:
                # Critical freshness rule: wait for any rare M-rescue first, then
                # open the gate and consume a new frame. The wall never takes this
                # lock and remains fully independent.
                with self._gpu_infer_lock:
                    self._request_group([cid])
                    rows = self.mailbox.wait_group([cid], versions, timeout=1.0)
                    if rows is None:
                        self._clear_requests()
                        with self.det_lock:
                            self.capture_timeouts += 1
                        due[cid] = time.monotonic() + 0.25
                        continue
                    version, captured_t, frame = rows[0]
                    versions[cid] = version
                    self._clear_requests()
                    self.job_q.put(
                        {
                            "cameras": [cid],
                            "frames": [frame],
                            "captured": [captured_t],
                        },
                        timeout=0.3,
                    )
                    result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                self._clear_requests()
                with self.det_lock:
                    self.det_error = "primary TRT86 result timeout"
                due[cid] = time.monotonic() + 0.5
                continue
            except Exception as exc:
                self._clear_requests()
                with self.det_lock:
                    self.det_error = f"primary {type(exc).__name__}:{exc}"
                due[cid] = time.monotonic() + 0.5
                continue

            if result is None:
                due[cid] = time.monotonic() + 0.25
                continue
            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = str(result.get("error") or "primary fatal")
                return
            if result.get("type") != "result":
                due[cid] = time.monotonic() + 0.25
                continue

            raw_rows = result.get("boxes", {}).get(cid, [])
            batch_ms = float(result.get("batch_ms") or 0.0)
            primary = self._process_primary(
                cid,
                float(captured_t),
                frame,
                raw_rows,
                batch_ms,
            )
            self._queue_rescue(primary)

            with self._cadence_lock:
                period = 1.0 / max(0.01, self.current_primary_hz)
            due[cid] = max(
                due[cid] + period,
                time.monotonic() + period * 0.45,
            )


def main() -> int:
    return DetectionLowLatency().run()


if __name__ == "__main__":
    raise SystemExit(main())
