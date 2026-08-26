from __future__ import annotations

import multiprocessing as mp
import os
import queue as pyqueue
import threading
import time
from collections import deque

from .detection import CameraDetectionV2
from .detection_only_pose_v3 import DetectionOnlyPoseV3
from .yolo26m_trt86_rescue import yolo26m_trt86_rescue_worker


class DetectionOnlyLowLatencyV4(DetectionOnlyPoseV3):
    """Low-latency golden display + adaptive S detector + CAM-05 M rescue.

    Display stays untouched and never waits on ML. The primary YOLO26s and the
    rescue YOLO26m share only the already-captured 672x384 BGR frame. Rescue has
    no second GStreamer branch/converter and runs in its own resident TRT8.6
    process. A main-process GPU lock prevents S and M kernels from contending;
    only detector cadence can wait, never the camera wall.
    """

    def __init__(self) -> None:
        self._gpu_infer_lock = threading.Lock()
        self._pose_call_lock = threading.Lock()
        self._cadence_lock = threading.Lock()
        self._rescue_stop = threading.Event()
        self._rescue_trigger_q: pyqueue.Queue = pyqueue.Queue(maxsize=1)
        self.rescue_job_q = None
        self.rescue_result_q = None
        self.rescue_worker = None
        self.rescue_thread = None
        self.rescue_ready = False
        self.rescue_error = ""
        self.rescue_calls = 0
        self.rescue_hits = 0
        self.rescue_skips = 0
        self.rescue_trt_ms = 0.0
        self.rescue_last_run = 0.0
        self._latest_capture_time: dict[str, float] = {}
        super().__init__()

        base_hz = max(0.10, float(os.environ.get("CAMERA_V2_DETECT_TARGET_HZ", "0.33")))
        self.primary_hz_min = max(
            0.10, float(os.environ.get("CAMERA_V2_PRIMARY_HZ_MIN", "0.28"))
        )
        self.primary_hz_max = max(
            self.primary_hz_min,
            float(os.environ.get("CAMERA_V2_PRIMARY_HZ_MAX", "0.40")),
        )
        self.current_primary_hz = min(
            self.primary_hz_max, max(self.primary_hz_min, base_hz)
        )

        self.rescue_enabled = os.environ.get(
            "CAMERA_V2_RESCUE_ENABLED", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.rescue_camera = os.environ.get(
            "CAMERA_V2_RESCUE_CAMERA", "CAM-05"
        ).strip() or "CAM-05"
        self.rescue_trigger_conf = max(
            0.08,
            float(os.environ.get("CAMERA_V2_RESCUE_TRIGGER_CONF", "0.18")),
        )
        self.rescue_min_interval = max(
            2.0,
            float(os.environ.get("CAMERA_V2_RESCUE_MIN_INTERVAL_SEC", "3.0")),
        )
        self.rescue_gpu_duty = min(
            0.20,
            max(0.03, float(os.environ.get("CAMERA_V2_RESCUE_GPU_DUTY", "0.08"))),
        )

        print(
            "CAMERA_LOWLAT_ARCH "
            "display=clean-wall/independent primary=YOLO26s/TRT8.6/672x384 "
            f"rescue=YOLO26m/TRT8.6/672x384/{self.rescue_camera} "
            "same_frame=1 extra_gstreamer_branch=0 concurrent_gpu_infer=0 "
            "nvdcf=0 osd=0 external_nms=0",
            flush=True,
        )
        print(
            "CAMERA_LOWLAT_POLICY "
            f"primary_hz={self.current_primary_hz:.2f} "
            f"range={self.primary_hz_min:.2f}-{self.primary_hz_max:.2f} "
            f"rescue_trigger_conf={self.rescue_trigger_conf:.2f} "
            f"rescue_min_interval={self.rescue_min_interval:.1f}s "
            f"rescue_gpu_duty_cap={self.rescue_gpu_duty:.0%}",
            flush=True,
        )

    def _pose_filter(self, cid: str, rows, frame):
        # Pose worker has one request/response queue; serialize callers from the
        # primary scheduler and rescue thread so response IDs can never cross.
        with self._pose_call_lock:
            return super()._pose_filter(cid, rows, frame)

    def _store_native_detection(
        self,
        cid: str,
        captured_t: float,
        native_rows,
        *,
        count_call: bool,
        batch_ms: float,
    ) -> None:
        now = time.monotonic()
        with self.det_lock:
            if count_call:
                self.det_calls += 1
                self.det_inputs += 1
                self.det_batch_ms = float(batch_ms)
            self.det_counts[cid] = len(native_rows)
            self._latest_detections[cid] = (now, list(native_rows))
            raw_cache = getattr(self, "_raw_detector_boxes", None)
            if raw_cache is None:
                raw_cache = {}
                self._raw_detector_boxes = raw_cache
            raw_cache[cid] = (float(captured_t), list(native_rows))
            self._latest_capture_time[cid] = float(captured_t)

    def _process_primary(
        self,
        cid: str,
        captured_t: float,
        frame,
        raw_rows,
        batch_ms: float,
    ) -> dict:
        now = time.monotonic()
        age_ms = max(0.0, (now - captured_t) * 1000.0)
        raw_max = max((float(score) for _coords, score in raw_rows), default=0.0)
        final_infer, pose_diag = self._pose_filter(cid, raw_rows, frame)
        final_native = self._scaled_detections(final_infer)
        self._store_native_detection(
            cid,
            captured_t,
            final_native,
            count_call=True,
            batch_ms=batch_ms,
        )
        self._detector_times[cid].append(now)
        self._result_age_samples.append(age_ms)

        print(
            "CAMERA_DETECTION_RESULT "
            f"cid={cid} raw={len(raw_rows)} raw_max={raw_max:.3f} "
            f"direct={getattr(pose_diag, 'direct', 0)} "
            f"cache_accept={getattr(pose_diag, 'cache_accept', 0)} "
            f"pose_checked={getattr(pose_diag, 'pose_checked', 0)} "
            f"pose_accept={getattr(pose_diag, 'pose_accept', 0)} "
            f"pose_reject={getattr(pose_diag, 'pose_reject', 0)} "
            f"soft_hold={getattr(pose_diag, 'soft_hold', 0)} "
            f"confirmed_reject={getattr(pose_diag, 'confirmed_reject', 0)} "
            f"final={len(final_native)} pose_ms={getattr(pose_diag, 'pose_ms', 0.0):.1f} "
            f"age={age_ms:.1f}ms",
            flush=True,
        )

        if self.det_calls <= 3 or self.det_calls % 20 == 0:
            print(
                "CAMERA_DETECTION_FRESHNESS "
                f"n={self.det_calls} result_age={age_ms:.1f}ms "
                f"trt_batch={batch_ms:.1f}ms target={self.current_primary_hz:.2f}Hz/cam",
                flush=True,
            )
        return {
            "cid": cid,
            "captured": float(captured_t),
            "frame": frame,
            "raw_count": len(raw_rows),
            "raw_max": raw_max,
            "final_infer": final_infer,
            "final_native": final_native,
        }

    def _queue_rescue(self, primary: dict) -> None:
        if not self.rescue_enabled or primary["cid"] != self.rescue_camera:
            return
        if primary["final_native"] and primary["raw_max"] >= self.rescue_trigger_conf:
            return
        # Latest-only trigger: an old CAM-05 frame is worthless if a newer miss
        # arrives before the rescue worker can run.
        try:
            while True:
                self._rescue_trigger_q.get_nowait()
                self.rescue_skips += 1
        except pyqueue.Empty:
            pass
        try:
            self._rescue_trigger_q.put_nowait(primary)
            print(
                "CAMERA_RESCUE_TRIGGER "
                f"cid={primary['cid']} primary_final={len(primary['final_native'])} "
                f"primary_max={primary['raw_max']:.3f}",
                flush=True,
            )
        except pyqueue.Full:
            self.rescue_skips += 1

    def _effective_rescue_interval(self) -> float:
        runtime_based = (self.rescue_trt_ms / 1000.0) / max(0.01, self.rescue_gpu_duty)
        return max(self.rescue_min_interval, runtime_based)

    def _rescue_loop(self) -> None:
        if self.rescue_result_q is None or self.rescue_job_q is None:
            return
        try:
            ready = self.rescue_result_q.get(timeout=90.0)
        except pyqueue.Empty:
            self.rescue_error = "rescue startup timeout"
            return
        if ready.get("type") != "ready":
            self.rescue_error = str(ready.get("error") or "rescue startup failed")
            return
        self.rescue_ready = True
        print(
            "CAMERA_RESCUE_READY "
            f"camera={self.rescue_camera} backend={ready.get('backend')} "
            f"model={ready.get('model')}",
            flush=True,
        )

        while not self._rescue_stop.is_set():
            try:
                primary = self._rescue_trigger_q.get(timeout=0.25)
            except pyqueue.Empty:
                continue

            now = time.monotonic()
            wait_needed = self._effective_rescue_interval() - (now - self.rescue_last_run)
            if wait_needed > 0.0:
                self.rescue_skips += 1
                print(
                    "CAMERA_RESCUE_SKIP "
                    f"reason=duty_guard remaining={wait_needed:.1f}s",
                    flush=True,
                )
                continue

            frame = primary["frame"]
            captured = float(primary["captured"])
            job = {
                "cameras": [self.rescue_camera],
                "frames": [frame],
                "captured": [captured],
            }
            try:
                # Serialize GPU inference only. Display is a different tee thread
                # and never takes this lock.
                with self._gpu_infer_lock:
                    self.rescue_job_q.put(job, timeout=0.3)
                    result = self.rescue_result_q.get(timeout=8.0)
            except pyqueue.Empty:
                self.rescue_error = "rescue result timeout"
                continue
            except Exception as exc:
                self.rescue_error = f"{type(exc).__name__}:{exc}"
                continue

            if result.get("type") != "result":
                self.rescue_error = str(result.get("error") or result.get("type"))
                continue
            raw_rows = result.get("boxes", {}).get(self.rescue_camera, [])
            self.rescue_trt_ms = float(result.get("batch_ms") or 0.0)
            self.rescue_last_run = time.monotonic()
            self.rescue_calls += 1

            final_infer, pose_diag = self._pose_filter(
                self.rescue_camera, raw_rows, frame
            )
            final_native = self._scaled_detections(final_infer)
            rescue_max = max((float(score) for _c, score in raw_rows), default=0.0)

            primary_count = len(primary["final_native"])
            rescue_count = len(final_native)
            merged = bool(
                rescue_count > primary_count
                or (
                    rescue_count > 0
                    and primary["raw_max"] < self.rescue_trigger_conf
                    and rescue_max > primary["raw_max"]
                )
            )
            if merged:
                self._store_native_detection(
                    self.rescue_camera,
                    captured,
                    final_native,
                    count_call=False,
                    batch_ms=self.rescue_trt_ms,
                )
                self.rescue_hits += 1

            print(
                "CAMERA_RESCUE_RESULT "
                f"cid={self.rescue_camera} primary={primary_count}/{primary['raw_max']:.3f} "
                f"rescue_raw={len(raw_rows)}/{rescue_max:.3f} "
                f"rescue_final={rescue_count} trt={self.rescue_trt_ms:.1f}ms "
                f"pose_checked={getattr(pose_diag, 'pose_checked', 0)} "
                f"pose_ms={getattr(pose_diag, 'pose_ms', 0.0):.1f} "
                f"merged={int(merged)} next_min_interval={self._effective_rescue_interval():.1f}s",
                flush=True,
            )

    def _scheduler(self) -> None:
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
        ids = [cid for cid in all_ids if not configured or cid in set(configured)]
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
            f"stagger={period / max(1, len(ids)):.3f}s rescue={self.rescue_camera}",
            flush=True,
        )

        while not self.det_stop.is_set():
            cid = min(ids, key=lambda x: due[x])
            now = time.monotonic()
            if due[cid] > now:
                if self.det_stop.wait(min(0.20, due[cid] - now)):
                    break
                continue

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

            job = {
                "cameras": [cid],
                "frames": [frame],
                "captured": [captured_t],
            }
            try:
                with self._gpu_infer_lock:
                    self.job_q.put(job, timeout=0.3)
                    result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "primary TRT86 result timeout"
                due[cid] = time.monotonic() + 0.5
                continue
            except Exception as exc:
                with self.det_lock:
                    self.det_error = f"primary {type(exc).__name__}:{exc}"
                due[cid] = time.monotonic() + 0.5
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
                cid, float(captured_t), frame, raw_rows, batch_ms
            )
            self._queue_rescue(primary)

            with self._cadence_lock:
                period = 1.0 / max(0.01, self.current_primary_hz)
            # Keep cameras phase-staggered instead of firing all six in a burst.
            due[cid] = max(due[cid] + period, time.monotonic() + period * 0.45)

    @staticmethod
    def _p95(values: deque[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        wall_p95 = self._p95(self.wall_intervals_ms)
        with self._cadence_lock:
            old_hz = self.current_primary_hz
            if wall_p95 is not None:
                if wall_p95 > 90.0:
                    self.current_primary_hz = max(
                        self.primary_hz_min, self.current_primary_hz - 0.04
                    )
                elif wall_p95 > 75.0:
                    self.current_primary_hz = max(
                        self.primary_hz_min, self.current_primary_hz - 0.02
                    )
                elif wall_p95 < 62.0:
                    self.current_primary_hz = min(
                        self.primary_hz_max, self.current_primary_hz + 0.01
                    )
            current_hz = self.current_primary_hz

        wall = "?" if wall_p95 is None else f"{wall_p95:.1f}ms"
        print(
            "CAMERA_LOWLAT_STATS "
            f"wall_p95={wall} primary_hz={current_hz:.2f} "
            f"cadence_changed={int(abs(current_hz - old_hz) > 1e-9)} "
            f"rescue_ready={int(self.rescue_ready)} calls={self.rescue_calls} "
            f"hits={self.rescue_hits} skips={self.rescue_skips} "
            f"rescue_trt={self.rescue_trt_ms:.1f}ms "
            f"rescue_interval={self._effective_rescue_interval():.1f}s"
            + (f" rescue_error={self.rescue_error}" if self.rescue_error else ""),
            flush=True,
        )

        # Restore the whole-process clean watchdog that V3 intentionally skipped
        # when it removed legacy detector stats.
        now = time.monotonic()
        for cid in self.sources:
            current = int(self.stats[cid].frames)
            if current != self._last_frames.get(cid, 0):
                self._last_frames[cid] = current
                self._last_progress[cid] = now
                continue
            started = self._source_started_at.get(cid)
            if started is None or now - started < self._stall_s:
                continue
            stalled_for = now - self._last_progress.get(cid, started)
            if stalled_for >= self._stall_s and not self._restart_requested:
                self._restart_requested = True
                self._restart_reason = f"{cid}-no-frame-{stalled_for:.1f}s"
                print(
                    "CAMERA_LOWLAT_PROCESS_RESTART "
                    f"reason={self._restart_reason}",
                    flush=True,
                )
                try:
                    self.loop.quit()
                except Exception:
                    pass
                return False
        return keep

    def run(self) -> int:
        ctx = mp.get_context("spawn")
        if self.rescue_enabled:
            self.rescue_job_q = ctx.Queue(maxsize=1)
            self.rescue_result_q = ctx.Queue(maxsize=2)
            self.rescue_worker = ctx.Process(
                target=yolo26m_trt86_rescue_worker,
                args=(self.rescue_job_q, self.rescue_result_q),
                name="camera-v2-yolo26m-rescue",
                daemon=True,
            )
            self.rescue_worker.start()
            self.rescue_thread = threading.Thread(
                target=self._rescue_loop,
                name="camera-v2-yolo26m-rescue-scheduler",
                daemon=True,
            )
            self.rescue_thread.start()

        try:
            # Call the detector base directly so pose lifetime also covers rescue.
            code = CameraDetectionV2.run(self)
        finally:
            self._rescue_stop.set()
            if self.rescue_job_q is not None:
                try:
                    self.rescue_job_q.put_nowait(None)
                except Exception:
                    pass
            if self.rescue_thread is not None:
                self.rescue_thread.join(timeout=2.0)
            if self.rescue_worker is not None:
                self.rescue_worker.join(timeout=3.0)
                if self.rescue_worker.is_alive():
                    self.rescue_worker.terminate()
                    self.rescue_worker.join(timeout=1.0)
            try:
                self.pose_gate.close()
            except Exception:
                pass

        if self._restart_requested:
            return 75
        return code


def main() -> int:
    return DetectionOnlyLowLatencyV4().run()


if __name__ == "__main__":
    raise SystemExit(main())
