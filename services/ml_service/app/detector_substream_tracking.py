from __future__ import annotations

import os
import signal
import time
from collections import deque

from services.ml_service.app.detector_substream import _percentile
from services.ml_service.app.detector_substream_prequeue_token import (
    DetectorSubstreamPrequeueTokenService,
)
from services.ml_service.app.local_tracker import MultiCameraLocalTracker, TrackerUpdate
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W, TRT86DetectorClient


class DetectorSubstreamTrackingService(DetectorSubstreamPrequeueTokenService):
    """Step 4: frozen V14 detector plus CPU-only, per-camera local tracking.

    Camera Service and the high-quality main stream are untouched. Tracking runs only
    after sparse TRT detections have returned, uses no video-frame GPU processing and
    keeps IDs local to each camera. Cross-camera ReID/global identity remains a later
    stage by design.
    """

    def __init__(self) -> None:
        super().__init__()
        self.track_low_thresh = max(
            self.conf, float(os.environ.get("ML_TRACK_LOW_THRESH", str(self.conf)))
        )
        self.track_high_thresh = max(
            self.track_low_thresh,
            float(os.environ.get("ML_TRACK_HIGH_THRESH", "0.30")),
        )
        self.track_new_thresh = max(
            self.track_high_thresh,
            float(os.environ.get("ML_TRACK_NEW_THRESH", "0.30")),
        )
        self.track_max_lost_sec = max(
            1.0, float(os.environ.get("ML_TRACK_MAX_LOST_SEC", "2.5"))
        )
        self.track_shadow_sec = min(
            self.track_max_lost_sec,
            max(0.0, float(os.environ.get("ML_TRACK_SHADOW_SEC", "0.9"))),
        )
        self.track_log_objects = os.environ.get("ML_TRACK_LOG_OBJECTS", "0").strip() == "1"
        self.trackers = MultiCameraLocalTracker(
            (camera.camera_id for camera in self.cameras),
            INPUT_W,
            CONTENT_H,
            low_thresh=self.track_low_thresh,
            high_thresh=self.track_high_thresh,
            new_track_thresh=self.track_new_thresh,
            confirm_hits=max(1, int(os.environ.get("ML_TRACK_CONFIRM_HITS", "2"))),
            tentative_ttl_sec=max(
                0.3, float(os.environ.get("ML_TRACK_TENTATIVE_TTL_SEC", "0.9"))
            ),
            shadow_sec=self.track_shadow_sec,
            max_lost_sec=self.track_max_lost_sec,
            appearance_weight=min(
                0.30, max(0.0, float(os.environ.get("ML_TRACK_APPEARANCE_WEIGHT", "0.18")))
            ),
        )
        self.track_step_ms: deque[float] = deque(maxlen=300)
        self.track_updates = {camera.camera_id: 0 for camera in self.cameras}
        self.track_updates_last = {camera.camera_id: 0 for camera in self.cameras}
        self.track_active = {camera.camera_id: 0 for camera in self.cameras}
        self.track_renderable = {camera.camera_id: 0 for camera in self.cameras}
        self.track_created_total = 0
        self.track_recovered_total = 0
        self.track_lost_total = 0
        self.track_removed_total = 0
        self.track_stats_created = 0
        self.track_stats_recovered = 0
        self.track_stats_lost = 0
        self.track_stats_removed = 0
        self.latest_tracks = {camera.camera_id: [] for camera in self.cameras}

    @staticmethod
    def _content_boxes(rows):
        """Remove TRT worker's 3 px top/bottom letterbox padding for 672x378 tracking."""
        boxes = []
        for row in rows:
            if len(row) != 5:
                continue
            x1, y1, x2, y2, score = (float(v) for v in row)
            y1 = max(0.0, min(float(CONTENT_H - 1), y1 - 3.0))
            y2 = max(0.0, min(float(CONTENT_H - 1), y2 - 3.0))
            x1 = max(0.0, min(float(INPUT_W - 1), x1))
            x2 = max(0.0, min(float(INPUT_W - 1), x2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2, score])
        return boxes

    def _record_tracks(self, update: TrackerUpdate, cid: str, seq: int, total_n: int) -> None:
        self.track_step_ms.append(update.step_ms)
        self.track_updates[cid] += 1
        self.track_active[cid] = update.active
        self.track_renderable[cid] = update.renderable
        self.track_created_total += update.created
        self.track_recovered_total += update.recovered
        self.track_lost_total += update.newly_lost
        self.track_removed_total += update.removed
        self.track_stats_created += update.created
        self.track_stats_recovered += update.recovered
        self.track_stats_lost += update.newly_lost
        self.track_stats_removed += update.removed
        self.latest_tracks[cid] = update.snapshots

        eventful = bool(
            update.created or update.recovered or update.newly_lost or update.removed
        )
        if total_n <= 12 or total_n % 20 == 0 or eventful:
            ids = ",".join(s.track_id for s in update.snapshots) or "-"
            print(
                "ML_TRACK_FRAME "
                f"camera={cid} frame_seq={seq} det={update.detections} "
                f"high={update.high_detections} low={update.low_detections} "
                f"active={update.active} render={update.renderable} "
                f"matched_high={update.matched_high} matched_low={update.matched_low} "
                f"new={update.created} recovered={update.recovered} "
                f"lost={update.newly_lost} removed={update.removed} "
                f"step={update.step_ms:.3f}ms ids={ids}",
                flush=True,
            )

        if self.track_log_objects:
            for snap in update.snapshots:
                b = snap.bbox_norm
                v = snap.velocity_norm_s
                print(
                    "ML_TRACK_OBJECT "
                    f"camera={cid} id={snap.track_id} state={snap.state} "
                    f"confirmed={int(snap.confirmed)} predicted={int(snap.predicted)} "
                    f"score={snap.score:.3f} hits={snap.hits} "
                    f"box_norm={b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f} "
                    f"vel_norm_s={v[0]:.4f},{v[1]:.4f},{v[2]:.4f},{v[3]:.4f} "
                    f"since_det={snap.since_detection_sec:.3f}s",
                    flush=True,
                )

    def _print_tracking_stats(self) -> None:
        now = time.monotonic()
        elapsed = max(1e-6, now - self.stats_at)
        update_rows = []
        active_rows = []
        render_rows = []
        for camera in self.cameras:
            cid = camera.camera_id
            count = self.track_updates[cid]
            delta = count - self.track_updates_last[cid]
            self.track_updates_last[cid] = count
            update_rows.append(f"{cid}:{delta / elapsed:.2f}Hz")
            active_rows.append(f"{cid}:{self.track_active[cid]}")
            render_rows.append(f"{cid}:{self.track_renderable[cid]}")

        print(
            "ML_TRACK_STATS "
            f"updates=[{' '.join(update_rows)}] active=[{' '.join(active_rows)}] "
            f"render=[{' '.join(render_rows)}] "
            f"created={self.track_stats_created} recovered={self.track_stats_recovered} "
            f"lost={self.track_stats_lost} removed={self.track_stats_removed} "
            f"step_p95={_percentile(self.track_step_ms, 0.95):.3f}ms "
            f"step_avg={sum(self.track_step_ms) / len(self.track_step_ms) if self.track_step_ms else 0.0:.3f}ms "
            f"low={self.track_low_thresh:.2f} high={self.track_high_thresh:.2f} "
            f"new={self.track_new_thresh:.2f} max_lost={self.track_max_lost_sec:.1f}s "
            f"shadow={self.track_shadow_sec:.1f}s cpu_only=1 appearance=model-free",
            flush=True,
        )
        self.track_stats_created = 0
        self.track_stats_recovered = 0
        self.track_stats_lost = 0
        self.track_stats_removed = 0

    def _print_stats(self) -> None:
        # Parent token telemetry owns/reset self.stats_at, so capture elapsed first via
        # the tracker counters by printing detector/token stats, then a cumulative
        # tracking snapshot. Tracker update Hz is also visible in ML_TRACK_FRAME.
        DetectorSubstreamPrequeueTokenService._print_stats(self)
        active_rows = " ".join(f"{c.camera_id}:{self.track_active[c.camera_id]}" for c in self.cameras)
        render_rows = " ".join(
            f"{c.camera_id}:{self.track_renderable[c.camera_id]}" for c in self.cameras
        )
        print(
            "ML_TRACK_STATS "
            f"active=[{active_rows}] render=[{render_rows}] "
            f"created_total={self.track_created_total} recovered_total={self.track_recovered_total} "
            f"lost_total={self.track_lost_total} removed_total={self.track_removed_total} "
            f"step_p95={_percentile(self.track_step_ms, 0.95):.3f}ms "
            f"step_avg={sum(self.track_step_ms) / len(self.track_step_ms) if self.track_step_ms else 0.0:.3f}ms "
            f"low={self.track_low_thresh:.2f} high={self.track_high_thresh:.2f} "
            f"new={self.track_new_thresh:.2f} max_lost={self.track_max_lost_sec:.1f}s "
            f"shadow={self.track_shadow_sec:.1f}s cpu_only=1 appearance=model-free",
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
            "ML_STEP4_TRACKING_PROFILE "
            "scope=per-camera cpu_only=1 algorithm=time-aware-bytetrack-style "
            "motion=constant-velocity-seconds association=high-low-two-stage "
            "appearance=model-free-24d-color-hint gpu_tracker=0 nvdcf=0 reid=0 global_id=0 "
            f"confirm_hits=2 max_lost={self.track_max_lost_sec:.1f}s "
            f"shadow={self.track_shadow_sec:.1f}s overlay_prediction=metadata-ready",
            flush=True,
        )
        print(
            "ML_DETECTOR_BOUNDARY camera_service=independent main_stream=0 camera_shm=0 "
            "tracker=cpu-local-step4 api=0 ui=0 substream_nvdec=1 sparse_convert=1 "
            f"scheduler=prequeue-wall-token-bucket-ready-first pending_depth={self.pending_depth} "
            f"token_capacity={self.token_capacity} blocking_capture_wait=0",
            flush=True,
        )

        self._start_sources()
        self.detector = TRT86DetectorClient()
        self._enable_paced_gate()

        try:
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

                content_boxes = self._content_boxes(result.boxes)
                track_update = self.trackers.update(cid, content_boxes, frame, captured_ns)
                n = sum(self.processed.values())
                self._record_tracks(track_update, cid, seq, n)

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
        finally:
            self.stop_requested = True
            self._stop_demand_scheduler()

        return 0


def main() -> int:
    service = DetectorSubstreamTrackingService()

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
