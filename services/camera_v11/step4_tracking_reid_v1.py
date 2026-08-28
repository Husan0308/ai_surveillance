from __future__ import annotations

import os
import signal
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from .step2_production_fp32 import _pct
from .step3_tracking_v2 import V11Step3TrackingV2
from .step4_identity_shadow_v1 import V11CrossCameraIdentityShadowV1
from .step4_reid_scheduler_v1 import ReIDCandidate, ReIDResult, V11ReIDSchedulerV1

ROOT = Path(__file__).resolve().parents[2]
FROZEN_STEP3_SHA = "d2c9e62f9ed2b5f80dc9a4d496e0fda94afddc51"


def _camera_rooms() -> dict[str, str]:
    path = Path(os.environ.get("CAMERA_CONFIG", "config/cameras.yaml")).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        str(row.get("id", "")): str(row.get("room", ""))
        for row in raw.get("cameras") or []
        if row.get("id")
    }


class V11Step4TrackingReIDV1(V11Step3TrackingV2):
    """Frozen Step3 plus asynchronous, bounded, calibration-first cross-camera ReID."""

    def __init__(self) -> None:
        super().__init__()
        self.camera_rooms = _camera_rooms()
        self.reid_refresh_sec = max(
            0.5, min(10.0, float(os.environ.get("V11_STEP4_REID_REFRESH_SEC", "1.0")))
        )
        self.reid_min_score = max(
            0.18, min(0.95, float(os.environ.get("V11_STEP4_REID_MIN_SCORE", "0.30")))
        )
        self.reid_min_height = max(
            32, min(240, int(os.environ.get("V11_STEP4_REID_MIN_HEIGHT", "64")))
        )
        self.reid_min_width = max(
            12, min(160, int(os.environ.get("V11_STEP4_REID_MIN_WIDTH", "24")))
        )
        self.reid_max_per_update = max(
            1, min(4, int(os.environ.get("V11_STEP4_REID_MAX_PER_UPDATE", "2")))
        )
        self.reid_submit_at: dict[tuple[str, str], int] = {}
        self.reid_submitted_by_camera = {camera.camera_id: 0 for camera in self.cameras}
        self.reid_skip_score = 0
        self.reid_skip_size = 0
        self.reid_skip_predicted = 0
        self.reid_submit_rejected = 0
        self.reid_crop_copy_values: deque[float] = deque(maxlen=2048)
        self.reid_result_lock = threading.RLock()
        self.reid_last_match: dict[tuple[str, str], tuple[str, str, str]] = {}

        self.identity_shadow = V11CrossCameraIdentityShadowV1(
            gallery_size=int(os.environ.get("V11_STEP4_REID_GALLERY_SIZE", "4")),
            ttl_sec=float(os.environ.get("V11_STEP4_REID_SHADOW_TTL_SEC", "30")),
            weak_similarity=float(os.environ.get("V11_STEP4_REID_WEAK_SIM", "0.66")),
            candidate_similarity=float(os.environ.get("V11_STEP4_REID_CANDIDATE_SIM", "0.76")),
            strong_similarity=float(os.environ.get("V11_STEP4_REID_STRONG_SIM", "0.84")),
            min_margin=float(os.environ.get("V11_STEP4_REID_MIN_MARGIN", "0.04")),
            strong_margin=float(os.environ.get("V11_STEP4_REID_STRONG_MARGIN", "0.025")),
        )
        self.reid_scheduler = V11ReIDSchedulerV1(
            self._on_reid_result,
            max_batch=int(os.environ.get("V11_STEP4_REID_MAX_BATCH", "2")),
            max_wait_ms=float(os.environ.get("V11_STEP4_REID_MAX_WAIT_MS", "3")),
            max_pending=int(os.environ.get("V11_STEP4_REID_MAX_PENDING", "12")),
            max_age_ms=float(os.environ.get("V11_STEP4_REID_MAX_AGE_MS", "300")),
        )
        self._reid_closed = False

        print(
            "CAMERA_V11_STEP4_ARCH "
            f"frozen_step3_sha={FROZEN_STEP3_SHA} reid=trt86-resnet50 scheduler=async-thread "
            f"max_batch={self.reid_scheduler.max_batch} max_wait_ms={self.reid_scheduler.max_wait_ms:.1f} "
            f"pending_max={self.reid_scheduler.max_pending} pending_policy=per-track-latest-overwrite "
            "display_blocking=0 detector_blocking=0 frame_queue=0 identity_mode=shadow-no-merge",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP4_POLICY "
            f"refresh={self.reid_refresh_sec:.2f}s min_score={self.reid_min_score:.2f} "
            f"min_crop={self.reid_min_width}x{self.reid_min_height} "
            f"max_per_camera_update={self.reid_max_per_update} predicted=skip "
            "confirmed_tracks_only=1 stale_policy=drop no_global_id_mutation=1",
            flush=True,
        )

    @staticmethod
    def _crop_quality(score: float, width: int, height: int) -> float:
        size = min(1.0, height / 180.0) * min(1.0, width / 80.0)
        return max(0.05, min(1.0, 0.70 * float(score) + 0.30 * size))

    def _schedule_reid(self, cid: str, snapshots, captured_ns: int) -> None:
        detector = self.detector
        if detector is None or detector.frame is None:
            return
        frame = detector.frame
        frame_h, frame_w = frame.shape[:2]
        refresh_ns = int(self.reid_refresh_sec * 1_000_000_000.0)

        eligible = []
        for snapshot in snapshots:
            if not snapshot.confirmed:
                continue
            if snapshot.predicted or snapshot.since_detection_sec > 0.25:
                self.reid_skip_predicted += 1
                continue
            if float(snapshot.score) < self.reid_min_score:
                self.reid_skip_score += 1
                continue
            key = (cid, snapshot.track_id)
            previous = self.reid_submit_at.get(key, 0)
            if previous and captured_ns - previous < refresh_ns:
                continue
            x1f, y1f, x2f, y2f = snapshot.bbox_xyxy
            x1 = max(0, min(frame_w - 1, int(np.floor(x1f))))
            y1 = max(0, min(frame_h - 1, int(np.floor(y1f))))
            x2 = max(x1 + 1, min(frame_w, int(np.ceil(x2f))))
            y2 = max(y1 + 1, min(frame_h, int(np.ceil(y2f))))
            width = x2 - x1
            height = y2 - y1
            if width < self.reid_min_width or height < self.reid_min_height:
                self.reid_skip_size += 1
                continue
            eligible.append((previous, -float(snapshot.score), snapshot, (x1, y1, x2, y2)))

        # Oldest-never-sampled tracks win first. This prevents a crowded camera from
        # repeatedly feeding the same high-score person while another track starves.
        eligible.sort(key=lambda row: (row[0] != 0, row[0], row[1]))
        for _previous, _neg_score, snapshot, (x1, y1, x2, y2) in eligible[: self.reid_max_per_update]:
            copy_started = time.perf_counter()
            crop = np.ascontiguousarray(frame[y1:y2, x1:x2].copy())
            self.reid_crop_copy_values.append((time.perf_counter() - copy_started) * 1000.0)
            candidate = ReIDCandidate(
                camera_id=cid,
                track_id=snapshot.track_id,
                room_id=self.camera_rooms.get(cid, ""),
                captured_ns=int(captured_ns),
                bbox_xyxy=tuple(float(v) for v in snapshot.bbox_xyxy),
                detector_score=float(snapshot.score),
                quality=self._crop_quality(float(snapshot.score), x2 - x1, y2 - y1),
                crop_bgr=crop,
            )
            if self.reid_scheduler.submit(candidate):
                self.reid_submit_at[(cid, snapshot.track_id)] = int(captured_ns)
                self.reid_submitted_by_camera[cid] += 1
            else:
                self.reid_submit_rejected += 1

    def _consume_tracking(self, cid: str, boxes: list[list[float]], captured_ns: int) -> None:
        # Deliberately duplicate only Step3's small bookkeeping adapter so the frozen
        # Step3 files remain byte-for-byte unchanged while Step4 can see snapshots.
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

        # Main detector/tracker path only performs bounded crop copies + O(1) submit.
        # All resize/preprocess/TRT/matching work happens on the ReID scheduler thread.
        self._schedule_reid(cid, update.snapshots, captured_ns)

    def _on_reid_result(self, result: ReIDResult) -> None:
        candidate = result.candidate
        decision = self.identity_shadow.observe(
            camera_id=candidate.camera_id,
            track_id=candidate.track_id,
            room_id=candidate.room_id,
            embedding=result.embedding,
            quality=candidate.quality,
            captured_at=candidate.captured_ns / 1_000_000_000.0,
        )
        state = str(decision["state"])
        peer = (str(decision["candidate_camera"]), str(decision["candidate_track"]))
        key = candidate.key
        signature = (state, peer[0], peer[1])
        with self.reid_result_lock:
            previous = self.reid_last_match.get(key)
            self.reid_last_match[key] = signature
        if state in {"CANDIDATE", "STRONG"} and signature != previous:
            print(
                "CAMERA_V11_STEP4_REID_MATCH "
                f"mode=shadow camera={candidate.camera_id} track={candidate.track_id} "
                f"peer_camera={peer[0]} peer_track={peer[1]} state={state} "
                f"score={float(decision['score']):.4f} margin={float(decision['margin']):.4f} "
                f"same_room={int(bool(decision['same_room']))} samples={int(decision['gallery_samples'])} "
                f"batch={result.batch_size} queue_wait_ms={result.queue_wait_ms:.1f} merge=0",
                flush=True,
            )

    def _print_stats(self) -> None:
        super()._print_stats()
        scheduler = self.reid_scheduler.snapshot()
        shadow = self.identity_shadow.snapshot()
        batch_hist = scheduler["batch_hist"]
        rows = [
            f"{camera.camera_id}:submitted={self.reid_submitted_by_camera[camera.camera_id]}"
            for camera in self.cameras
        ]
        print(
            "CAMERA_V11_STEP4_REID "
            + " | ".join(rows)
            + f" pending={scheduler['pending']} completed={scheduler['completed']}"
            + f" replaced={scheduler['replaced']} overflow_drop={scheduler['overflow_drops']}"
            + f" stale_drop={scheduler['stale_drops']} submit_rejected={self.reid_submit_rejected}"
            + f" batches={scheduler['batches']} batch1={batch_hist.get(1, 0)} batch2={batch_hist.get(2, 0)}"
            + f" queue_p50={scheduler['queue_p50_ms']:.1f}ms queue_p95={scheduler['queue_p95_ms']:.1f}ms"
            + f" infer_p50={scheduler['infer_p50_ms']:.1f}ms infer_p95={scheduler['infer_p95_ms']:.1f}ms"
            + f" crop_copy_p50={_pct(self.reid_crop_copy_values, 0.50):.3f}ms"
            + f" crop_copy_p95={_pct(self.reid_crop_copy_values, 0.95):.3f}ms"
            + f" skip_predicted={self.reid_skip_predicted} skip_score={self.reid_skip_score}"
            + f" skip_size={self.reid_skip_size} worker_errors={scheduler['worker_errors']}"
            + f" shadow_tracks={shadow['tracks']} observations={shadow['observations']}"
            + f" weak={shadow['weak']} candidates={shadow['candidates']} strong={shadow['strong']}"
            + f" ambiguous={shadow['ambiguous']} identity_merge=0",
            flush=True,
        )

    def run(self) -> int:
        self.reid_scheduler.start()
        print(
            "CAMERA_V11_STEP4_REID_SCHEDULER_READY "
            f"max_batch={self.reid_scheduler.max_batch} max_wait_ms={self.reid_scheduler.max_wait_ms:.1f} "
            f"max_pending={self.reid_scheduler.max_pending} max_age_ms={self.reid_scheduler.max_age_ms:.0f}",
            flush=True,
        )
        try:
            return super().run()
        finally:
            self._close_reid()

    def _close_reid(self) -> None:
        if self._reid_closed:
            return
        self._reid_closed = True
        self.reid_scheduler.close()

    def close(self) -> None:
        self._close_reid()
        super().close()


def main() -> int:
    service = V11Step4TrackingReIDV1()

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
