from __future__ import annotations

import os
import signal
import threading

from .step4_reid_gallery_v1 import DiverseReIDGalleryV1
from .step4_reid_quality_runtime_v1 import (
    FROZEN_PRODUCTION_SHA,
    GateCandidate,
    ReIDCropQualityDecision,
    V11Step4ReIDQualityV1,
)
from .step4_reid_scheduler_v1 import (
    ReIDCandidateV1,
    ReIDResultV1,
    V11ReIDSchedulerV1,
)


class V11Step4ReIDGalleryRuntimeV1(V11Step4ReIDQualityV1):
    """Passing Step-1 crops -> async FP32 embeddings -> local-track galleries."""

    def __init__(self) -> None:
        super().__init__()
        self.reid_sampling_interval_ns = int(
            max(1.0, float(os.environ.get("V11_STEP4_REID_SAMPLE_SEC", "1.0")))
            * 1_000_000_000.0
        )
        self.quality_run_sec = max(
            0.0,
            float(
                os.environ.get(
                    "V11_STEP4_GALLERY_RUN_SEC", str(self.quality_run_sec)
                )
            ),
        )
        self.reid_gallery = DiverseReIDGalleryV1(
            expected_dimension=256,
            capacity=8,
            bootstrap_samples=3,
            duplicate_cosine=0.975,
            quality_replace_gain=0.08,
            expiry_sec=12.0,
        )
        self.reid_scheduler = V11ReIDSchedulerV1(
            self._on_reid_result,
            max_batch=2,
            max_wait_ms=4.0,
            max_pending=12,
            max_age_ms=1000.0,
        )
        self.reid_submit_at: dict[tuple[str, str], int] = {}
        self.active_track_ids: dict[str, frozenset[str]] = {
            camera.camera_id: frozenset() for camera in self.cameras
        }
        self.reid_submit_lock = threading.RLock()
        self.reid_sampling_skip = 0
        self.reid_submit_rejected = 0
        self.reid_inactive_result_drop = 0
        self.reid_closed = False
        print(
            "CAMERA_V11_STEP4_REID_GALLERY_V1_ARCH "
            f"frozen_production_sha={FROZEN_PRODUCTION_SHA} "
            "source=step1-accepted-native-crops engine=known-good-resnet50-fp32-trt86 "
            "embedding_dimension=256 normalized=required worker=one async=1 "
            "camera_tracker_wait_for_reid=0 pending=bounded-12 "
            "queue=latest-only keyed_by=camera_id+local_track_id "
            "pair_scoring=0 cross_camera_decision=0 reciprocal=0 one_to_one=0 "
            "room_id=0 global_id=0 face=0 handoff=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP4_REID_GALLERY_V1_POLICY "
            f"sample_interval={self.reid_sampling_interval_ns / 1e9:.3f}s "
            "gallery_capacity=8 bootstrap=3 duplicate_cosine=0.975 "
            "quality_replace_gain=0.080 "
            "full_retention=quality50+nearest_diversity35+recency15 deterministic=1 "
            "prototype_average=0",
            flush=True,
        )

    def _quality_track_update(
        self, camera_id: str, track_ids: tuple[str, ...], captured_ns: int
    ) -> None:
        active = frozenset(str(track_id) for track_id in track_ids)
        self.active_track_ids[camera_id] = active
        self.reid_gallery.touch_active(camera_id, active, int(captured_ns))
        with self.reid_submit_lock:
            for key in [
                key
                for key in self.reid_submit_at
                if key[0] == camera_id and key[1] not in active
            ]:
                del self.reid_submit_at[key]

    def _accepted_crop(
        self, candidate: GateCandidate, decision: ReIDCropQualityDecision
    ) -> None:
        # This callback runs on the Step-1 CPU gate worker, never the camera or
        # tracker thread. Scheduler submission only takes a short condition lock.
        crop = decision.crop_bgr
        if crop is None or decision.source_bbox_xyxy is None:
            self.reid_submit_rejected += 1
            return
        key = (candidate.camera_id, candidate.track_id)
        if candidate.track_id not in self.active_track_ids.get(
            candidate.camera_id, frozenset()
        ):
            self.reid_inactive_result_drop += 1
            return
        with self.reid_submit_lock:
            previous = self.reid_submit_at.get(key, 0)
            if (
                previous
                and int(candidate.captured_ns) - previous
                < self.reid_sampling_interval_ns
            ):
                self.reid_sampling_skip += 1
                return
            request = ReIDCandidateV1(
                camera_id=candidate.camera_id,
                local_track_id=candidate.track_id,
                captured_ns=int(candidate.captured_ns),
                bbox_xyxy=tuple(
                    float(value) for value in decision.source_bbox_xyxy
                ),
                detector_confidence=float(candidate.detector_score),
                quality_score=float(decision.quality_score),
                crop_bgr=crop,
            )
            if self.reid_scheduler.submit(request):
                self.reid_submit_at[key] = int(candidate.captured_ns)
            else:
                self.reid_submit_rejected += 1

    def _on_reid_result(self, result: ReIDResultV1) -> None:
        candidate = result.candidate
        if candidate.local_track_id not in self.active_track_ids.get(
            candidate.camera_id, frozenset()
        ):
            self.reid_inactive_result_drop += 1
            return
        self.reid_gallery.update(
            camera_id=candidate.camera_id,
            local_track_id=candidate.local_track_id,
            timestamp_ns=candidate.captured_ns,
            embedding=result.embedding,
            quality_score=candidate.quality_score,
            detector_confidence=candidate.detector_confidence,
            bbox_xyxy=candidate.bbox_xyxy,
        )

    def _print_gallery_stats(self) -> None:
        reid = self.reid_scheduler.snapshot()
        gallery = self.reid_gallery.snapshot()
        print(
            "CAMERA_V11_STEP4_REID_GALLERY_V1 "
            f"reid_submitted={reid['reid_submitted']} "
            f"reid_completed={reid['reid_completed']} "
            f"reid_pending={reid['reid_pending']} "
            f"reid_replaced_pending={reid['reid_replaced_pending']} "
            f"reid_overflow_drop={reid['reid_overflow_drop']} "
            f"reid_stale_drop={reid['reid_stale_drop']} "
            f"reid_worker_errors={reid['reid_worker_errors']} "
            f"gallery_tracks={gallery['gallery_tracks']} "
            f"gallery_samples={gallery['gallery_samples']} "
            f"gallery_tracks_ge3={gallery['gallery_tracks_ge3']} "
            f"gallery_max_samples={gallery['gallery_max_samples']} "
            f"gallery_bootstrap_add={gallery['gallery_bootstrap_add']} "
            f"gallery_diverse_add={gallery['gallery_diverse_add']} "
            f"gallery_duplicate_drop={gallery['gallery_duplicate_drop']} "
            f"gallery_quality_replace={gallery['gallery_quality_replace']} "
            "gallery_full_reject_or_replace="
            f"{gallery['gallery_full_reject_or_replace']} "
            f"gallery_invalid_reject={gallery['gallery_invalid_reject']} "
            f"gallery_cleanup={gallery['gallery_cleanup']} "
            f"reid_queue_p50_ms={reid['reid_queue_p50_ms']:.3f}ms "
            f"reid_queue_p95_ms={reid['reid_queue_p95_ms']:.3f}ms "
            f"reid_infer_p50_ms={reid['reid_infer_p50_ms']:.3f}ms "
            f"reid_infer_p95_ms={reid['reid_infer_p95_ms']:.3f}ms "
            f"gallery_update_p50_ms={gallery['gallery_update_p50_ms']:.3f}ms "
            f"gallery_update_p95_ms={gallery['gallery_update_p95_ms']:.3f}ms "
            f"sampling_skip={self.reid_sampling_skip} "
            f"submit_rejected={self.reid_submit_rejected} "
            f"inactive_result_drop={self.reid_inactive_result_drop}",
            flush=True,
        )

    def _print_stats(self) -> None:
        super()._print_stats()
        self._print_gallery_stats()

    def run(self) -> int:
        try:
            self.reid_scheduler.start()
            return super().run()
        finally:
            self._close_reid()

    def _close_reid(self) -> None:
        if self.reid_closed:
            return
        self.reid_closed = True
        self.reid_scheduler.close(drain=True, timeout_sec=8.0)
        self._print_gallery_stats()

    def close(self) -> None:
        self._close_reid()
        super().close()


def main() -> int:
    service = V11Step4ReIDGalleryRuntimeV1()

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
