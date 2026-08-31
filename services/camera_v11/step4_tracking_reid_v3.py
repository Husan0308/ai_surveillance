from __future__ import annotations

import os
import signal
import time

import numpy as np

from .step2_production_fp32 import _pct
from .step4_reid_quality_diversity_v1 import V11ReIDQualityDiversityGateV1
from .step4_reid_scheduler_v1 import ReIDCandidate, ReIDResult
from .step4_tracking_reid_v2 import V11Step4TrackingReIDV2


class V11Step4TrackingReIDV3(V11Step4TrackingReIDV2):
    """Step4 V2 + cheap crop quality gate + bounded diverse local ReID gallery.

    This step only improves what evidence is allowed into the existing identity
    layers. It does not change detector/tracker IDs, room thresholds, handoff
    topology, or production/global identity state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reid_quality_gate = V11ReIDQualityDiversityGateV1(
            min_quality=float(os.environ.get("V11_STEP4_REID_Q_MIN", "0.34")),
            min_blur=float(os.environ.get("V11_STEP4_REID_Q_MIN_BLUR", "18")),
            min_aspect_hw=float(os.environ.get("V11_STEP4_REID_Q_MIN_ASPECT", "0.90")),
            max_aspect_hw=float(os.environ.get("V11_STEP4_REID_Q_MAX_ASPECT", "6.0")),
            reject_edge_contacts=int(os.environ.get("V11_STEP4_REID_Q_EDGE_CONTACTS", "2")),
            duplicate_cosine=float(os.environ.get("V11_STEP4_REID_DIVERSITY_COS", "0.975")),
            replace_quality_gain=float(os.environ.get("V11_STEP4_REID_DIVERSITY_REPLACE_GAIN", "0.08")),
            gallery_size=int(os.environ.get("V11_STEP4_REID_DIVERSITY_GALLERY", "8")),
            bootstrap_samples=int(os.environ.get("V11_STEP4_REID_DIVERSITY_BOOTSTRAP", "3")),
        )
        self.reid_quality_reject_log_at: dict[tuple[str, str, str], int] = {}
        self.reid_diversity_drop_log_at: dict[tuple[str, str, str], int] = {}
        self.reid_gate_log_interval_ns = int(
            max(2.0, float(os.environ.get("V11_STEP4_REID_GATE_LOG_SEC", "5.0")))
            * 1_000_000_000.0
        )
        print(
            "CAMERA_V11_STEP4_REID_QUALITY_V1 "
            f"min_quality={self.reid_quality_gate.min_quality:.2f} "
            f"min_blur={self.reid_quality_gate.min_blur:.1f} "
            f"aspect={self.reid_quality_gate.min_aspect_hw:.2f}-{self.reid_quality_gate.max_aspect_hw:.2f} "
            f"edge_contacts_reject={self.reid_quality_gate.reject_edge_contacts} "
            f"bootstrap={self.reid_quality_gate.bootstrap_samples} "
            f"diversity_cos={self.reid_quality_gate.duplicate_cosine:.3f} "
            f"gallery={self.reid_quality_gate.gallery_size} "
            "scope=evidence-only tracker_mutation=0 room_threshold_mutation=0 global_merge=0",
            flush=True,
        )

    def _emit_gate_reject(self, candidate_key: tuple[str, str], reason: str, captured_ns: int, text: str) -> None:
        key = (candidate_key[0], candidate_key[1], reason)
        previous = self.reid_quality_reject_log_at.get(key, 0)
        if previous and captured_ns - previous < self.reid_gate_log_interval_ns:
            return
        self.reid_quality_reject_log_at[key] = captured_ns
        print("CAMERA_V11_STEP4_REID_QUALITY_REJECT " + text, flush=True)

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

        eligible.sort(key=lambda row: (row[0] != 0, row[0], row[1]))
        for _previous, _neg_score, snapshot, (x1, y1, x2, y2) in eligible[: self.reid_max_per_update]:
            copy_started = time.perf_counter()
            crop = np.ascontiguousarray(frame[y1:y2, x1:x2].copy())
            self.reid_crop_copy_values.append((time.perf_counter() - copy_started) * 1000.0)

            gate = self.reid_quality_gate.evaluate_crop(
                crop_bgr=crop,
                detector_score=float(snapshot.score),
                frame_width=frame_w,
                frame_height=frame_h,
                bbox_xyxy=(x1, y1, x2, y2),
            )
            if not gate.accepted:
                self._emit_gate_reject(
                    (cid, snapshot.track_id),
                    gate.reason,
                    int(captured_ns),
                    f"camera={cid} track={snapshot.track_id} reason={gate.reason} "
                    f"quality={gate.quality:.3f} blur={gate.blur:.1f} "
                    f"aspect={gate.aspect_hw:.2f} edge_contacts={gate.edge_contacts} "
                    f"gate_ms={gate.step_ms:.3f}",
                )
                continue

            candidate = ReIDCandidate(
                camera_id=cid,
                track_id=snapshot.track_id,
                room_id=self.camera_rooms.get(cid, ""),
                captured_ns=int(captured_ns),
                bbox_xyxy=tuple(float(v) for v in snapshot.bbox_xyxy),
                detector_score=float(snapshot.score),
                quality=float(gate.quality),
                crop_bgr=crop,
            )
            if self.reid_scheduler.submit(candidate):
                self.reid_submit_at[(cid, snapshot.track_id)] = int(captured_ns)
                self.reid_submitted_by_camera[cid] += 1
            else:
                self.reid_submit_rejected += 1

    def _on_reid_result(self, result: ReIDResult) -> None:
        candidate = result.candidate
        accepted, reason, nearest = self.reid_quality_gate.accept_embedding(
            camera_id=candidate.camera_id,
            track_id=candidate.track_id,
            embedding=result.embedding,
            quality=candidate.quality,
        )
        if not accepted:
            key = (candidate.camera_id, candidate.track_id, reason)
            previous = self.reid_diversity_drop_log_at.get(key, 0)
            if not previous or candidate.captured_ns - previous >= self.reid_gate_log_interval_ns:
                self.reid_diversity_drop_log_at[key] = candidate.captured_ns
                print(
                    "CAMERA_V11_STEP4_REID_DIVERSITY_DROP "
                    f"camera={candidate.camera_id} track={candidate.track_id} reason={reason} "
                    f"nearest_cos={nearest:.4f} quality={candidate.quality:.3f} "
                    "downstream_gallery_update=0",
                    flush=True,
                )
            return
        super()._on_reid_result(result)

    def _print_stats(self) -> None:
        super()._print_stats()
        gate = self.reid_quality_gate.snapshot()
        print(
            "CAMERA_V11_STEP4_REID_GATE "
            f"crop_checked={gate['crop_checked']} crop_accepted={gate['crop_accepted']} "
            f"reject_edge={gate['reject_edge']} reject_blur={gate['reject_blur']} "
            f"reject_aspect={gate['reject_aspect']} reject_quality={gate['reject_quality']} "
            f"embedding_checked={gate['embedding_checked']} "
            f"embedding_accepted={gate['embedding_accepted']} bootstrap={gate['bootstrap_accepted']} "
            f"duplicate_drop={gate['duplicate_drops']} duplicate_replace={gate['duplicate_replacements']} "
            f"gate_p50={gate['gate_p50_ms']:.3f}ms gate_p95={gate['gate_p95_ms']:.3f}ms "
            f"blur_p50={gate['blur_p50']:.1f} quality_p50={gate['quality_p50']:.3f} "
            f"nearest_cos_p50={gate['nearest_cos_p50']:.4f} nearest_cos_p95={gate['nearest_cos_p95']:.4f} "
            f"gate_tracks={gate['tracks']} tracker_mutation=0 global_merge=0",
            flush=True,
        )


def main() -> int:
    service = V11Step4TrackingReIDV3()

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
