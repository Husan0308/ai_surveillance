from __future__ import annotations

import os
import signal
from pathlib import Path

import yaml

from .step4_reid_gallery_runtime_v1 import V11Step4ReIDGalleryRuntimeV1
from .step4_reid_pair_shadow_cached_v2 import V11GalleryPairShadowWorkerCachedV2
from .step4_reid_scheduler_v1 import ReIDResultV1


ROOT = Path(__file__).resolve().parents[2]


def load_camera_rooms_v1(
    camera_ids: tuple[str, ...], config_path: str | Path | None = None
) -> dict[str, str]:
    path = Path(config_path or os.environ.get("CAMERA_CONFIG", ROOT / "config/cameras.yaml"))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rooms = {
        str(row.get("id", "")).strip(): str(row.get("room", "")).strip()
        for row in raw.get("cameras") or []
        if bool(row.get("enabled", True))
    }
    missing = [camera_id for camera_id in camera_ids if not rooms.get(camera_id)]
    if missing:
        raise ValueError(f"camera room metadata missing for: {','.join(missing)}")
    return {camera_id: rooms[camera_id] for camera_id in camera_ids}


class V11Step4ReIDPairRuntimeV1(V11Step4ReIDGalleryRuntimeV1):
    """Step 2 galleries plus asynchronous, diagnostics-only multi-shot scoring."""

    def __init__(self) -> None:
        super().__init__()
        self.quality_run_sec = max(
            0.0,
            float(
                os.environ.get("V11_STEP4_PAIR_RUN_SEC", str(self.quality_run_sec))
            ),
        )
        camera_ids = tuple(camera.camera_id for camera in self.cameras)
        self.pair_camera_rooms = load_camera_rooms_v1(camera_ids)
        tsv_setting = os.environ.get(
            "V11_STEP4_PAIR_TSV", str(ROOT / "artifacts/reid/step4_pair_scores_v1.tsv")
        ).strip()
        self.pair_tsv_path = None if tsv_setting.lower() in ("", "0", "off", "none") else tsv_setting
        self.pair_worker = V11GalleryPairShadowWorkerCachedV2(
            self.reid_gallery.gallery_views,
            self.pair_camera_rooms,
            tsv_path=self.pair_tsv_path,
            max_candidates=24,
            recent_sec=12.0,
        )
        self.pair_closed = False
        room_map = ",".join(
            f"{camera_id}:{room}" for camera_id, room in sorted(self.pair_camera_rooms.items())
        )
        print(
            "CAMERA_V11_STEP4_REID_PAIR_SCORER_V1_ARCH "
            "input=independent-local-track-galleries samples=3..8 matrix=max8x8 "
            "compute=cpu-numpy gpu_inference=0 worker=one async=1 dirty_slot=latest-only cadence=2.0s "
            "camera_display_block=0 tracker_mutation=0 identity_decision=0 threshold=0 "
            "reciprocal=0 one_to_one=0 room_id_assignment=0 global_id=0 "
            "provisional_confirmed=0 face=0 cross_room_handoff=0 "
            "candidate_scope=active-recent-cross-camera max_candidates=24 "
            "evidence_cache=shared-validated-gallery-matrix-v2 "
            "priority=CAM-01+CAM-04",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP4_REID_PAIR_SCORER_V1_FORMULA "
            "robust_score=0.40*top3_mean+0.25*median_of_best_matches+"
            "0.20*p75_score+0.15*max_score diagnostics_only=1 "
            f"rooms={room_map} tsv={self.pair_tsv_path or 'disabled'} raw_embeddings_tsv=0",
            flush=True,
        )

    def _on_reid_result(self, result: ReIDResultV1) -> None:
        super()._on_reid_result(result)
        self.pair_worker.notify()

    def _print_pair_stats(self) -> None:
        row = self.pair_worker.snapshot()
        print(
            "CAMERA_V11_STEP4_REID_PAIR_SCORER_V1 "
            f"pairs_considered={row['pairs_considered']} "
            f"pairs_scored={row['pairs_scored']} "
            f"pairs_insufficient={row['pairs_insufficient']} "
            f"pairs_invalid={row['pairs_invalid']} "
            f"same_room_pairs={row['same_room_pairs']} "
            f"different_room_pairs={row['different_room_pairs']} "
            f"score_p50={row['score_p50_ms']:.3f}ms "
            f"score_p95={row['score_p95_ms']:.3f}ms "
            f"worker_errors={row['worker_errors']}",
            flush=True,
        )

    def _print_stats(self) -> None:
        super()._print_stats()
        self._print_pair_stats()

    def run(self) -> int:
        try:
            self.pair_worker.start()
            return super().run()
        finally:
            self._close_pair_worker()

    def _close_pair_worker(self) -> None:
        if self.pair_closed:
            return
        self.pair_closed = True
        self.pair_worker.close(timeout_sec=3.0)
        self._print_pair_stats()

    def close(self) -> None:
        self._close_pair_worker()
        super().close()


def main() -> int:
    service = V11Step4ReIDPairRuntimeV1()

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
