from __future__ import annotations

import os
import signal

from .bbox_overlay_ipc_v1 import BboxStateWriter, local_track_number
from .bbox_single_target_lock_v1 import SingleTargetBboxLockV1, SingleTargetLockConfigV1
from .step3_single_occupant_tracker_v1 import V11SingleOccupantTrackerV1
from .step3_tracking_v2 import V11Step3TrackingV2


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class V11Step3BboxPublisherV1(V11Step3TrackingV2):
    """Accepted V11 detector path plus one-person conservative bbox publishing.

    Frozen Step3 files are left untouched. This bbox-only runtime swaps only the
    per-camera tracker instances. Existing ByteTrack-style low-confidence recovery
    remains active, but once one confirmed person exists in a camera, unmatched boxes
    cannot mint another local ID until the confirmed occupant has been absent for a
    bounded grace period. During uncertainty the display holds/blanks rather than
    showing a second false-positive box.
    """

    def __init__(self) -> None:
        super().__init__()

        camera_ids = [camera.camera_id for camera in self.cameras]

        # Detector floor/cadence stay unchanged. These knobs affect only NEW-track
        # admission and recovery of the already-confirmed one-person target.
        self.fp_new_track_conf = float(os.environ.get("V11_BBOX_NEW_TRACK_CONF", "0.50"))
        self.fp_confirm_hits = int(os.environ.get("V11_BBOX_TRACK_CONFIRM_HITS", "3"))
        self.fp_tentative_ttl_sec = float(os.environ.get("V11_BBOX_TENTATIVE_TTL_SEC", "1.60"))
        self.fp_duplicate_iou = float(os.environ.get("V11_BBOX_DUPLICATE_IOU", "0.60"))
        self.fp_shadow_sec = float(os.environ.get("V11_BBOX_TRACK_SHADOW_SEC", "1.20"))
        self.fp_max_lost_sec = float(os.environ.get("V11_BBOX_TRACK_MAX_LOST_SEC", "4.50"))
        self.fp_low_recovery_sec = float(os.environ.get("V11_BBOX_LOW_RECOVERY_SEC", "2.50"))
        self.fp_single_occupant_block_sec = float(
            os.environ.get("V11_BBOX_SINGLE_OCCUPANT_BLOCK_SEC", "4.50")
        )

        self.tracker = V11SingleOccupantTrackerV1(
            camera_ids,
            new_track_thresh=self.fp_new_track_conf,
            confirm_hits=self.fp_confirm_hits,
            tentative_ttl_sec=self.fp_tentative_ttl_sec,
            live_duplicate_iou=self.fp_duplicate_iou,
            shadow_sec=self.fp_shadow_sec,
            max_lost_sec=self.fp_max_lost_sec,
            low_recovery_sec=self.fp_low_recovery_sec,
            single_occupant_block_sec=self.fp_single_occupant_block_sec,
        )

        self.confirmed_ids_seen: dict[str, set[str]] = {cid: set() for cid in camera_ids}
        self.bbox_writer = BboxStateWriter()
        self.bbox_writer.reset()
        self.bbox_publish_ok = 0
        self.bbox_publish_errors = 0
        self.single_target = SingleTargetBboxLockV1(
            SingleTargetLockConfigV1(
                enabled=_env_bool("V11_BBOX_SINGLE_TARGET", True),
                acquire_updates=int(os.environ.get("V11_BBOX_LOCK_ACQUIRE_UPDATES", "2")),
                min_hits=int(os.environ.get("V11_BBOX_LOCK_MIN_HITS", str(self.fp_confirm_hits))),
                min_confidence=float(os.environ.get("V11_BBOX_LOCK_MIN_CONF", "0.30")),
                min_area_norm=float(os.environ.get("V11_BBOX_LOCK_MIN_AREA", "0.008")),
                hold_sec=float(os.environ.get("V11_BBOX_LOCK_HOLD_SEC", "1.20")),
                release_sec=float(os.environ.get("V11_BBOX_LOCK_RELEASE_SEC", "4.50")),
                handoff_window_sec=float(os.environ.get("V11_BBOX_LOCK_HANDOFF_SEC", "3.50")),
                handoff_updates=int(os.environ.get("V11_BBOX_LOCK_HANDOFF_UPDATES", "2")),
                handoff_min_iou=float(os.environ.get("V11_BBOX_LOCK_HANDOFF_MIN_IOU", "0.08")),
                handoff_max_center_distance=float(
                    os.environ.get("V11_BBOX_LOCK_HANDOFF_MAX_CENTER", "0.20")
                ),
                handoff_min_area_ratio=float(
                    os.environ.get("V11_BBOX_LOCK_HANDOFF_MIN_AREA_RATIO", "0.35")
                ),
                handoff_max_area_ratio=float(
                    os.environ.get("V11_BBOX_LOCK_HANDOFF_MAX_AREA_RATIO", "2.80")
                ),
            )
        )
        print(
            "CAMERA_V11_BBOX_PUBLISHER_ARCH base=step3-v2 detector_changed=0 "
            "tracker_core_changed=0 tracker_policy=single-occupant-fp-guard-v3 "
            "ipc=atomic-tmpfs latest_only=1 queue=0 reid=0 global_id=0 "
            f"single_target={int(self.single_target.config.enabled)}",
            flush=True,
        )
        print(
            "CAMERA_V11_BBOX_FP_GUARD_POLICY "
            f"new_track_conf={self.fp_new_track_conf:.2f} "
            f"confirm_hits={self.fp_confirm_hits} "
            f"tentative_ttl={self.fp_tentative_ttl_sec:.2f}s "
            f"duplicate_iou={self.fp_duplicate_iou:.2f} "
            f"shadow={self.fp_shadow_sec:.2f}s "
            f"max_lost={self.fp_max_lost_sec:.2f}s "
            f"low_recovery={self.fp_low_recovery_sec:.2f}s "
            f"single_occupant_block={self.fp_single_occupant_block_sec:.2f}s "
            "detector_low=0.18 detector_high=0.30 low_boxes_can_update=1",
            flush=True,
        )
        print(
            "CAMERA_V11_BBOX_SINGLE_TARGET_POLICY "
            f"enabled={int(self.single_target.config.enabled)} "
            f"acquire_updates={self.single_target.config.acquire_updates} "
            f"min_hits={self.single_target.config.min_hits} "
            f"min_conf={self.single_target.config.min_confidence:.2f} "
            f"min_area={self.single_target.config.min_area_norm:.4f} "
            f"hold={self.single_target.config.hold_sec:.2f}s "
            f"release={self.single_target.config.release_sec:.2f}s "
            f"handoff={self.single_target.config.handoff_window_sec:.2f}s "
            f"handoff_updates={self.single_target.config.handoff_updates} "
            f"handoff_iou={self.single_target.config.handoff_min_iou:.2f} "
            f"handoff_center={self.single_target.config.handoff_max_center_distance:.2f}",
            flush=True,
        )

    def _consume_tracking(self, cid: str, boxes: list[list[float]], captured_ns: int) -> None:
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

        raw_tracks = []
        for snapshot in update.snapshots:
            if snapshot.confirmed and snapshot.state != "removed":
                self.confirmed_ids_seen[cid].add(str(snapshot.track_id))
            if not snapshot.confirmed or snapshot.state == "removed":
                continue
            try:
                local_id = local_track_number(snapshot.track_id)
            except ValueError:
                self.track_prefix_errors += 1
                continue
            raw_tracks.append(
                {
                    "track_id": str(snapshot.track_id),
                    "local_id": local_id,
                    "state": str(snapshot.state),
                    "predicted": bool(snapshot.predicted),
                    "confidence": float(snapshot.score),
                    "hits": int(snapshot.hits),
                    "age_sec": float(snapshot.age_sec),
                    "bbox_norm": [float(v) for v in snapshot.bbox_norm],
                    "velocity_norm_s": [float(v) for v in snapshot.velocity_norm_s],
                    "since_detection_sec": float(snapshot.since_detection_sec),
                }
            )

        tracks = self.single_target.select(cid, raw_tracks, captured_ns)
        try:
            self.bbox_writer.publish(cid, captured_ns, tracks)
            self.bbox_publish_ok += 1
        except Exception as exc:
            self.bbox_publish_errors += 1
            if self.bbox_publish_errors <= 3 or self.bbox_publish_errors % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_PUBLISH warning="
                    f"{type(exc).__name__}:{exc} errors={self.bbox_publish_errors}",
                    flush=True,
                )

    def _print_stats(self) -> None:
        super()._print_stats()
        print(
            "CAMERA_V11_BBOX_PUBLISHER "
            f"published={self.bbox_publish_ok} errors={self.bbox_publish_errors} "
            f"path={self.bbox_writer.path}",
            flush=True,
        )
        for camera in self.cameras:
            cid = camera.camera_id
            row = self.single_target.stats(cid)
            print(
                "CAMERA_V11_BBOX_SINGLE_TARGET "
                f"camera={cid} enabled={row['enabled']} locked={row['locked']} "
                f"candidate={row['candidate']} acquired={row['acquired']} "
                f"handoff={row['handoff']} released={row['released']} "
                f"hold_outputs={row['hold_outputs']} suppressed={row['suppressed']} "
                f"input_max={row['input_max']} output_max={row['output_max']} "
                f"violations={row['violations']}",
                flush=True,
            )
            print(
                "CAMERA_V11_BBOX_FP_TRACKER "
                f"camera={cid} confirmed_total={len(self.confirmed_ids_seen[cid])} "
                f"blocked_new={self.tracker.blocked_new_total(cid)}",
                flush=True,
            )


def main() -> int:
    service = V11Step3BboxPublisherV1()

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
