from __future__ import annotations

import signal

from .bbox_overlay_ipc_v1 import BboxStateWriter, local_track_number
from .step3_tracking_v2 import V11Step3TrackingV2


class V11Step3BboxPublisherV1(V11Step3TrackingV2):
    """Accepted V11 Step3 tracker plus a latest-only bbox metadata side channel."""

    def __init__(self) -> None:
        super().__init__()
        self.bbox_writer = BboxStateWriter()
        self.bbox_writer.reset()
        self.bbox_publish_ok = 0
        self.bbox_publish_errors = 0
        print(
            "CAMERA_V11_BBOX_PUBLISHER_ARCH base=step3-v2 tracker_changed=0 detector_changed=0 "
            "ipc=atomic-tmpfs latest_only=1 queue=0 reid=0 global_id=0",
            flush=True,
        )

    def _consume_tracking(self, cid: str, boxes: list[list[float]], captured_ns: int) -> None:
        # Deliberately mirrors frozen Step3 bookkeeping. The only new action is
        # publishing already-computed snapshots after the tracker update.
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

        tracks = []
        for snapshot in update.snapshots:
            if not snapshot.confirmed or snapshot.state == "removed":
                continue
            try:
                local_id = local_track_number(snapshot.track_id)
            except ValueError:
                self.track_prefix_errors += 1
                continue
            tracks.append(
                {
                    "track_id": str(snapshot.track_id),
                    "local_id": local_id,
                    "state": str(snapshot.state),
                    "predicted": bool(snapshot.predicted),
                    "confidence": float(snapshot.score),
                    "bbox_norm": [float(v) for v in snapshot.bbox_norm],
                    "velocity_norm_s": [float(v) for v in snapshot.velocity_norm_s],
                    "since_detection_sec": float(snapshot.since_detection_sec),
                }
            )
        try:
            self.bbox_writer.publish(cid, captured_ns, tracks)
            self.bbox_publish_ok += 1
        except Exception as exc:
            # Overlay failure must never take down detection/tracking/cameras.
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
