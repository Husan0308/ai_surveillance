from __future__ import annotations

import os
import sys
import time

from .runtime_bbox_v7_prod import NvDCFProductionBBoxRuntime
from .visibility_policy_v72 import should_hold_last_good


class NvDCFStableBBoxRuntime(NvDCFProductionBBoxRuntime):
    """V7.2: V7 compute scheduling plus bounded no-flicker visibility.

    V7.1 moved detector frame capture outside the serialized GPU lane. Live Pascal
    measurements showed that this was a regression: TRT batch latency jumped from
    about 16-20 ms to about 170-190 ms, source FPS fell into the low teens and NvDCF
    collapsed to about 3 Hz. V7.2 deliberately inherits the original V7 detector
    scheduler unchanged, so detector conversion/capture + TRT remain serialized from
    tracker work exactly as in the last known-good live run.

    The only temporal display change versus V7 is a bounded last-real-box grace window
    for empty normal NvDCF output batches. This suppresses Active/Inactive metadata
    blinking without synthesizing motion. The stored bbox coordinates and timestamp are
    never refreshed by empty batches, so a stale box cannot live indefinitely.
    """

    def __init__(self) -> None:
        self.empty_hold_ms = max(
            120.0,
            min(500.0, float(os.environ.get("CAMERA_V2_DISPLAY_EMPTY_HOLD_MS", "300"))),
        )
        self.v72_empty_holds = 0
        self.v72_empty_expires = 0
        self.v72_real_updates = 0
        super().__init__()
        self.display_track_max_age_ms = max(
            self.display_track_max_age_ms,
            self.empty_hold_ms + 40.0,
        )
        print(
            "CAMERA_BBOX_V72_PROFILE "
            f"empty_hold={self.empty_hold_ms:.0f}ms "
            f"display_max_age={self.display_track_max_age_ms:.0f}ms "
            "last_good_only=1 prediction=0 detector_scheduler=v7-serialized-proven",
            flush=True,
        )

    def _tracker_probe(self, _pad, info):
        try:
            if not self.analytics_enabled:
                return self.Gst.PadProbeReturn.OK
            buffer = info.get_buffer()
            if buffer is None:
                return self.Gst.PadProbeReturn.OK
            try:
                rows = self.bridge.copy_tracks(buffer, max_rows=256)
                now = time.monotonic()
                grouped = {int(source_id): [] for source_id in self.index_camera}
                sx = self.display_width / float(self.track_width)
                sy = self.display_height / float(self.track_height)
                filtered = 0

                for row in rows:
                    source_id = int(row["source_id"])
                    if source_id not in grouped:
                        continue
                    conf = float(row["tracker_confidence"])
                    if conf < 0.0:
                        conf = float(row["confidence"])
                    if conf < self.min_display_track_conf:
                        filtered += 1
                        continue

                    left = float(row["left"]) * sx
                    top = float(row["top"]) * sy
                    right = (float(row["left"]) + float(row["width"])) * sx
                    bottom = (float(row["top"]) + float(row["height"])) * sy
                    object_id = int(row["object_id"])
                    raw_box = (left, top, right, bottom)
                    self._record_jump(source_id, object_id, raw_box)
                    display_box = self._stable_display_box(
                        source_id,
                        object_id,
                        raw_box,
                        now,
                    )
                    grouped[source_id].append(
                        (
                            object_id,
                            display_box[0],
                            display_box[1],
                            display_box[2],
                            display_box[3],
                            conf,
                        )
                    )

                published = {}
                suppressed = 0
                for source_id, tracks in grouped.items():
                    kept, count = self._dedup_v7(tracks)
                    published[source_id] = kept
                    suppressed += count

                held = 0
                expired = 0
                real_updates = 0
                with self.track_cache_lock:
                    for source_id, tracks in published.items():
                        if tracks:
                            self.track_cache[source_id] = (now, tracks)
                            real_updates += 1
                            continue

                        previous = self.track_cache.get(source_id)
                        if (
                            previous is not None
                            and previous[1]
                            and should_hold_last_good(
                                previous[0],
                                now,
                                self.empty_hold_ms,
                            )
                        ):
                            # Keep the last REAL NvDCF bbox and its original timestamp.
                            # Empty batches cannot move it and cannot refresh its lifetime.
                            held += 1
                            continue

                        if previous is not None:
                            self.track_cache.pop(source_id, None)
                            expired += 1

                    self.tracked_now = sum(
                        len(tracks) for _updated, tracks in self.track_cache.values()
                    )
                    self.tracker_batches += 1
                    self.v7_low_conf_filtered += filtered
                    self.v7_duplicates_suppressed += suppressed
                    self.v72_empty_holds += held
                    self.v72_empty_expires += expired
                    self.v72_real_updates += real_updates

                    active_keys = {
                        (source_id, int(track[0]))
                        for source_id, (_updated, tracks) in self.track_cache.items()
                        for track in tracks
                    }

                for key, state in list(self._display_sizes.items()):
                    if key not in active_keys and now - state.seen_at > 1.0:
                        self._display_sizes.pop(key, None)
                        self._last_raw_boxes.pop(key, None)
            except Exception as exc:
                print(
                    f"CAMERA_BBOX_V72_TRACK warning={type(exc).__name__}:{exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return self.Gst.PadProbeReturn.OK
        finally:
            if self._tracker_lane_held:
                self._tracker_lane_held = False
                self.gpu_lane.release()

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_BBOX_V72_STATS "
            f"real_updates={self.v72_real_updates} "
            f"empty_holds={self.v72_empty_holds} "
            f"empty_expires={self.v72_empty_expires} "
            f"hold_ms={self.empty_hold_ms:.0f} "
            "prediction=0 capture_before_lane=0",
            flush=True,
        )
        return keep


def main() -> int:
    return NvDCFStableBBoxRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
