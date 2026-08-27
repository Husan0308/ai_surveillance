from __future__ import annotations

import os
import queue as pyqueue
import sys
import time

from .runtime_bbox_v7_prod import NvDCFProductionBBoxRuntime
from .visibility_policy_v71 import should_hold_last_good


class NvDCFNoFlickerRuntime(NvDCFProductionBBoxRuntime):
    """V7.1: preserve current-frame NvDCF localization without blinking.

    Two measured problems are fixed here:

    1. V7 published an empty cache entry immediately whenever NvDCF emitted no normal
       object metadata for one batch. NvDCF may temporarily put a valid target into
       inactive/shadow state, so this created visible off/on blinking. V7.1 keeps the
       last *real* NvDCF rectangle for a tightly bounded grace window. Coordinates are
       never extrapolated and the original timestamp is preserved.

    2. The serialized Pascal GPU lane was acquired before requesting a detector frame.
       At 20 FPS this wasted roughly one frame period inside the GPU lock. V7.1 captures
       first, then acquires the lane only around TRT submission/result, leaving much more
       time for NvDCF while still preventing NvDCF/TRT compute overlap.
    """

    def __init__(self) -> None:
        self.empty_hold_ms = max(
            120.0,
            min(600.0, float(os.environ.get("CAMERA_V2_DISPLAY_EMPTY_HOLD_MS", "320"))),
        )
        self.v71_empty_holds = 0
        self.v71_empty_expires = 0
        self.v71_capture_before_lane = 0
        super().__init__()
        self.display_track_max_age_ms = max(
            self.display_track_max_age_ms,
            self.empty_hold_ms + 30.0,
        )
        print(
            "CAMERA_BBOX_V71_PROFILE "
            f"empty_hold={self.empty_hold_ms:.0f}ms "
            f"display_max_age={self.display_track_max_age_ms:.0f}ms "
            "last_good_only=1 prediction=0 capture_before_gpu_lane=1",
            flush=True,
        )

    def _tracker_probe(self, _pad, info):
        # Fully override the V7 probe so an empty current tracker batch is handled
        # atomically under track_cache_lock; the display thread can never observe a
        # transient [] between clear and restore.
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
                with self.track_cache_lock:
                    for source_id, tracks in published.items():
                        if tracks:
                            self.track_cache[source_id] = (now, tracks)
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
                            # Deliberately leave both the bbox AND its old timestamp
                            # untouched. Repeated empty batches therefore cannot refresh
                            # a stale rectangle indefinitely.
                            held += 1
                            continue

                        if previous is not None:
                            self.track_cache.pop(source_id, None)
                            expired += 1

                    visible_tracks = [
                        track
                        for _updated, tracks in self.track_cache.values()
                        for track in tracks
                    ]
                    self.tracked_now = len(visible_tracks)
                    self.tracker_batches += 1
                    self.v7_low_conf_filtered += filtered
                    self.v7_duplicates_suppressed += suppressed
                    self.v71_empty_holds += held
                    self.v71_empty_expires += expired

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
                    f"CAMERA_BBOX_V71_TRACK warning={type(exc).__name__}:{exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return self.Gst.PadProbeReturn.OK
        finally:
            if self._tracker_lane_held:
                self._tracker_lane_held = False
                self.gpu_lane.release()

    def _detector_scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "TRT86 startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "TRT86 startup failed")
            return

        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_CLEAN_DETECT_READY "
            f"model={ready.get('model')} backend={ready.get('backend')} "
            f"target={self.detect_hz:.2f}Hz/cam "
            "capture=fresh-before-gpu-lane gpu_lane=trt-only",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        versions = {cid: 0 for cid in ids}
        index = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            cid = ids[index % len(ids)]
            index += 1

            if self.stats[cid].frames <= 0:
                self.det_stop.wait(0.03)
                continue

            # Capture at source cadence before taking the compute lock. The current V7
            # logs showed ~16-20 ms TRT but ~70-82 ms lock hold, proving that the old
            # ordering spent most of the lock waiting for this 20 FPS frame.
            self._request_capture(cid)
            row = self.mailbox.wait_new(cid, versions[cid], timeout=0.8)
            if row is None:
                self._clear_capture(cid)
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.025)
                continue

            version, captured, frame = row
            versions[cid] = version
            self._clear_capture(cid)
            self.v71_capture_before_lane += 1

            wait_started = time.monotonic()
            acquired = False
            while not self.det_stop.is_set():
                if self.gpu_lane.acquire(timeout=0.05):
                    acquired = True
                    break
            if not acquired:
                break

            lane_started = time.monotonic()
            lane_wait_ms = (lane_started - wait_started) * 1000.0
            try:
                try:
                    self.job_q.put(
                        {"cameras": [cid], "frames": [frame], "captured": [captured]},
                        timeout=0.3,
                    )
                    result = self.result_q.get(timeout=5.0)
                except pyqueue.Empty:
                    with self.det_lock:
                        self.det_error = "TRT86 result timeout"
                    self.det_stop.wait(0.05)
                    continue

                if result.get("type") == "fatal":
                    with self.det_lock:
                        self.det_error = result.get("error", "TRT86 fatal")
                    return
                if result.get("type") == "batch_error":
                    with self.det_lock:
                        self.det_error = result.get("error", "TRT86 batch error")
                    self.det_stop.wait(0.10)
                    continue
                if result.get("type") != "result":
                    continue

                completed = time.monotonic()
                rows = result.get("boxes", {}).get(cid, [])
                boxes = self._map_detector_rows(rows)
                self._publish_detector(cid, captured, boxes)
                age_ms = max(0.0, (completed - captured) * 1000.0)
                batch_ms = float(result.get("batch_ms") or 0.0)
                with self.det_lock:
                    self.det_calls += 1
                    self.det_inputs += 1
                    self.det_batch_ms = batch_ms
                    self.det_counts[cid] = len(boxes)
                    self.det_result_age_ms = age_ms
                    self.detector_times[cid].append(completed)
                    self.det_error = ""
            finally:
                lane_hold_ms = (time.monotonic() - lane_started) * 1000.0
                with self.lane_stats_lock:
                    self.lane_detector_wait_ms.append(lane_wait_ms)
                    self.lane_detector_hold_ms.append(lane_hold_ms)
                self.gpu_lane.release()

            desired_interval = 1.0 / max(0.1, self.detect_hz * len(ids))
            elapsed = time.monotonic() - cycle_started
            self.det_stop.wait(max(0.005, desired_interval - elapsed))

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_BBOX_V71_STATS "
            f"empty_holds={self.v71_empty_holds} "
            f"empty_expires={self.v71_empty_expires} "
            f"capture_before_lane={self.v71_capture_before_lane} "
            f"hold_ms={self.empty_hold_ms:.0f} predictor=0",
            flush=True,
        )
        return keep


def main() -> int:
    return NvDCFNoFlickerRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
