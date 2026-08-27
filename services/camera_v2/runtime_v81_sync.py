from __future__ import annotations

import math
import os
import sys
import time
from collections import deque

from .runtime_v8_pascal import PascalBatchLowLatencyRuntime


class PascalStickySyncRuntime(PascalBatchLowLatencyRuntime):
    """V8.1: keep V8 batch-6 throughput, fix temporal/visibility bbox errors.

    V8 solved GPU starvation, but two independent temporal bugs remained:

    1. The display/native copy threshold was 0.28 and V8 also raised NvDCF's
       minTrackerConfidence back to 0.28. A moving DCF target can briefly score below
       that while still being a valid visual track, which made the downstream bbox
       disappear before Python could even observe the low confidence.

    2. An asynchronous detector result is ~100-200 ms old when it reaches the tracker
       mux. Injecting that old rectangle into a current video frame pulls a moving
       NvDCF target backward. Existing targets therefore use their newest real NvDCF
       geometry when a stale detector observation is associated with them. Detector
       confidence remains detector confidence; only stale geometry is currentized.

    This runtime deliberately does NOT bring back a long CPU velocity predictor.
    The visible center comes from real NvDCF measurements. Cache hold is short and
    bounded, and sources absent from a tracker output batch are no longer fabricated
    as explicit empty updates.
    """

    def __init__(self) -> None:
        self.v81_currentize_after_ms = max(
            40.0,
            min(180.0, float(os.environ.get("CAMERA_V81_CURRENTIZE_AFTER_MS", "70"))),
        )
        self.v81_new_target_max_age_ms = max(
            100.0,
            min(400.0, float(os.environ.get("CAMERA_V81_NEW_TARGET_MAX_AGE_MS", "240"))),
        )
        self.v81_raw_track_max_age_ms = max(
            80.0,
            min(350.0, float(os.environ.get("CAMERA_V81_RAW_TRACK_MAX_AGE_MS", "180"))),
        )
        self.v81_empty_detector_skip_age_ms = max(
            40.0,
            min(250.0, float(os.environ.get("CAMERA_V81_EMPTY_DETECTOR_SKIP_AGE_MS", "80"))),
        )
        self.v81_detector_currentized = 0
        self.v81_detector_raw_new = 0
        self.v81_detector_stale_dropped = 0
        self.v81_empty_detector_skips = 0
        self.v81_cache_prunes = 0
        self.v81_real_track_updates = 0
        self.v81_batches_with_tracks = 0
        self.v81_track_conf_samples: deque[float] = deque(maxlen=4096)
        self.v81_overlay_age_samples: deque[float] = deque(maxlen=4096)
        self.v81_overlay_draws = 0
        self.v81_overlay_held_draws = 0
        self._latest_raw_tracks: dict[
            int,
            tuple[float, list[tuple[int, float, float, float, float, float]]],
        ] = {}
        super().__init__()
        print(
            "CAMERA_V81_SYNC "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"native_display_conf={self.min_display_track_conf:.2f} "
            f"currentize_after={self.v81_currentize_after_ms:.0f}ms "
            f"new_target_max_age={self.v81_new_target_max_age_ms:.0f}ms "
            f"raw_track_max_age={self.v81_raw_track_max_age_ms:.0f}ms "
            f"cache_hold={self.empty_hold_ms:.0f}ms predictor=0",
            flush=True,
        )

    def _prepare_tracker_files(self):
        lib, generated = super()._prepare_tracker_files()
        lines = generated.read_text(encoding="utf-8").splitlines()

        # Restore the lower confidence policy that QualityCameraRuntime already used.
        # V8 accidentally overwrote it with 0.28. The display/native extraction floor
        # is separately controlled by CAMERA_V2_MIN_DISPLAY_TRACK_CONF (default 0.10
        # in the V8.1 launcher), so valid low-confidence moving targets remain visible.
        self._replace_yaml_key(lines, "minTrackerConfidence", "0.12")
        self._replace_yaml_key(lines, "probationAge", "1")
        self._replace_yaml_key(lines, "earlyTerminationAge", "1")
        shadow_frames = max(12, int(round(self.track_fps * 1.50)))
        self._replace_yaml_key(lines, "maxShadowTrackingAge", str(shadow_frames))
        self._replace_yaml_key(
            lines,
            "minTrackingConfidenceDuringInactive",
            "0.12",
            required=False,
        )
        self._insert_target_management_key(lines, "outputShadowTracks", "0")
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_V81_NVDCF "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            "min_tracker_conf=0.12 probation=1 early_termination=1 "
            f"shadow_frames={shadow_frames} output_shadow=0 "
            "features=ColorNames hog=0 feature_level=2",
            flush=True,
        )
        return lib, generated

    @staticmethod
    def _percentile(values, p: float) -> float:
        rows = sorted(float(v) for v in values)
        if not rows:
            return 0.0
        idx = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * float(p)))))
        return rows[idx]

    @staticmethod
    def _center_distance(a, b) -> float:
        acx = 0.5 * (a[0] + a[2])
        acy = 0.5 * (a[1] + a[3])
        bcx = 0.5 * (b[0] + b[2])
        bcy = 0.5 * (b[1] + b[3])
        return math.hypot(acx - bcx, acy - bcy)

    @staticmethod
    def _diag(box) -> float:
        return math.hypot(max(2.0, box[2] - box[0]), max(2.0, box[3] - box[1]))

    def _match_detector_to_raw_tracks(self, source_id: int, boxes, now: float):
        latest = self._latest_raw_tracks.get(int(source_id))
        if latest is None:
            return list(boxes), 0, 0
        updated, raw_tracks = latest
        raw_age_ms = max(0.0, (now - updated) * 1000.0)
        if raw_age_ms > self.v81_raw_track_max_age_ms or not raw_tracks:
            return list(boxes), 0, 0

        used: set[int] = set()
        output = []
        matched = 0
        unmatched = 0
        for det in boxes:
            dbox = tuple(float(v) for v in det[:4])
            dconf = float(det[4])
            best_index = -1
            best_score = -1.0
            for index, track in enumerate(raw_tracks):
                if index in used:
                    continue
                tbox = tuple(float(v) for v in track[1:5])
                iou = self._iou(dbox, tbox)
                distance = self._center_distance(dbox, tbox)
                scale = max(self._diag(dbox), self._diag(tbox), 4.0)
                near = max(0.0, 1.0 - distance / (0.85 * scale))
                if iou < 0.05 and distance > 0.65 * scale:
                    continue
                score = iou + 0.35 * near
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index >= 0:
                used.add(best_index)
                track = raw_tracks[best_index]
                output.append(
                    (
                        float(track[1]),
                        float(track[2]),
                        float(track[3]),
                        float(track[4]),
                        dconf,
                    )
                )
                matched += 1
            else:
                output.append(det)
                unmatched += 1
        return output, matched, unmatched

    def _inject_detector_probe(self, _pad, info):
        """Inject detector semantics without applying stale moving geometry blindly."""
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        with self.pending_lock:
            pending = dict(self.pending)

        applied = 0
        stale = 0
        max_age = 0.0
        for cid, source_id in self.camera_index.items():
            row = pending.get(cid)
            if row is None:
                continue
            seq, captured, boxes = row
            if seq <= self.injected_seq.get(cid, 0):
                continue
            age_ms = max(0.0, (now - captured) * 1000.0)
            if age_ms > self.max_result_age_ms:
                self.injected_seq[cid] = seq
                stale += 1
                continue

            latest = self._latest_raw_tracks.get(int(source_id))
            has_recent_active = bool(
                latest is not None
                and latest[1]
                and (now - latest[0]) * 1000.0 <= self.v81_raw_track_max_age_ms
            )

            # A detector false-negative must not tell NvDCF that an active target was
            # explicitly absent. Treat an empty, delayed async result as a no-inference
            # tracker frame. This preserves visual tracking until the next real correction.
            if (
                not boxes
                and has_recent_active
                and age_ms >= self.v81_empty_detector_skip_age_ms
            ):
                self.injected_seq[cid] = seq
                self.v81_empty_detector_skips += 1
                continue

            inject_boxes = list(boxes)
            if age_ms >= self.v81_currentize_after_ms and boxes:
                inject_boxes, matched, unmatched = self._match_detector_to_raw_tracks(
                    int(source_id), boxes, now
                )
                self.v81_detector_currentized += matched
                if unmatched:
                    if age_ms <= self.v81_new_target_max_age_ms:
                        self.v81_detector_raw_new += unmatched
                    else:
                        # At this age an unmatched rectangle is unsafe as a new-target
                        # geometry. Keep only boxes that were currentized to active tracks.
                        fresh = []
                        latest_tracks = latest[1] if latest is not None else []
                        for box in inject_boxes:
                            is_current = False
                            for track in latest_tracks:
                                tbox = tuple(float(v) for v in track[1:5])
                                if self._iou(tuple(float(v) for v in box[:4]), tbox) >= 0.90:
                                    is_current = True
                                    break
                            if is_current:
                                fresh.append(box)
                        self.v81_detector_stale_dropped += max(0, len(inject_boxes) - len(fresh))
                        inject_boxes = fresh

            if not inject_boxes and has_recent_active:
                # Same protection as an empty detector result: do not set bInferDone
                # with zero boxes on a frame that already has a recent visual target.
                self.injected_seq[cid] = seq
                self.v81_empty_detector_skips += 1
                continue

            result = self.bridge.apply_detector_result(buffer, source_id, inject_boxes)
            if result == -2:
                continue
            if result < 0:
                continue
            self.injected_seq[cid] = seq
            applied += 1
            max_age = max(max_age, age_ms)

        if applied or stale:
            with self.det_lock:
                self.detector_frames_applied += applied
                self.stale_results += stale
                self.det_result_age_ms = max_age
        return self.Gst.PadProbeReturn.OK

    def _tracker_probe(self, _pad, info):
        """Publish only real NvDCF rows; never fabricate six empty source updates."""
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            rows = self.bridge.copy_tracks(buffer, max_rows=256)
            now = time.monotonic()
            grouped = {}
            raw_grouped = {}
            sx = self.display_width / float(self.track_width)
            sy = self.display_height / float(self.track_height)

            for row in rows:
                source_id = int(row["source_id"])
                if source_id not in self.index_camera:
                    continue
                conf = float(row["tracker_confidence"])
                if conf < 0.0:
                    conf = float(row["confidence"])
                self.v81_track_conf_samples.append(conf)
                if conf < self.min_display_track_conf:
                    continue

                raw_left = float(row["left"])
                raw_top = float(row["top"])
                raw_right = raw_left + float(row["width"])
                raw_bottom = raw_top + float(row["height"])
                object_id = int(row["object_id"])
                raw_grouped.setdefault(source_id, []).append(
                    (object_id, raw_left, raw_top, raw_right, raw_bottom, conf)
                )

                raw_box = (
                    raw_left * sx,
                    raw_top * sy,
                    raw_right * sx,
                    raw_bottom * sy,
                )
                self._record_jump(source_id, object_id, raw_box)
                display_box = self._stable_display_box(source_id, object_id, raw_box, now)
                grouped.setdefault(source_id, []).append(
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
                kept, count = self._dedup_v8(tracks)
                if kept:
                    published[source_id] = kept
                suppressed += count

            # Important: an omitted source in a partial nvstreammux batch is NO UPDATE,
            # not an empty tracker frame. V8 used grouped={all six: []}, which created
            # hundreds of false hold events and visibly froze old rectangles.
            with self.track_cache_lock:
                for source_id, tracks in published.items():
                    self.track_cache[source_id] = (now, tracks)
                    self.v81_real_track_updates += 1
                for source_id, row in list(self.track_cache.items()):
                    updated, tracks = row
                    if (now - updated) * 1000.0 > self.empty_hold_ms:
                        self.track_cache.pop(source_id, None)
                        self.v81_cache_prunes += 1
                self.tracked_now = sum(
                    len(tracks) for _updated, tracks in self.track_cache.values()
                )
                self.tracker_batches += 1
                self.v8_duplicates_suppressed += suppressed

            for source_id, tracks in raw_grouped.items():
                if tracks:
                    self._latest_raw_tracks[source_id] = (now, tracks)
            for source_id, row in list(self._latest_raw_tracks.items()):
                if (now - row[0]) * 1000.0 > self.v81_raw_track_max_age_ms:
                    self._latest_raw_tracks.pop(source_id, None)

            if published:
                self.v81_batches_with_tracks += 1
        except Exception as exc:
            print(
                f"CAMERA_V81_TRACK warning={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        return self.Gst.PadProbeReturn.OK

    def _display_overlay_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None or not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.track_cache_lock:
            cache = dict(self.track_cache)
        for source_id in self.index_camera:
            row = cache.get(source_id)
            if row is None:
                continue
            updated, tracks = row
            age_ms = max(0.0, (now - updated) * 1000.0)
            if age_ms > self.display_track_max_age_ms:
                continue
            self.v81_overlay_age_samples.append(age_ms)
            self.v81_overlay_draws += 1
            if age_ms > (1000.0 / max(1.0, self.track_fps)) * 1.35:
                self.v81_overlay_held_draws += 1
            self.bridge.add_tracked_boxes(buffer, source_id, tracks)
        self.bridge.apply_local_track_style(buffer)
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        confs = list(self.v81_track_conf_samples)
        ages = list(self.v81_overlay_age_samples)
        conf_p10 = self._percentile(confs, 0.10)
        conf_p50 = self._percentile(confs, 0.50)
        age_p50 = self._percentile(ages, 0.50)
        age_p95 = self._percentile(ages, 0.95)
        held_ratio = (
            self.v81_overlay_held_draws / float(max(1, self.v81_overlay_draws))
        )
        print(
            "CAMERA_V81_STATS "
            f"real_updates={self.v81_real_track_updates} cache_prunes={self.v81_cache_prunes} "
            f"currentized={self.v81_detector_currentized} raw_new={self.v81_detector_raw_new} "
            f"stale_new_dropped={self.v81_detector_stale_dropped} "
            f"empty_detector_skips={self.v81_empty_detector_skips} "
            f"track_conf_p10={conf_p10:.3f} track_conf_p50={conf_p50:.3f} "
            f"overlay_age_p50={age_p50:.0f}ms overlay_age_p95={age_p95:.0f}ms "
            f"held_draw_ratio={held_ratio:.3f} "
            "false_empty_fabrication=0 stale_geometry_currentization=1 predictor=0",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalStickySyncRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
