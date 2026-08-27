from __future__ import annotations

import math
import os
import sys
import time
from collections import deque

from .runtime_v91_inprocess_trt86 import PascalInProcessTrt86Runtime


class PascalCurrentFrameBboxRuntime(PascalInProcessTrt86Runtime):
    """V9.2: keep V9.1 GPU architecture, fix moving-box temporal semantics.

    V9.1 proved the detector should remain in-process on the CUDA primary context.
    The remaining visible lag/freeze is on the bbox path:

    * V8/V9.1 fabricated an explicit empty update for every source missing from a
      partial nvstreammux tracker batch.  The display then held the previous box,
      which looks like a rectangle freezing behind a walking person.
    * async detector observations are often 80-220 ms old by the time they are
      injected into a current tracker frame.  Applying their old geometry verbatim
      can pull an otherwise-current NvDCF target backwards.
    * the 0.28 native/display confidence floor is too aggressive for short DCF
      confidence dips during motion; valid current rows can disappear before Python
      sees them.

    V9.2 therefore changes bbox temporal semantics only.  It does not add a long
    velocity predictor and does not alter the V9.1 detector process/context design.
    """

    def __init__(self) -> None:
        self.v92_currentize_after_ms = max(
            40.0,
            min(140.0, float(os.environ.get("CAMERA_V92_CURRENTIZE_AFTER_MS", "60"))),
        )
        self.v92_new_target_max_age_ms = max(
            120.0,
            min(320.0, float(os.environ.get("CAMERA_V92_NEW_TARGET_MAX_AGE_MS", "220"))),
        )
        self.v92_raw_track_max_age_ms = max(
            100.0,
            min(260.0, float(os.environ.get("CAMERA_V92_RAW_TRACK_MAX_AGE_MS", "170"))),
        )
        self.v92_empty_detector_skip_age_ms = max(
            40.0,
            min(180.0, float(os.environ.get("CAMERA_V92_EMPTY_DETECTOR_SKIP_AGE_MS", "70"))),
        )
        self.v92_detector_currentized = 0
        self.v92_detector_raw_new = 0
        self.v92_detector_stale_dropped = 0
        self.v92_empty_detector_skips = 0
        self.v92_cache_prunes = 0
        self.v92_real_track_updates = 0
        self.v92_batches_with_tracks = 0
        self.v92_track_conf_samples: deque[float] = deque(maxlen=4096)
        self.v92_overlay_age_samples: deque[float] = deque(maxlen=4096)
        self.v92_overlay_draws = 0
        self._latest_raw_tracks: dict[
            int,
            tuple[float, list[tuple[int, float, float, float, float, float]]],
        ] = {}
        super().__init__()
        print(
            "CAMERA_V92_ARCH "
            f"detector=v91-inprocess-primary tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"display_conf={self.min_display_track_conf:.2f} cache_hold={self.empty_hold_ms:.0f}ms "
            f"display_max_age={self.display_track_max_age_ms:.0f}ms "
            f"currentize_after={self.v92_currentize_after_ms:.0f}ms "
            "partial_batch=present-rows-only stale_detector=currentized predictor=0",
            flush=True,
        )

    def _prepare_tracker_files(self):
        lib, generated = super()._prepare_tracker_files()
        lines = generated.read_text(encoding="utf-8").splitlines()

        # Keep a moving visual target active through short confidence dips.  Shadow
        # output remains disabled; only regular NvDCF object metadata is displayed.
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
            "CAMERA_V92_NVDCF "
            "min_tracker_conf=0.12 probation=1 early_termination=1 "
            f"shadow_frames={shadow_frames} output_shadow=0 "
            f"feature_level={getattr(self, 'v85_feature_level', 1)}",
            flush=True,
        )
        return lib, generated

    @staticmethod
    def _percentile_v92(values, p: float) -> float:
        rows = sorted(float(v) for v in values)
        if not rows:
            return 0.0
        idx = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * float(p)))))
        return rows[idx]

    @staticmethod
    def _center_distance_v92(a, b) -> float:
        acx = 0.5 * (a[0] + a[2])
        acy = 0.5 * (a[1] + a[3])
        bcx = 0.5 * (b[0] + b[2])
        bcy = 0.5 * (b[1] + b[3])
        return math.hypot(acx - bcx, acy - bcy)

    @staticmethod
    def _diag_v92(box) -> float:
        return math.hypot(max(2.0, box[2] - box[0]), max(2.0, box[3] - box[1]))

    def _match_detector_to_raw_tracks_v92(self, source_id: int, boxes, now: float):
        latest = self._latest_raw_tracks.get(int(source_id))
        if latest is None:
            return list(boxes), 0, 0
        updated, raw_tracks = latest
        raw_age_ms = max(0.0, (now - updated) * 1000.0)
        if raw_age_ms > self.v92_raw_track_max_age_ms or not raw_tracks:
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
                distance = self._center_distance_v92(dbox, tbox)
                scale = max(self._diag_v92(dbox), self._diag_v92(tbox), 4.0)
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
        """Do not yank current NvDCF geometry backwards with an old async box."""
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
                and (now - latest[0]) * 1000.0 <= self.v92_raw_track_max_age_ms
            )

            # A delayed false-negative is not evidence that a currently tracked
            # person vanished.  Skip the old empty detector update and let NvDCF
            # continue on the current frame.
            if (
                not boxes
                and has_recent_active
                and age_ms >= self.v92_empty_detector_skip_age_ms
            ):
                self.injected_seq[cid] = seq
                self.v92_empty_detector_skips += 1
                continue

            inject_boxes = list(boxes)
            if age_ms >= self.v92_currentize_after_ms and boxes:
                inject_boxes, matched, unmatched = self._match_detector_to_raw_tracks_v92(
                    int(source_id), boxes, now
                )
                self.v92_detector_currentized += matched
                if unmatched:
                    if age_ms <= self.v92_new_target_max_age_ms:
                        self.v92_detector_raw_new += unmatched
                    else:
                        # Very old unmatched boxes are unsafe new-target geometry.
                        fresh = []
                        latest_tracks = latest[1] if latest is not None else []
                        for box in inject_boxes:
                            keep = any(
                                self._iou(
                                    tuple(float(v) for v in box[:4]),
                                    tuple(float(v) for v in track[1:5]),
                                )
                                >= 0.90
                                for track in latest_tracks
                            )
                            if keep:
                                fresh.append(box)
                        self.v92_detector_stale_dropped += max(
                            0, len(inject_boxes) - len(fresh)
                        )
                        inject_boxes = fresh

            if not inject_boxes and has_recent_active:
                self.injected_seq[cid] = seq
                self.v92_empty_detector_skips += 1
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
        """Update only sources that actually produced NvDCF rows in this batch."""
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
            filtered = 0

            for row in rows:
                source_id = int(row["source_id"])
                if source_id not in self.index_camera:
                    continue
                conf = float(row["tracker_confidence"])
                if conf < 0.0:
                    conf = float(row["confidence"])
                self.v92_track_conf_samples.append(conf)
                if conf < self.min_display_track_conf:
                    filtered += 1
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

            # Critical fix: nvstreammux can push a batch when its timeout expires
            # before all sources contribute.  Missing source != explicit empty track.
            # V9.1 initialized all six sources to [], repeatedly refreshing a held old
            # rectangle.  V9.2 updates only real rows and expires them by wall age.
            real_updates = 0
            pruned = 0
            with self.track_cache_lock:
                for source_id, tracks in published.items():
                    self.track_cache[source_id] = (now, tracks)
                    real_updates += 1

                for source_id, row in list(self.track_cache.items()):
                    updated, _tracks = row
                    if (now - updated) * 1000.0 > self.empty_hold_ms:
                        self.track_cache.pop(source_id, None)
                        pruned += 1

                self.tracked_now = sum(
                    len(tracks) for _updated, tracks in self.track_cache.values()
                )
                self.tracker_batches += 1
                self.v8_low_conf_filtered += filtered
                self.v8_duplicates_suppressed += suppressed
                self.v8_real_updates += real_updates
                self.v8_empty_expires += pruned
                self.v92_real_track_updates += real_updates
                self.v92_cache_prunes += pruned
                active_keys = {
                    (source_id, int(track[0]))
                    for source_id, (_updated, tracks) in self.track_cache.items()
                    for track in tracks
                }

            for source_id, tracks in raw_grouped.items():
                if tracks:
                    self._latest_raw_tracks[source_id] = (now, tracks)
            for source_id, row in list(self._latest_raw_tracks.items()):
                if (now - row[0]) * 1000.0 > self.v92_raw_track_max_age_ms:
                    self._latest_raw_tracks.pop(source_id, None)

            for key, state in list(self._display_sizes.items()):
                if key not in active_keys and now - state.seen_at > 0.7:
                    self._display_sizes.pop(key, None)
                    self._last_raw_boxes.pop(key, None)

            if published:
                self.v92_batches_with_tracks += 1
        except Exception as exc:
            print(
                f"CAMERA_V92_TRACK warning={type(exc).__name__}:{exc}",
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
            self.v92_overlay_age_samples.append(age_ms)
            self.v92_overlay_draws += len(tracks)
            self.bridge.add_tracked_boxes(buffer, source_id, tracks)
        self.bridge.apply_local_track_style(buffer)
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        conf = list(self.v92_track_conf_samples)
        ages = list(self.v92_overlay_age_samples)
        print(
            "CAMERA_V92_STATS "
            f"real_updates={self.v92_real_track_updates} cache_prunes={self.v92_cache_prunes} "
            f"batches_with_tracks={self.v92_batches_with_tracks} "
            f"currentized={self.v92_detector_currentized} raw_new={self.v92_detector_raw_new} "
            f"stale_dropped={self.v92_detector_stale_dropped} empty_detector_skips={self.v92_empty_detector_skips} "
            f"track_conf_p10={self._percentile_v92(conf, 0.10):.3f} "
            f"overlay_age_p50={self._percentile_v92(ages, 0.50):.0f}ms "
            f"overlay_age_p95={self._percentile_v92(ages, 0.95):.0f}ms "
            f"overlay_draws={self.v92_overlay_draws} "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms tracker_batches={self.tracker_batches} tracked_now={self.tracked_now}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalCurrentFrameBboxRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
