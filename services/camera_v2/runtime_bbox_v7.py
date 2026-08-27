from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

from .runtime import CleanCameraRuntime, DETECT_W
from .runtime_lane import SerializedGpuLaneRuntime


@dataclass
class _DisplaySizeState:
    width: float
    height: float
    width_expanded_at: float
    height_expanded_at: float
    seen_at: float


def _stable_size(
    previous: float,
    current: float,
    last_expand_at: float,
    now: float,
    *,
    hold_sec: float,
    shrink_alpha: float,
) -> tuple[float, float]:
    """Open immediately around a larger NvDCF box; close slowly after a short hold."""
    previous = max(1.0, float(previous))
    current = max(1.0, float(current))
    if current >= previous:
        return current, float(now)
    if float(now) - float(last_expand_at) <= float(hold_sec):
        return previous, float(last_expand_at)
    value = previous + float(shrink_alpha) * (current - previous)
    return max(current, value), float(last_expand_at)


def _expand_box(
    box: tuple[float, float, float, float],
    frame_width: float,
    frame_height: float,
    *,
    side_margin: float,
    top_margin: float,
    bottom_margin: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    return (
        max(0.0, x1 - width * side_margin),
        max(0.0, y1 - height * top_margin),
        min(float(frame_width - 1), x2 + width * side_margin),
        min(float(frame_height - 1), y2 + height * bottom_margin),
    )


class NvDCFStickyBBoxRuntime(SerializedGpuLaneRuntime):
    """Step 4 V7: sparse YOLO correction + real NvDCF localization on video frames.

    V6 attempted to bridge 2 Hz detector gaps with a CPU velocity predictor. That can
    smooth a rectangle but it cannot see the person between detector observations.
    V7 restores the proven architecture: detector metadata is only a correction/input;
    NvDCF consumes the actual camera frames and owns temporal localization.

    The display policy is deliberately weak and cannot feed back into association:
      * use only current, sufficiently-confident NvDCF objects;
      * never render shadow targets;
      * move bbox center directly with the current NvDCF result (no velocity lead);
      * open width/height immediately, but hold/close them briefly to avoid limb clipping;
      * add the historical full-body 6/4/7 percent display-only safety margin.
    """

    def __init__(self) -> None:
        self.min_display_track_conf = float(
            os.environ.get("CAMERA_V2_MIN_DISPLAY_TRACK_CONF", "0.28")
        )
        self.display_side_margin = float(
            os.environ.get("CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN", "0.06")
        )
        self.display_top_margin = float(
            os.environ.get("CAMERA_V2_DISPLAY_BOX_TOP_MARGIN", "0.04")
        )
        self.display_bottom_margin = float(
            os.environ.get("CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN", "0.07")
        )
        self.display_size_hold_sec = float(
            os.environ.get("CAMERA_V2_DISPLAY_SIZE_HOLD_SEC", "0.22")
        )
        self.display_shrink_alpha = float(
            os.environ.get("CAMERA_V2_DISPLAY_SHRINK_ALPHA", "0.42")
        )
        self.jump_diag_limit = float(
            os.environ.get("CAMERA_V2_TRACK_JUMP_DIAG_LIMIT", "1.00")
        )
        self._display_sizes: dict[tuple[int, int], _DisplaySizeState] = {}
        self._last_raw_boxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        self.v7_low_conf_filtered = 0
        self.v7_duplicates_suppressed = 0
        self.v7_teleport_events = 0
        self.v7_empty_cache_clears = 0
        super().__init__()
        print(
            "CAMERA_BBOX_V7_PROFILE "
            f"tracker=NvDCF/{self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"detector={self.detect_hz:.2f}Hz/cam current_nvdcf_only=1 shadow_render=0 "
            f"min_display_conf={self.min_display_track_conf:.2f} "
            f"margin={self.display_side_margin:.2f}/{self.display_top_margin:.2f}/{self.display_bottom_margin:.2f} "
            f"size_hold={self.display_size_hold_sec:.2f}s shrink_alpha={self.display_shrink_alpha:.2f} "
            "custom_velocity_predictor=0",
            flush=True,
        )

    def _prepare_tracker_files(self):
        # Match the old known-good Pascal geometry, but run the visual tracker at the
        # camera cadence. NVIDIA recommends spending budget on NvDCF visual tracking
        # rather than trying to synthesize skipped-frame boxes in application code.
        self.track_width = max(
            320,
            min(DETECT_W, int(os.environ.get("CAMERA_V2_TRACK_WIDTH", "512"))),
        )
        self.track_height = max(
            192,
            min(384, int(os.environ.get("CAMERA_V2_TRACK_HEIGHT", "288"))),
        )

        lib, generated = CleanCameraRuntime._prepare_tracker_files(self)
        lines = generated.read_text(encoding="utf-8").splitlines()
        detector_floor = os.environ.get("CAMERA_V2_DETECT_CONF", "0.08")
        shadow_frames = max(12, int(round(self.track_fps * 0.90)))

        # Restore the conservative profile that previously behaved well on the office
        # cameras. Shadow memory is retained internally for a short reacquisition window
        # but never emitted to the wall, so it cannot create frozen/ghost rectangles.
        self._replace_yaml_key(lines, "minDetectorConfidence", detector_floor)
        self._replace_yaml_key(lines, "enableBboxUnClipping", "0")
        self._replace_yaml_key(lines, "minIouDiff4NewTarget", "0.72")
        self._replace_yaml_key(lines, "minTrackerConfidence", "0.28")
        self._replace_yaml_key(lines, "probationAge", "2")
        self._replace_yaml_key(lines, "maxShadowTrackingAge", str(shadow_frames))
        self._replace_yaml_key(lines, "earlyTerminationAge", "1")
        self._replace_yaml_key(lines, "minIou4TargetDuplicate", "0.94", required=False)
        self._replace_yaml_key(lines, "targetDuplicateRunInterval", "5", required=False)
        self._replace_yaml_key(lines, "useColorNames", "0", required=False)
        self._replace_yaml_key(lines, "useHog", "1", required=False)
        self._replace_yaml_key(lines, "featureImgSizeLevel", "3", required=False)
        self._replace_yaml_key(lines, "searchRegionPaddingScale", "1", required=False)
        self._replace_yaml_key(lines, "usePrediction4Assoc", "1", required=False)
        self._replace_yaml_key(
            lines, "minTrackingConfidenceDuringInactive", "0.40", required=False
        )
        self._insert_target_management_key(lines, "outputShadowTracks", "0")
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_BBOX_V7_NVDCF "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"min_detector_conf={detector_floor} min_tracker_conf=0.28 probation=2 "
            f"shadow_frames={shadow_frames} output_shadow=0 hog=1 feature_level=3 "
            "bbox_unclip=0 reid=0",
            flush=True,
        )
        return lib, generated

    @staticmethod
    def _iou(a, b) -> float:
        inter = CleanCameraRuntime._intersection(a, b)
        if inter <= 0.0:
            return 0.0
        union = CleanCameraRuntime._area(a) + CleanCameraRuntime._area(b) - inter
        return inter / union if union > 0.0 else 0.0

    def _dedup_v7(self, tracks):
        ordered = sorted(tracks, key=lambda row: float(row[5]), reverse=True)
        kept = []
        suppressed = 0
        for row in ordered:
            box = row[1:5]
            area = max(1.0, self._area(box))
            duplicate = False
            for other in kept:
                other_box = other[1:5]
                other_area = max(1.0, self._area(other_box))
                inter = self._intersection(box, other_box)
                containment = inter / max(1.0, min(area, other_area))
                if self._iou(box, other_box) >= 0.72 or containment >= 0.94:
                    duplicate = True
                    break
            if duplicate:
                suppressed += 1
            else:
                kept.append(row)
        return kept, suppressed

    def _stable_display_box(
        self,
        source_id: int,
        object_id: int,
        raw_box: tuple[float, float, float, float],
        now: float,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = raw_box
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        key = (int(source_id), int(object_id))
        previous = self._display_sizes.get(key)
        if previous is None:
            stable_w = width
            stable_h = height
            width_expanded_at = now
            height_expanded_at = now
        else:
            stable_w, width_expanded_at = _stable_size(
                previous.width,
                width,
                previous.width_expanded_at,
                now,
                hold_sec=self.display_size_hold_sec,
                shrink_alpha=self.display_shrink_alpha,
            )
            stable_h, height_expanded_at = _stable_size(
                previous.height,
                height,
                previous.height_expanded_at,
                now,
                hold_sec=self.display_size_hold_sec,
                shrink_alpha=self.display_shrink_alpha,
            )
        self._display_sizes[key] = _DisplaySizeState(
            stable_w,
            stable_h,
            width_expanded_at,
            height_expanded_at,
            now,
        )
        base = (
            cx - 0.5 * stable_w,
            cy - 0.5 * stable_h,
            cx + 0.5 * stable_w,
            cy + 0.5 * stable_h,
        )
        return _expand_box(
            base,
            self.display_width,
            self.display_height,
            side_margin=self.display_side_margin,
            top_margin=self.display_top_margin,
            bottom_margin=self.display_bottom_margin,
        )

    def _record_jump(
        self,
        source_id: int,
        object_id: int,
        box: tuple[float, float, float, float],
    ) -> None:
        key = (int(source_id), int(object_id))
        previous = self._last_raw_boxes.get(key)
        self._last_raw_boxes[key] = box
        if previous is None:
            return
        px1, py1, px2, py2 = previous
        x1, y1, x2, y2 = box
        pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        diag = math.hypot(max(2.0, px2 - px1), max(2.0, py2 - py1))
        if math.hypot(cx - pcx, cy - pcy) > self.jump_diag_limit * max(1.0, diag):
            self.v7_teleport_events += 1

    def _tracker_probe(self, _pad, info):
        # This replaces QualityCameraRuntime._tracker_probe but preserves the serialized
        # GPU-lane release contract from SerializedGpuLaneRuntime.
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
                        source_id, object_id, raw_box, now
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

                active_keys = {
                    (source_id, int(row[0]))
                    for source_id, tracks in published.items()
                    for row in tracks
                }
                for key, state in list(self._display_sizes.items()):
                    if key not in active_keys and now - state.seen_at > 1.0:
                        self._display_sizes.pop(key, None)
                        self._last_raw_boxes.pop(key, None)

                with self.track_cache_lock:
                    for source_id, tracks in published.items():
                        previous = self.track_cache.get(source_id)
                        if not tracks and previous is not None and previous[1]:
                            self.v7_empty_cache_clears += 1
                        # Publish every source on every NvDCF batch, including []: an
                        # empty current frame must clear an old box immediately instead
                        # of leaving it visible until a stale-cache timeout expires.
                        self.track_cache[source_id] = (now, tracks)
                    self.tracked_now = sum(len(tracks) for tracks in published.values())
                    self.tracker_batches += 1
                    self.v7_low_conf_filtered += filtered
                    self.v7_duplicates_suppressed += suppressed
            except Exception as exc:
                print(
                    f"CAMERA_BBOX_V7_TRACK warning={type(exc).__name__}:{exc}",
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
            "CAMERA_BBOX_V7_STATS "
            f"current_nvdcf_only=1 predictor=0 shadow_render=0 "
            f"low_conf_filtered={self.v7_low_conf_filtered} "
            f"duplicates_suppressed={self.v7_duplicates_suppressed} "
            f"teleport_events={self.v7_teleport_events} "
            f"empty_cache_clears={self.v7_empty_cache_clears}",
            flush=True,
        )
        return keep


def main() -> int:
    return NvDCFStickyBBoxRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
