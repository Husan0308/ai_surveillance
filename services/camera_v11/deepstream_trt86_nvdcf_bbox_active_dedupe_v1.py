from __future__ import annotations

import os
import time

from services.camera_v11.active_dedupe_bridge import ActiveDedupeBridge
from services.camera_v11.deepstream_trt86_multi_v1 import map_detector_boxes_to_display
from services.camera_v11.deepstream_trt86_nvdcf_bbox_cam01_v1 import (
    V11DeepStreamTRT86NvDCFBBoxCam01V1,
    _iou_xyxy,
)
from services.camera_v11.deepstream_trt86_nvdcf_bbox_shadow_display_v1 import (
    V11DeepStreamTRT86NvDCFShadowDisplayV1,
)


class V11DeepStreamTRT86NvDCFActiveDedupeV1(V11DeepStreamTRT86NvDCFShadowDisplayV1):
    """Suppress duplicate ACTIVE OSD boxes without collapsing two real people.

    CAM-01 can occasionally keep two ACTIVE NvDCF targets on the same physical
    person. This layer is deliberately downstream of nvtracker. It uses the most
    recent detector boxes as one-to-one evidence: if two strongly-overlapping
    ACTIVE tracks are supported by two distinct detector persons, both are kept.
    Otherwise the weaker duplicate is hidden from downstream/OSD metadata only.
    """

    def __init__(self) -> None:
        raw = os.environ.get("V11_BBOX_ACTIVE_DEDUPE_CAMERAS", "CAM-01")
        self.active_dedupe_cameras = {
            part.strip() for part in raw.split(",") if part.strip()
        }
        self.active_dedupe_pair_iou = max(
            0.35,
            min(0.95, float(os.environ.get("V11_BBOX_ACTIVE_DEDUPE_PAIR_IOU", "0.62"))),
        )
        self.active_dedupe_pair_ios = max(
            0.55,
            min(0.99, float(os.environ.get("V11_BBOX_ACTIVE_DEDUPE_PAIR_IOS", "0.84"))),
        )
        self.active_dedupe_center_iou = max(
            0.10,
            min(0.80, float(os.environ.get("V11_BBOX_ACTIVE_DEDUPE_CENTER_IOU", "0.28"))),
        )
        self.active_dedupe_detector_iou = max(
            0.10,
            min(0.80, float(os.environ.get("V11_BBOX_ACTIVE_DEDUPE_DETECTOR_IOU", "0.22"))),
        )
        self.active_dedupe_detector_age_sec = max(
            0.20,
            min(1.20, float(os.environ.get("V11_BBOX_ACTIVE_DEDUPE_DETECTOR_AGE_SEC", "0.70"))),
        )
        self.active_dedupe_hold_frames = max(
            1,
            min(30, int(os.environ.get("V11_BBOX_ACTIVE_DEDUPE_HOLD_FRAMES", "12"))),
        )
        self._active_dedupe_first_seen: dict[str, dict[int, int]] = {}
        self._active_dedupe_alias: dict[str, dict[int, dict]] = {}
        self._active_dedupe_last_hidden: dict[str, tuple[int, ...]] = {}
        self._active_dedupe_hidden_total: dict[str, int] = {}
        self._active_dedupe_errors: dict[str, int] = {}
        self.active_dedupe_bridge: ActiveDedupeBridge | None = None

        super().__init__()

        if self.active_dedupe_cameras & set(self.bbox_track_cameras):
            self.active_dedupe_bridge = ActiveDedupeBridge()

        for cid in self.bbox_track_cameras:
            self._active_dedupe_first_seen[cid] = {}
            self._active_dedupe_alias[cid] = {}
            self._active_dedupe_last_hidden[cid] = ()
            self._active_dedupe_hidden_total[cid] = 0
            self._active_dedupe_errors[cid] = 0

        print(
            "CAMERA_V11_BBOX_ACTIVE_DEDUPE_ARCH "
            f"cameras={','.join(sorted(self.active_dedupe_cameras)) or 'off'} "
            f"pair_iou={self.active_dedupe_pair_iou:.2f} "
            f"pair_ios={self.active_dedupe_pair_ios:.2f} "
            f"center_iou={self.active_dedupe_center_iou:.2f} "
            f"detector_iou={self.active_dedupe_detector_iou:.2f} "
            f"detector_age_sec={self.active_dedupe_detector_age_sec:.2f} "
            f"hold_frames={self.active_dedupe_hold_frames} "
            "mode=detector-guided-active-active-display-dedupe tracker_feedback=0",
            flush=True,
        )

    @staticmethod
    def _box_center_distance_norm(a, b) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
        acx = (ax1 + ax2) * 0.5
        acy = (ay1 + ay2) * 0.5
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5
        scale = max(
            1.0,
            min(
                max(ax2 - ax1, ay2 - ay1),
                max(bx2 - bx1, by2 - by1),
            ),
        )
        dx = acx - bcx
        dy = acy - bcy
        return ((dx * dx + dy * dy) ** 0.5) / scale

    def _active_duplicate_geometry(self, a, b) -> bool:
        iou = _iou_xyxy(a, b)
        if iou >= self.active_dedupe_pair_iou:
            return True
        ios = self._intersection_over_smaller(a, b)
        if ios >= self.active_dedupe_pair_ios and self._box_center_distance_norm(a, b) <= 0.42:
            return True
        if iou >= self.active_dedupe_center_iou and (
            self._center_inside(a, b) or self._center_inside(b, a)
        ):
            return self._box_center_distance_norm(a, b) <= 0.38
        return False

    def _two_distinct_detector_support(self, a, b, detector_boxes) -> bool:
        if len(detector_boxes) < 2:
            return False
        a_scores = [_iou_xyxy(a, det) for det in detector_boxes]
        b_scores = [_iou_xyxy(b, det) for det in detector_boxes]
        threshold = self.active_dedupe_detector_iou
        for ai, a_score in enumerate(a_scores):
            if a_score < threshold:
                continue
            for bi, b_score in enumerate(b_scores):
                if ai == bi or b_score < threshold:
                    continue
                return True
        return False

    @staticmethod
    def _best_detector_iou(box, detector_boxes) -> float:
        if not detector_boxes:
            return 0.0
        return max(_iou_xyxy(box, det) for det in detector_boxes)

    def _track_quality(
        self,
        row: dict,
        box,
        detector_boxes,
        tracker_frame: int,
        first_seen: dict[int, int],
    ) -> float:
        detector_score = self._best_detector_iou(box, detector_boxes)
        tracker_conf = float(row.get("tracker_confidence", -1.0))
        tracker_conf = max(0.0, min(1.0, tracker_conf if tracker_conf >= 0.0 else 0.0))
        object_id = int(row["object_id"])
        age = max(0, tracker_frame - int(first_seen.get(object_id, tracker_frame)))
        age_score = min(1.0, age / 80.0)
        return detector_score * 1.65 + tracker_conf * 0.70 + age_score * 0.16

    def _fresh_detector_boxes(self, cid: str):
        now = time.monotonic()
        with self.lock:
            state = self.states[cid]
            snapshot = state.latest_snapshot
            age = now - snapshot.completed_mono if snapshot.completed_mono > 0 else 999.0
            boxes = snapshot.boxes if age <= self.active_dedupe_detector_age_sec else ()
        mapped = (
            map_detector_boxes_to_display(boxes, self.width, self.height)
            if boxes
            else []
        )
        return mapped, age

    def _apply_active_dedupe(
        self,
        cid: str,
        buffer,
        tracks: list[dict],
        tracker_frame: int,
    ) -> tuple[list[dict], tuple[int, ...], float, int]:
        if (
            cid not in self.active_dedupe_cameras
            or self.active_dedupe_bridge is None
            or len(tracks) < 2
        ):
            return tracks, (), 999.0, 0

        first_seen = self._active_dedupe_first_seen.setdefault(cid, {})
        aliases = self._active_dedupe_alias.setdefault(cid, {})
        by_id = {int(row["object_id"]): row for row in tracks}
        for object_id in by_id:
            first_seen.setdefault(object_id, tracker_frame)

        detector_boxes, detector_age = self._fresh_detector_boxes(cid)
        boxes = {object_id: self._row_xyxy(row) for object_id, row in by_id.items()}

        # Remove stale aliases immediately if the pair no longer exists, diverges,
        # expires, or a fresh detector now proves that two distinct people exist.
        for loser, entry in list(aliases.items()):
            keeper = int(entry["keeper"])
            if (
                loser not in by_id
                or keeper not in by_id
                or tracker_frame > int(entry["expires"])
                or not self._active_duplicate_geometry(boxes[loser], boxes[keeper])
            ):
                aliases.pop(loser, None)
                continue
            if detector_boxes and self._two_distinct_detector_support(
                boxes[loser], boxes[keeper], detector_boxes
            ):
                aliases.pop(loser, None)

        # Only create a new duplicate decision when a recent detector snapshot is
        # available. This protects two real nested people during detector gaps.
        if detector_boxes:
            ids = list(by_id)
            pairs = []
            for index, left_id in enumerate(ids):
                for right_id in ids[index + 1 :]:
                    left_box = boxes[left_id]
                    right_box = boxes[right_id]
                    if not self._active_duplicate_geometry(left_box, right_box):
                        continue
                    if self._two_distinct_detector_support(
                        left_box, right_box, detector_boxes
                    ):
                        continue
                    overlap = max(
                        _iou_xyxy(left_box, right_box),
                        self._intersection_over_smaller(left_box, right_box),
                    )
                    pairs.append((overlap, left_id, right_id))
            pairs.sort(reverse=True)

            suppressed_now = set(aliases)
            for _overlap, left_id, right_id in pairs:
                if left_id in suppressed_now or right_id in suppressed_now:
                    continue
                left_quality = self._track_quality(
                    by_id[left_id],
                    boxes[left_id],
                    detector_boxes,
                    tracker_frame,
                    first_seen,
                )
                right_quality = self._track_quality(
                    by_id[right_id],
                    boxes[right_id],
                    detector_boxes,
                    tracker_frame,
                    first_seen,
                )
                if abs(left_quality - right_quality) < 0.03:
                    left_first = int(first_seen.get(left_id, tracker_frame))
                    right_first = int(first_seen.get(right_id, tracker_frame))
                    keeper = left_id if left_first <= right_first else right_id
                else:
                    keeper = left_id if left_quality > right_quality else right_id
                loser = right_id if keeper == left_id else left_id
                aliases[loser] = {
                    "keeper": keeper,
                    "expires": tracker_frame + self.active_dedupe_hold_frames,
                }
                suppressed_now.add(loser)

        hidden_ids = tuple(
            sorted(
                loser
                for loser, entry in aliases.items()
                if loser in by_id
                and int(entry["keeper"]) in by_id
                and self._active_duplicate_geometry(
                    boxes[loser], boxes[int(entry["keeper"])]
                )
            )
        )
        hidden = 0
        if hidden_ids:
            hidden = self.active_dedupe_bridge.hide_track_ids(buffer, 0, hidden_ids)
            if hidden < 0:
                raise RuntimeError(f"active dedupe hide returned {hidden}")
            tracks = self.meta_bridge.copy_tracks(buffer, max_rows=64)

        # Bound bookkeeping for long runtimes.
        live_ids = set(by_id)
        for object_id, frame_num in list(first_seen.items()):
            if object_id not in live_ids and tracker_frame - int(frame_num) > 200:
                first_seen.pop(object_id, None)
                aliases.pop(object_id, None)

        return tracks, hidden_ids, detector_age, hidden

    def _tracker_output_probe(self, _pad, info, cid: str):
        if cid not in self.active_dedupe_cameras:
            return super()._tracker_output_probe(_pad, info, cid)

        # Call the accepted NvDCF base probe exactly once. Active duplicate
        # filtering happens after tracker state is final for this frame and before
        # any live-shadow rectangle is added.
        result = V11DeepStreamTRT86NvDCFBBoxCam01V1._tracker_output_probe(
            self, _pad, info, cid
        )
        buffer = info.get_buffer()
        if buffer is None:
            return result

        tracks = self.meta_bridge.copy_tracks(buffer, max_rows=64)
        tracker_frame = int(self._nvdcf_tracker_frames.get(cid, 0))
        hidden_ids: tuple[int, ...] = ()
        detector_age = 999.0
        hidden = 0
        try:
            tracks, hidden_ids, detector_age, hidden = self._apply_active_dedupe(
                cid, buffer, tracks, tracker_frame
            )
        except Exception as exc:
            errors = self._active_dedupe_errors.get(cid, 0) + 1
            self._active_dedupe_errors[cid] = errors
            if errors <= 5 or errors % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_ACTIVE_DEDUPE_WARNING "
                    f"camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )

        previous_hidden = self._active_dedupe_last_hidden.get(cid, ())
        self._active_dedupe_last_hidden[cid] = hidden_ids
        if hidden > 0:
            self._active_dedupe_hidden_total[cid] = (
                self._active_dedupe_hidden_total.get(cid, 0) + hidden
            )
        if (
            hidden_ids != previous_hidden
            or (hidden_ids and tracker_frame % 20 == 0)
        ):
            alias_text = ",".join(
                f"{loser}->{int(entry['keeper'])}"
                for loser, entry in sorted(self._active_dedupe_alias.get(cid, {}).items())
                if loser in hidden_ids
            ) or "none"
            print(
                "CAMERA_V11_BBOX_ACTIVE_DEDUPE "
                f"camera={cid} tracker_frame={tracker_frame} "
                f"visible_after={len(tracks)} "
                f"hidden_ids={','.join(map(str, hidden_ids)) or 'none'} "
                f"aliases={alias_text} hidden={hidden} "
                f"detector_age_ms={detector_age * 1000.0:.1f} "
                f"hidden_total={self._active_dedupe_hidden_total.get(cid, 0)}",
                flush=True,
            )

        # From this point onward use the post-dedupe ACTIVE truth for shadow
        # diagnostics and shadow-vs-active suppression.
        active_ids = {int(row["object_id"]) for row in tracks}
        active_boxes = [self._row_xyxy(row) for row in tracks]

        last_active = self._shadow_last_active_frame.setdefault(cid, {})
        for object_id in active_ids:
            last_active[object_id] = tracker_frame

        self._log_shadow_diag(cid, buffer, tracks, tracker_frame)

        if self.shadow_diag is None:
            return result

        try:
            rows = self.shadow_diag.copy_shadow_tracks(buffer, max_rows=64)
            latest: dict[int, dict] = {}
            for row in rows:
                object_id = int(row["object_id"])
                previous = latest.get(object_id)
                if previous is None or int(row["frame_num"]) >= int(previous["frame_num"]):
                    latest[object_id] = row

            selected: list[tuple[int, dict, int]] = []
            suppressed: list[tuple[int, str]] = []
            for object_id, row in latest.items():
                if object_id in active_ids:
                    continue
                if str(row.get("state", "")).upper() != "INACTIVE":
                    continue
                if not self._valid_shadow_row(row):
                    continue
                if float(row.get("confidence", -1.0)) < self.shadow_display_min_conf:
                    continue

                last_seen = last_active.get(object_id)
                if last_seen is None:
                    continue
                inactive_frames = tracker_frame - int(last_seen)
                if inactive_frames <= 0 or inactive_frames > self.shadow_display_frames:
                    continue

                shadow_box = self._shadow_xyxy(row)
                conflict = next(
                    (
                        active_box
                        for active_box in active_boxes
                        if self._shadow_conflicts_with_active(shadow_box, active_box)
                    ),
                    None,
                )
                if conflict is not None:
                    suppressed.append((object_id, "active-overlap"))
                    continue
                selected.append((object_id, row, inactive_frames))

            if suppressed:
                self._shadow_suppressed_total[cid] = (
                    self._shadow_suppressed_total.get(cid, 0) + len(suppressed)
                )

            shadow_boxes = []
            for object_id, row, _inactive_frames in selected:
                x1, y1, x2, y2 = self._shadow_xyxy(row)
                shadow_boxes.append(
                    (
                        object_id,
                        x1,
                        y1,
                        x2,
                        y2,
                        max(0.01, float(row.get("confidence", 0.01))),
                    )
                )

            added = 0
            if shadow_boxes:
                added = self.meta_bridge.add_tracked_boxes(buffer, 0, shadow_boxes)
                if added < 0:
                    raise RuntimeError(f"shadow display add returned {added}")
                self._shadow_display_total[cid] = (
                    self._shadow_display_total.get(cid, 0) + int(added)
                )

            display_ids = tuple(sorted(object_id for object_id, _row, _gap in selected))
            suppressed_ids = tuple(sorted(object_id for object_id, _reason in suppressed))
            previous_ids = self._shadow_display_last_ids.get(cid, ())
            previous_suppressed = self._shadow_suppressed_last_ids.get(cid, ())
            self._shadow_display_last_ids[cid] = display_ids
            self._shadow_suppressed_last_ids[cid] = suppressed_ids
            if (
                display_ids != previous_ids
                or suppressed_ids != previous_suppressed
                or (display_ids and tracker_frame % 20 == 0)
            ):
                details = []
                for object_id, row, inactive_frames in selected:
                    details.append(
                        f"{object_id}:gap{inactive_frames}:conf{float(row['confidence']):.3f}:"
                        f"bbox={float(row['left']):.1f},{float(row['top']):.1f},"
                        f"{float(row['width']):.1f},{float(row['height']):.1f}"
                    )
                print(
                    "CAMERA_V11_BBOX_SHADOW_DISPLAY "
                    f"camera={cid} tracker_frame={tracker_frame} "
                    f"active_ids={','.join(map(str, sorted(active_ids))) or 'none'} "
                    f"draw_ids={','.join(map(str, display_ids)) or 'none'} "
                    f"suppressed_ids={','.join(map(str, suppressed_ids)) or 'none'} "
                    f"added={added} details={'|'.join(details) or 'none'} "
                    f"draw_total={self._shadow_display_total.get(cid, 0)} "
                    f"suppressed_total={self._shadow_suppressed_total.get(cid, 0)}",
                    flush=True,
                )

            cleanup_age = max(100, self.shadow_display_frames * 4)
            live_shadow_ids = set(latest)
            for object_id, frame_num in list(last_active.items()):
                if (
                    object_id not in active_ids
                    and object_id not in live_shadow_ids
                    and tracker_frame - int(frame_num) > cleanup_age
                ):
                    last_active.pop(object_id, None)

        except Exception as exc:
            errors = self._shadow_display_errors.get(cid, 0) + 1
            self._shadow_display_errors[cid] = errors
            if errors <= 5 or errors % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_SHADOW_DISPLAY_WARNING "
                    f"camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )
        return result


def main() -> int:
    return V11DeepStreamTRT86NvDCFActiveDedupeV1().run()


if __name__ == "__main__":
    raise SystemExit(main())
