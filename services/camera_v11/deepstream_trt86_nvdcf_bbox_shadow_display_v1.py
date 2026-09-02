from __future__ import annotations

import os

from services.camera_v11.deepstream_trt86_nvdcf_bbox_cam01_v1 import (
    V11DeepStreamTRT86NvDCFBBoxCam01V1,
    _iou_xyxy,
)
from services.camera_v11.deepstream_trt86_nvdcf_bbox_continuity_v1 import (
    V11DeepStreamTRT86NvDCFBBoxContinuityV1,
)


class V11DeepStreamTRT86NvDCFShadowDisplayV1(V11DeepStreamTRT86NvDCFBBoxContinuityV1):
    """Use NvDCF live INACTIVE/shadow boxes to bridge short OSD gaps.

    Cameras selected by ``V11_BBOX_SHADOW_DISPLAY_CAMERAS`` bypass the older
    frozen continuity hold and instead draw the current
    ``NVDS_TRACKER_SHADOW_LIST_META`` rectangle for a short window after a target
    leaves ACTIVE output.

    The shadow rectangle is injected only after nvtracker, on its src probe, so it
    cannot feed back into association, ReAssoc, detector matching, counts, or future
    tracker state. A recently-active requirement, a strict frame window, and
    active-vs-shadow duplicate suppression prevent long-lived/duplicate OSD boxes.
    """

    def __init__(self) -> None:
        raw = os.environ.get("V11_BBOX_SHADOW_DISPLAY_CAMERAS", "CAM-05")
        self.shadow_display_cameras = {
            part.strip() for part in raw.split(",") if part.strip()
        }
        self.shadow_display_frames = max(
            1,
            min(40, int(os.environ.get("V11_BBOX_SHADOW_DISPLAY_FRAMES", "10"))),
        )
        self.shadow_display_min_conf = max(
            0.0,
            min(1.0, float(os.environ.get("V11_BBOX_SHADOW_DISPLAY_MIN_CONF", "0.22"))),
        )
        self.shadow_display_suppress_iou = max(
            0.10,
            min(0.95, float(os.environ.get("V11_BBOX_SHADOW_DISPLAY_SUPPRESS_IOU", "0.50"))),
        )
        # IoU alone is weak when an old shadow box is smaller/larger or vertically
        # shifted relative to the newly ACTIVE box. Intersection-over-smaller-area
        # catches that same-person case without touching tracker state.
        self.shadow_display_suppress_ios = max(
            0.30,
            min(0.99, float(os.environ.get("V11_BBOX_SHADOW_DISPLAY_SUPPRESS_IOS", "0.62"))),
        )
        # A final guard handles the common seated/partial-body case: one box center
        # remains inside the other although raw IoU is modest because their heights
        # differ. This is display-only and is evaluated only for INACTIVE shadows.
        self.shadow_display_center_iou = max(
            0.05,
            min(0.80, float(os.environ.get("V11_BBOX_SHADOW_DISPLAY_CENTER_IOU", "0.18"))),
        )
        self._shadow_last_active_frame: dict[str, dict[int, int]] = {}
        self._shadow_display_last_ids: dict[str, tuple[int, ...]] = {}
        self._shadow_suppressed_last_ids: dict[str, tuple[int, ...]] = {}
        self._shadow_display_total: dict[str, int] = {}
        self._shadow_suppressed_total: dict[str, int] = {}
        self._shadow_display_errors: dict[str, int] = {}
        super().__init__()

        for cid in self.bbox_track_cameras:
            self._shadow_last_active_frame[cid] = {}
            self._shadow_display_last_ids[cid] = ()
            self._shadow_suppressed_last_ids[cid] = ()
            self._shadow_display_total[cid] = 0
            self._shadow_suppressed_total[cid] = 0
            self._shadow_display_errors[cid] = 0

        print(
            "CAMERA_V11_BBOX_SHADOW_DISPLAY_ARCH "
            f"cameras={','.join(sorted(self.shadow_display_cameras)) or 'off'} "
            f"frames={self.shadow_display_frames} "
            f"min_conf={self.shadow_display_min_conf:.2f} "
            f"duplicate_suppress_iou={self.shadow_display_suppress_iou:.2f} "
            f"duplicate_suppress_ios={self.shadow_display_suppress_ios:.2f} "
            f"center_iou={self.shadow_display_center_iou:.2f} "
            "mode=live-shadow-bbox display-only tracker_feedback=0",
            flush=True,
        )

    @staticmethod
    def _shadow_xyxy(row: dict) -> tuple[float, float, float, float]:
        left = float(row["left"])
        top = float(row["top"])
        return (
            left,
            top,
            left + float(row["width"]),
            top + float(row["height"]),
        )

    @staticmethod
    def _valid_shadow_row(row: dict) -> bool:
        return (
            float(row.get("width", 0.0)) > 1.0
            and float(row.get("height", 0.0)) > 1.0
            and float(row.get("left", -1.0)) >= 0.0
            and float(row.get("top", -1.0)) >= 0.0
        )

    @staticmethod
    def _intersection_over_smaller(a, b) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller = min(area_a, area_b)
        return intersection / smaller if smaller > 0.0 else 0.0

    @staticmethod
    def _center_inside(inner, outer) -> bool:
        ix1, iy1, ix2, iy2 = (float(v) for v in inner[:4])
        ox1, oy1, ox2, oy2 = (float(v) for v in outer[:4])
        cx = (ix1 + ix2) * 0.5
        cy = (iy1 + iy2) * 0.5
        return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

    def _shadow_conflicts_with_active(self, shadow_box, active_box) -> bool:
        iou = _iou_xyxy(shadow_box, active_box)
        if iou >= self.shadow_display_suppress_iou:
            return True
        if self._intersection_over_smaller(shadow_box, active_box) >= self.shadow_display_suppress_ios:
            return True
        if iou >= self.shadow_display_center_iou and (
            self._center_inside(shadow_box, active_box)
            or self._center_inside(active_box, shadow_box)
        ):
            return True
        return False

    def _tracker_output_probe(self, _pad, info, cid: str):
        # Cameras not selected for live-shadow display keep the accepted frozen
        # continuity behavior unchanged.
        if cid not in self.shadow_display_cameras:
            return super()._tracker_output_probe(_pad, info, cid)

        # Selected cameras call the NvDCF base probe directly so the old frozen
        # hold is skipped. Their only gap bridge is the live shadow rectangle.
        result = V11DeepStreamTRT86NvDCFBBoxCam01V1._tracker_output_probe(
            self, _pad, info, cid
        )
        buffer = info.get_buffer()
        if buffer is None:
            return result

        try:
            tracks = self.meta_bridge.copy_tracks(buffer, max_rows=64)
            tracker_frame = int(self._nvdcf_tracker_frames.get(cid, 0))
            active_ids = {int(row["object_id"]) for row in tracks}
            active_boxes = [self._row_xyxy(row) for row in tracks]

            last_active = self._shadow_last_active_frame.setdefault(cid, {})
            for object_id in active_ids:
                last_active[object_id] = tracker_frame

            # Read NvDCF misc metadata before adding any display-only rectangles.
            self._log_shadow_diag(cid, buffer, tracks, tracker_frame)

            if self.shadow_diag is None:
                return result

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
                    # Never draw a shadow target that was not already ACTIVE in
                    # this runtime; this excludes tentative/new ghosts.
                    continue
                inactive_frames = tracker_frame - int(last_seen)
                if inactive_frames <= 0 or inactive_frames > self.shadow_display_frames:
                    continue

                shadow_box = self._shadow_xyxy(row)
                # Never draw an old INACTIVE rectangle beside/on top of a spatially
                # equivalent current ACTIVE target, even when the two rectangles
                # differ enough in height/position that raw IoU alone would miss it.
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
                # V11 uses a one-stream mux per camera, so tracker metadata and
                # detector injection use source_id=0 at this point in the pipeline.
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

            # Bound bookkeeping even during very long runs. This does not alter
            # NvDCF state; it only forgets obsolete display eligibility history.
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
    return V11DeepStreamTRT86NvDCFShadowDisplayV1().run()


if __name__ == "__main__":
    raise SystemExit(main())