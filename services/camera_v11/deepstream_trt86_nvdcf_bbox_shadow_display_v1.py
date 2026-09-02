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
    """Use NvDCF's live INACTIVE/shadow bbox to bridge short CAM-05 OSD gaps.

    CAM-01..04 keep the already accepted six-frame frozen continuity layer. For
    selected cameras (CAM-05 by default), this class bypasses that frozen hold and
    instead draws the current `NVDS_TRACKER_SHADOW_LIST_META` rectangle for a very
    short window after the target leaves ACTIVE output.

    The shadow rectangle is injected only after nvtracker, on its src probe, so it
    cannot feed back into association, ReAssoc, detector matching, counts, or future
    tracker state. A recently-active requirement and a strict frame window prevent a
    long-lived shadow target from becoming a ghost OSD box.
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
        self._shadow_last_active_frame: dict[str, dict[int, int]] = {}
        self._shadow_display_last_ids: dict[str, tuple[int, ...]] = {}
        self._shadow_display_total: dict[str, int] = {}
        self._shadow_display_errors: dict[str, int] = {}
        super().__init__()

        for cid in self.bbox_track_cameras:
            self._shadow_last_active_frame[cid] = {}
            self._shadow_display_last_ids[cid] = ()
            self._shadow_display_total[cid] = 0
            self._shadow_display_errors[cid] = 0

        print(
            "CAMERA_V11_BBOX_SHADOW_DISPLAY_ARCH "
            f"cameras={','.join(sorted(self.shadow_display_cameras)) or 'off'} "
            f"frames={self.shadow_display_frames} "
            f"min_conf={self.shadow_display_min_conf:.2f} "
            f"duplicate_suppress_iou={self.shadow_display_suppress_iou:.2f} "
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

    def _tracker_output_probe(self, _pad, info, cid: str):
        # Keep the existing frozen continuity behavior untouched on CAM-01..04.
        if cid not in self.shadow_display_cameras:
            return super()._tracker_output_probe(_pad, info, cid)

        # On CAM-05 call the NvDCF base probe directly so the old frozen hold is
        # skipped. The only gap bridge on this camera will be the live shadow bbox.
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

            # Keep the existing diagnostic output. It reads NvDCF misc metadata
            # before we add any display-only shadow rectangles.
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
                    # Never draw a shadow target that was not already an ACTIVE
                    # target in this runtime; this excludes tentative/new ghosts.
                    continue
                inactive_frames = tracker_frame - int(last_seen)
                if inactive_frames <= 0 or inactive_frames > self.shadow_display_frames:
                    continue

                shadow_box = self._shadow_xyxy(row)
                # If a spatially equivalent current target is already active under
                # another ID, never draw the old shadow ID on top of it.
                if any(
                    _iou_xyxy(shadow_box, active_box) >= self.shadow_display_suppress_iou
                    for active_box in active_boxes
                ):
                    continue
                selected.append((object_id, row, inactive_frames))

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
            previous_ids = self._shadow_display_last_ids.get(cid, ())
            self._shadow_display_last_ids[cid] = display_ids
            if display_ids != previous_ids or (display_ids and tracker_frame % 20 == 0):
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
                    f"added={added} details={'|'.join(details) or 'none'} "
                    f"draw_total={self._shadow_display_total.get(cid, 0)}",
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
