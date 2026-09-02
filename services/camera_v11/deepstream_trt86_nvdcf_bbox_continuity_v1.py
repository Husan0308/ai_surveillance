from __future__ import annotations

import os

from services.camera_v11.deepstream_trt86_nvdcf_bbox_cam01_v1 import (
    V11DeepStreamTRT86NvDCFBBoxCam01V1,
    _iou_xyxy,
)
from services.camera_v11.shadow_meta_diag import ShadowMetaDiag


class V11DeepStreamTRT86NvDCFBBoxContinuityV1(V11DeepStreamTRT86NvDCFBBoxCam01V1):
    """Hide short NvDCF metadata gaps without feeding synthetic boxes back to tracking.

    DeepStream can keep a target alive internally while not emitting a current-frame
    NvDsObjectMeta on every sparse-inference frame. A true timestamp-aligned past-frame
    merge requires buffering/reordering video frames. For the live OSD path we instead
    keep a very short display-only copy of the last accepted NvDCF rectangle.

    The held rectangle is injected only on the tracker *src* pad, after NvDCF has already
    processed the frame, so it cannot participate in association, ReAssoc, counts, or
    future tracker state. It is also suppressed when a current track already overlaps the
    cached rectangle, which avoids a short duplicate box if NvDCF returns with a new ID.

    Separately, this runtime reads NVDS_TRACKER_SHADOW_LIST_META for diagnostics only.
    That lets us prove whether an apparent active-bbox gap still exists inside NvDCF as an
    INACTIVE/shadow target before changing any more association or lifecycle thresholds.
    """

    def __init__(self) -> None:
        self.display_hold_frames = max(
            0,
            min(12, int(os.environ.get("V11_BBOX_DISPLAY_HOLD_FRAMES", "6"))),
        )
        self.display_hold_suppress_iou = max(
            0.10,
            min(0.95, float(os.environ.get("V11_BBOX_DISPLAY_HOLD_SUPPRESS_IOU", "0.50"))),
        )
        raw_shadow_diag = os.environ.get("V11_BBOX_SHADOW_DIAG_CAMERAS", "CAM-05")
        self.shadow_diag_cameras = {
            part.strip() for part in raw_shadow_diag.split(",") if part.strip()
        }
        self._display_hold_cache: dict[str, dict[int, dict]] = {}
        self._display_hold_last_ids: dict[str, tuple[int, ...]] = {}
        self._display_hold_total: dict[str, int] = {}
        self._display_hold_errors: dict[str, int] = {}
        self._shadow_diag_last_ids: dict[str, tuple[int, ...]] = {}
        self._shadow_diag_errors: dict[str, int] = {}
        self.shadow_diag: ShadowMetaDiag | None = None
        super().__init__()
        if self.shadow_diag_cameras & set(self.bbox_track_cameras):
            self.shadow_diag = ShadowMetaDiag()
        for cid in self.bbox_track_cameras:
            self._display_hold_cache[cid] = {}
            self._display_hold_last_ids[cid] = ()
            self._display_hold_total[cid] = 0
            self._display_hold_errors[cid] = 0
            self._shadow_diag_last_ids[cid] = ()
            self._shadow_diag_errors[cid] = 0
        print(
            "CAMERA_V11_BBOX_CONTINUITY_ARCH "
            f"cameras={','.join(self.bbox_track_cameras)} "
            f"hold_frames={self.display_hold_frames} "
            f"duplicate_suppress_iou={self.display_hold_suppress_iou:.2f} "
            f"shadow_diag={','.join(sorted(self.shadow_diag_cameras)) or 'off'} "
            "mode=display-only tracker_feedback=0 past_frame_enabled=1",
            flush=True,
        )

    @staticmethod
    def _row_xyxy(row: dict) -> tuple[float, float, float, float]:
        left = float(row["left"])
        top = float(row["top"])
        return (
            left,
            top,
            left + float(row["width"]),
            top + float(row["height"]),
        )

    def _log_shadow_diag(self, cid: str, buffer, tracks: list[dict], tracker_frame: int) -> None:
        if self.shadow_diag is None or cid not in self.shadow_diag_cameras:
            return
        try:
            rows = self.shadow_diag.copy_shadow_tracks(buffer, max_rows=64)
            # Defensive de-duplication: keep the newest frame if multiple shadow
            # user-meta blocks happen to contain the same local ID.
            latest: dict[int, dict] = {}
            for row in rows:
                object_id = int(row["object_id"])
                previous = latest.get(object_id)
                if previous is None or int(row["frame_num"]) >= int(previous["frame_num"]):
                    latest[object_id] = row

            shadow_ids = tuple(sorted(latest))
            active_ids = tuple(sorted({int(row["object_id"]) for row in tracks}))
            previous_ids = self._shadow_diag_last_ids.get(cid, ())
            self._shadow_diag_last_ids[cid] = shadow_ids

            if shadow_ids != previous_ids or (shadow_ids and tracker_frame % 20 == 0):
                details = []
                for object_id in shadow_ids:
                    row = latest[object_id]
                    details.append(
                        f"{object_id}:{row['state']}:f{int(row['frame_num'])}:age{int(row['age'])}:"
                        f"conf{float(row['confidence']):.3f}:vis{float(row['visibility']):.3f}:"
                        f"bbox={float(row['left']):.1f},{float(row['top']):.1f},"
                        f"{float(row['width']):.1f},{float(row['height']):.1f}"
                    )
                print(
                    "CAMERA_V11_BBOX_SHADOW "
                    f"camera={cid} tracker_frame={tracker_frame} "
                    f"active_ids={','.join(map(str, active_ids)) or 'none'} "
                    f"shadow_ids={','.join(map(str, shadow_ids)) or 'none'} "
                    f"shadow_count={len(shadow_ids)} "
                    f"details={'|'.join(details) or 'none'}",
                    flush=True,
                )
        except Exception as exc:
            errors = self._shadow_diag_errors.get(cid, 0) + 1
            self._shadow_diag_errors[cid] = errors
            if errors <= 5 or errors % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_SHADOW_WARNING "
                    f"camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )

    def _tracker_output_probe(self, _pad, info, cid: str):
        # Preserve the accepted NvDCF styling, diagnostics and tracker truth first.
        result = super()._tracker_output_probe(_pad, info, cid)

        buffer = info.get_buffer()
        if buffer is None:
            return result

        # Read NvDCF's own shadow-list metadata before adding any display-only held
        # rectangle. This diagnostic never mutates tracker or OSD metadata.
        tracks_for_diag = self.meta_bridge.copy_tracks(buffer, max_rows=64)
        tracker_frame = int(self._nvdcf_tracker_frames.get(cid, 0))
        self._log_shadow_diag(cid, buffer, tracks_for_diag, tracker_frame)

        if self.display_hold_frames <= 0:
            return result

        try:
            tracks = tracks_for_diag
            cache = self._display_hold_cache.setdefault(cid, {})

            current_ids: set[int] = set()
            current_boxes: list[tuple[float, float, float, float]] = []
            for row in tracks:
                object_id = int(row["object_id"])
                current_ids.add(object_id)
                current_boxes.append(self._row_xyxy(row))
                # Copy the row because ctypes-backed data must never be retained.
                cache[object_id] = {
                    "frame": tracker_frame,
                    "row": dict(row),
                }

            held_rows: list[tuple[int, dict, int]] = []
            expired_ids: list[int] = []
            for object_id, entry in list(cache.items()):
                if object_id in current_ids:
                    continue
                age_frames = tracker_frame - int(entry["frame"])
                if age_frames <= 0:
                    continue
                if age_frames > self.display_hold_frames:
                    expired_ids.append(object_id)
                    continue

                row = entry["row"]
                cached_box = self._row_xyxy(row)
                # If NvDCF already emitted a spatially equivalent current box under
                # another ID, do not draw the stale held box as a duplicate.
                if any(
                    _iou_xyxy(cached_box, current_box) >= self.display_hold_suppress_iou
                    for current_box in current_boxes
                ):
                    continue
                held_rows.append((object_id, row, age_frames))

            for object_id in expired_ids:
                cache.pop(object_id, None)

            held_boxes = []
            for object_id, row, _age_frames in held_rows:
                x1, y1, x2, y2 = self._row_xyxy(row)
                conf = max(
                    0.01,
                    float(row.get("confidence", -1.0)),
                    float(row.get("tracker_confidence", -1.0)),
                )
                held_boxes.append((object_id, x1, y1, x2, y2, conf))

            added = 0
            if held_boxes:
                # Each V11 camera has a one-stream mux and detector injection already
                # uses source_id=0. This call occurs after nvtracker, before OSD.
                added = self.meta_bridge.add_tracked_boxes(buffer, 0, held_boxes)
                if added < 0:
                    raise RuntimeError(f"display continuity add returned {added}")
                self._display_hold_total[cid] = self._display_hold_total.get(cid, 0) + int(added)

            held_ids = tuple(sorted(int(row[0]) for row in held_rows))
            previous_held = self._display_hold_last_ids.get(cid, ())
            self._display_hold_last_ids[cid] = held_ids
            if held_ids != previous_held or (held_ids and tracker_frame % 20 == 0):
                ages = ",".join(str(age) for _oid, _row, age in held_rows) or "none"
                ids_text = ",".join(str(value) for value in held_ids) or "none"
                print(
                    "CAMERA_V11_BBOX_CONTINUITY "
                    f"camera={cid} tracker_frame={tracker_frame} current={len(current_ids)} "
                    f"held={len(held_ids)} added={added} held_ids={ids_text} ages={ages} "
                    f"hold_total={self._display_hold_total.get(cid, 0)}",
                    flush=True,
                )
        except Exception as exc:
            errors = self._display_hold_errors.get(cid, 0) + 1
            self._display_hold_errors[cid] = errors
            if errors <= 5 or errors % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_CONTINUITY_WARNING "
                    f"camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )
        return result


def main() -> int:
    return V11DeepStreamTRT86NvDCFBBoxContinuityV1().run()


if __name__ == "__main__":
    raise SystemExit(main())
