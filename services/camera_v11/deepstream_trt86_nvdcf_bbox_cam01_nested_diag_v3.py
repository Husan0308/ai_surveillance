from __future__ import annotations

import time

from services.camera_v11.deepstream_trt86_multi_v1 import map_detector_boxes_to_display
from services.camera_v11.deepstream_trt86_nvdcf_bbox_cam01_nested_v2 import (
    V11DeepStreamTRT86NvDCFBBoxCam01NestedV2,
)
from services.camera_v11.deepstream_trt86_nvdcf_bbox_cam01_v1 import (
    FRESH_OVERLAP_DIAGNOSTIC_SEC,
    _center_inside,
    _iou_xyxy,
    _track_xyxy,
)


class V11DeepStreamTRT86NvDCFBBoxCam01NestedDiagV3(
    V11DeepStreamTRT86NvDCFBBoxCam01NestedV2
):
    """Measure why a fresh CAM-01 detector person does not become a distinct NvDCF target.

    This is diagnostic-only. It does not change detector boxes, tracker metadata,
    target creation thresholds, association weights, continuity, or shadow display.
    For each detector cycle actually injected into NvDCF, it logs every detector
    person's confidence and maximum IoU against the current active tracker boxes.
    """

    def __init__(self) -> None:
        self._cam01_nested_diag_last_sequence = 0
        self._cam01_nested_diag_errors = 0
        super().__init__()
        print(
            "CAMERA_V11_BBOX_CAM01_NESTED_DIAG_V3_ARCH "
            "camera=CAM-01 mode=read-only fields=conf,max_iou,best_track,nested,new_candidate "
            "min_iou_diff=0.80 tracker_feedback=0",
            flush=True,
        )

    def _tracker_output_probe(self, _pad, info, cid: str):
        # Inspect tracker truth before the parent continuity layer can append any
        # display-only held rectangle to NvDsObjectMeta.
        buffer = info.get_buffer()
        if cid == "CAM-01" and buffer is not None:
            try:
                now = time.monotonic()
                with self.lock:
                    state = self.states[cid]
                    snapshot = state.latest_snapshot
                    sequence = int(snapshot.sequence)
                    injected_sequence = int(self._nvdcf_last_injected_sequence.get(cid, 0))
                    age = (
                        now - snapshot.completed_mono
                        if snapshot.completed_mono > 0
                        else 999.0
                    )
                    raw_boxes = tuple(snapshot.boxes)

                if (
                    sequence > self._cam01_nested_diag_last_sequence
                    and sequence == injected_sequence
                    and age <= FRESH_OVERLAP_DIAGNOSTIC_SEC
                ):
                    self._cam01_nested_diag_last_sequence = sequence
                    tracks = self.meta_bridge.copy_tracks(buffer, max_rows=64)
                    track_boxes = [_track_xyxy(row) for row in tracks]
                    scaled = (
                        map_detector_boxes_to_display(raw_boxes, self.width, self.height)
                        if raw_boxes
                        else []
                    )

                    details: list[str] = []
                    blocked_by_iou = 0
                    nested_count = 0
                    for index, det in enumerate(scaled):
                        confidence = float(det[4]) if len(det) > 4 else -1.0
                        best_iou = 0.0
                        best_track_id = -1
                        for row, track_box in zip(tracks, track_boxes):
                            score = _iou_xyxy(det, track_box)
                            if score > best_iou:
                                best_iou = score
                                best_track_id = int(row["object_id"])

                        nested = any(_center_inside(det, track_box) for track_box in track_boxes)
                        # Per NVIDIA semantics, an unmatched detector object is
                        # eligible to instantiate a distinct target only when its
                        # max IoU to an existing same-class target is LOWER than
                        # minIouDiff4NewTarget (0.80 in this CAM-01 profile).
                        new_candidate = best_iou < 0.80
                        if not new_candidate:
                            blocked_by_iou += 1
                        if nested:
                            nested_count += 1
                        details.append(
                            f"d{index}:conf{confidence:.3f}:maxIoU{best_iou:.3f}:"
                            f"best{best_track_id if best_track_id >= 0 else 'none'}:"
                            f"nested{int(nested)}:newCandidate{int(new_candidate)}"
                        )

                    print(
                        "CAMERA_V11_BBOX_CAM01_NESTED_DIAG "
                        f"sequence={sequence} age_ms={age * 1000.0:.1f} "
                        f"detector={len(scaled)} active_tracks={len(tracks)} "
                        f"blocked_by_iou={blocked_by_iou} nested_detector_boxes={nested_count} "
                        f"details={'|'.join(details) or 'none'}",
                        flush=True,
                    )
            except Exception as exc:
                self._cam01_nested_diag_errors += 1
                errors = self._cam01_nested_diag_errors
                if errors <= 5 or errors % 100 == 0:
                    print(
                        "CAMERA_V11_BBOX_CAM01_NESTED_DIAG_WARNING "
                        f"warning={type(exc).__name__}:{exc} errors={errors}",
                        flush=True,
                    )

        return super()._tracker_output_probe(_pad, info, cid)


def main() -> int:
    return V11DeepStreamTRT86NvDCFBBoxCam01NestedDiagV3().run()


if __name__ == "__main__":
    raise SystemExit(main())
