from __future__ import annotations

import threading

from .person_tracking import CameraPersonTrackingV2 as _BaseTracking
from .tracker_profile import prepare_sparse_tracker_config


class CameraPersonTrackingFinal(_BaseTracking):
    """Final Camera V2 person detector + sparse-aware NvDCF tracking.

    Fixes two issues from the first NvDCF integration:
    1) external YOLO results now emulate primary-inference frame metadata by setting
       NvDsFrameMeta.bInferDone only on frames where YOLO actually ran;
    2) the stock max_perf tracker profile is patched for sparse detections so a new
       person is activated immediately and duplicate-target creation is stricter.
    """

    def __init__(self) -> None:
        self.detector_frames_applied = 0
        super().__init__()

    def _resolve_tracker_files(self):
        lib, stock = super()._resolve_tracker_files()
        config = prepare_sparse_tracker_config(stock)
        return lib, config

    def _publish_detector_result(self, cid: str, boxes) -> None:
        # Publish EVERY YOLO result, including an empty result. The native bridge
        # marks that source frame bInferDone=True, matching the semantics of PGIE.
        # Frames between detector calls are left bInferDone=False so NvDCF knows
        # inference was skipped rather than interpreting it as a false negative.
        with self.pending_lock:
            self.pending_seq += 1
            self.pending[cid] = (self.pending_seq, list(boxes))

    def _inject_detector_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        boxes_added = 0
        frames_applied = 0
        with self.pending_lock:
            pending = dict(self.pending)

        for cid, source_id in self.camera_index.items():
            row = pending.get(cid)
            if row is None:
                continue
            seq, boxes = row
            if seq <= self.injected_seq.get(cid, 0):
                continue

            result = self.bridge.apply_detector_result(buffer, source_id, boxes)
            # -2 means this partial mux batch did not contain this source yet.
            if result == -2:
                continue
            if result < 0:
                continue

            self.injected_seq[cid] = seq
            frames_applied += 1
            boxes_added += result

        if frames_applied or boxes_added:
            with self.det_lock:
                self.detector_frames_applied += frames_applied
                self.meta_boxes += boxes_added
        return self.Gst.PadProbeReturn.OK

    def _tracker_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            # Re-apply OSD border style after nvtracker updates rect_params and count
            # real tracked objects. This closes the case where tracking is valid but
            # the downstream OSD receives zero-width/default styling.
            count = self.bridge.style_and_count_tracked(buffer)
            if count >= 0:
                with self.det_lock:
                    self.tracked_now = count
                    self.tracker_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self.det_lock:
            applied = self.detector_frames_applied
            tracked = self.tracked_now
        print(
            "CAMERA_TRACK_FINAL "
            f"detector_frames={applied} tracked_now={tracked} "
            f"config={self.tracker_config} bInferDone_contract=1 sparse_profile=1",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonTrackingFinal().run()


if __name__ == "__main__":
    raise SystemExit(main())
