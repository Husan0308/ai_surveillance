from __future__ import annotations

import os
from pathlib import Path

# Final quality defaults must be set before importing detection/person_tracking,
# because those modules read their defaults at import time.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "640")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "1")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.14")
os.environ.setdefault("CAMERA_V2_DETECT_IOU", "0.45")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault("CAMERA_V2_DETECT_GPU_DUTY", "0.30")
os.environ.setdefault("CAMERA_V2_DETECT_GPU_DUTY_MIN", "0.18")
os.environ.setdefault("CAMERA_V2_DETECT_GPU_DUTY_MAX", "0.34")
os.environ.setdefault("CAMERA_V2_TRACKER_WIDTH", "640")
os.environ.setdefault("CAMERA_V2_TRACKER_HEIGHT", "384")
# Keep the bbox fed into NvDCF close to the detector's actual person region.
# The native bridge expands only the rendered rectangle after tracking.
os.environ.setdefault("CAMERA_V2_TRACK_BOX_SIDE_MARGIN", "0.02")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_TOP_MARGIN", "0.01")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN", "0.03")
os.environ.setdefault("CAMERA_V2_DEDUP_IOU", "0.48")
os.environ.setdefault("CAMERA_V2_DEDUP_CONTAINMENT", "0.78")

from .person_tracking import CameraPersonTrackingV2 as _BaseTracking
from .tracker_profile import prepare_sparse_tracker_config


class CameraPersonTrackingFinal(_BaseTracking):
    """Camera V2 + higher-resolution YOLO26m + balanced NvDCF tracking.

    Quality goals:
    * improve small/far-person recall with 640x384 inference;
    * keep CUDA bursts short with micro-batch 1;
    * use DeepStream's balanced NvDCF perf profile instead of max_perf when present;
    * suppress duplicate target creation;
    * keep detector-skipped frames distinct through bInferDone semantics.
    """

    def __init__(self) -> None:
        self.detector_frames_applied = 0
        super().__init__()

    def _resolve_tracker_files(self):
        lib, stock_max_perf = super()._resolve_tracker_files()
        perf = stock_max_perf.with_name("config_tracker_NvDCF_perf.yml")
        stock = perf if perf.exists() else stock_max_perf
        config = prepare_sparse_tracker_config(stock)
        return lib, config

    def _publish_detector_result(self, cid: str, boxes) -> None:
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
            f"detector=640x384/micro1 conf=0.14 nms_iou=0.45 "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"config={self.tracker_config} bInferDone_contract=1 sparse_profile=1",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonTrackingFinal().run()


if __name__ == "__main__":
    raise SystemExit(main())
