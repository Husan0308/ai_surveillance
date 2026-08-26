from __future__ import annotations

import os
from pathlib import Path

# Must be resolved before detection/person_tracking modules are imported.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "672")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "1")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.08")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault(
    "CAMERA_V2_DETECT_ACTIVE_CAMERAS",
    "CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06",
)
os.environ.setdefault("CAMERA_V2_DETECT_TARGET_HZ", "0.55")
os.environ.setdefault("CAMERA_V2_DETECT_MIN_HZ", "0.35")
os.environ.setdefault("CAMERA_V2_DETECT_MAX_HZ", "0.75")
os.environ.setdefault("CAMERA_V2_MAX_DETECT_RESULT_AGE_MS", "350")
os.environ.setdefault("CAMERA_V2_TRACKER_WIDTH", "512")
os.environ.setdefault("CAMERA_V2_TRACKER_HEIGHT", "288")
os.environ.setdefault("CAMERA_V2_MIN_DISPLAY_TRACK_CONF", "0.10")

from .person_tracking_pascal_trt86 import (
    CameraPersonTrackingPascalTRT86,
    _insert_target_management_key,
    _set_key,
)


class CameraPersonTrackingTRT86RestoreStable(CameraPersonTrackingPascalTRT86):
    """Restored stable bbox path with the proven fresh TensorRT 8.6 detector.

    YOLO26s B1 only seeds fresh detections. NvDCF remains the single owner of
    camera-local IDs and every-frame bbox motion. The native smoother styles only
    existing NvDCF objects. No sticky display tracker and no external YOLO NMS.
    """

    def __init__(self) -> None:
        super().__init__()
        # Exact-frame parity proved bilinear fixes the CAM-02 recall regression.
        for index in range(len(self.cameras)):
            converter = self.pipeline.get_by_name(f"detect_convert_{index}")
            if converter is not None:
                self._set_if(converter, "interpolation-method", 1)
                self._set_if(converter, "compute-hw", 1)
        self.latency_compensator.max_projection_s = 0.10
        self.latency_compensator.projection_gain = 0.42
        print(
            "RESTORE_STABLE_TRT_ARCH detector=YOLO26s/TRT8.6/B1/672x384 "
            "capture=JIT-no-prefetch preprocess=gpu-bilinear bbox_owner=NvDCF "
            "external_nms=0 geometry_dedup=0 sticky=0",
            flush=True,
        )

    def _dedup_and_expand(self, rows):
        """Map YOLO26 E2E detections to the 5-scalar NvDCF seed contract.

        YOLO26 one-to-one E2E already returns final detections, so no external
        geometry dedup/NMS is applied here. ``_scaled_detections`` intentionally
        returns ``((x1, y1, x2, y2), conf)`` for legacy callers, while the
        DetectorLatencyCompensator consumes flat ``(x1, y1, x2, y2, conf)`` rows.
        Keep this adapter explicit so the async scheduler cannot crash after the
        first successful TensorRT result.
        """
        scaled = self._scaled_detections(rows)
        output: list[tuple[float, float, float, float, float]] = []
        for coords, conf in scaled:
            x1, y1, x2, y2 = (float(v) for v in coords)
            if x2 <= x1 or y2 <= y1:
                continue
            output.append((x1, y1, x2, y2, float(conf)))
        return output

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        lines = path.read_text(encoding="utf-8").splitlines()
        _set_key(lines, "enableBboxUnClipping", "0")
        # NVIDIA recommends lowering this threshold when duplicate trackers are
        # created from overlapping detector boxes. Keep 0.45 to preserve nearby
        # people while suppressing same-person re-spawns.
        _set_key(lines, "minIouDiff4NewTarget", "0.45")
        _set_key(lines, "minTrackerConfidence", "0.10")
        _set_key(lines, "probationAge", "1")
        _set_key(lines, "maxShadowTrackingAge", "55")
        _set_key(lines, "earlyTerminationAge", "3")
        _set_key(lines, "minTrackingConfidenceDuringInactive", "0.10", required=False)
        _set_key(lines, "minIou4TargetDuplicate", "0.90", required=False)
        _set_key(lines, "targetDuplicateRunInterval", "1", required=False)
        _insert_target_management_key(lines, "outputShadowTracks", "1")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        for required in (
            "minIouDiff4NewTarget: 0.45",
            "probationAge: 1",
            "maxShadowTrackingAge: 55",
            "outputShadowTracks: 1",
        ):
            if required not in text:
                raise RuntimeError(f"RESTORE_STABLE_NVDCF missing {required}")
        print(
            "RESTORE_STABLE_NVDCF minIouDiff4NewTarget=0.45 probationAge=1 "
            "maxShadowTrackingAge=55 duplicateIoU=0.90 duplicateInterval=1 verified=1",
            flush=True,
        )
        return path


def main() -> int:
    return CameraPersonTrackingTRT86RestoreStable().run()


if __name__ == "__main__":
    raise SystemExit(main())
