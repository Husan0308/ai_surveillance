from __future__ import annotations

import os
from pathlib import Path

# Resolve these before detection/tracking modules import their constants.
# Detector capture branches are before nvstreammux, so reducing the mux working
# frame does NOT reduce the 672x384 detector tensor or its source detail.
os.environ.setdefault("CAMERA_V2_FRAME_WIDTH", "960")
os.environ.setdefault("CAMERA_V2_FRAME_HEIGHT", "540")
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "672")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "1")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.08")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault(
    "CAMERA_V2_DETECT_ACTIVE_CAMERAS",
    "CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06",
)
# The TRT8.6 detector lives in a separate process/CUDA context because the
# DeepStream/TRT version on this host cannot build Pascal SM61 engines. Keep the
# detector sparse enough that those short context slices do not dominate display.
os.environ.setdefault("CAMERA_V2_DETECT_TARGET_HZ", "0.24")
os.environ.setdefault("CAMERA_V2_DETECT_MIN_HZ", "0.12")
os.environ.setdefault("CAMERA_V2_DETECT_MAX_HZ", "0.35")
os.environ.setdefault("CAMERA_V2_MAX_DETECT_RESULT_AGE_MS", "350")
# NVIDIA's DeepStream performance examples use reduced, 32-aligned NvDCF working
# sizes. 480x288 is sufficient for local continuity while YOLO periodically
# corrects the geometry from the full-resolution source branch.
os.environ.setdefault("CAMERA_V2_TRACKER_WIDTH", "480")
os.environ.setdefault("CAMERA_V2_TRACKER_HEIGHT", "288")
os.environ.setdefault("CAMERA_V2_MIN_DISPLAY_TRACK_CONF", "0.10")

from .person_tracking_pascal_trt86 import (
    CameraPersonTrackingPascalTRT86,
    _insert_target_management_key,
    _set_key,
)


class CameraPersonTrackingTRT86RestoreStable(CameraPersonTrackingPascalTRT86):
    """Final low-latency local tracking baseline for the Pascal GPU.

    YOLO26s/TensorRT 8.6 supplies fresh detector corrections. NvDCF is the only
    owner of camera-local IDs and bbox motion. The display smoother only modifies
    current NvDCF rectangles and never creates a second/ghost object.
    """

    def __init__(self) -> None:
        self._stable_track_probe_n = 0
        super().__init__()

        # Detector parity: GPU bilinear matched the Ultralytics reference better
        # than the old nearest-neighbour Camera V2 preprocessing path.
        for index in range(len(self.cameras)):
            converter = self.pipeline.get_by_name(f"detect_convert_{index}")
            if converter is not None:
                self._set_if(converter, "interpolation-method", 1)
                self._set_if(converter, "compute-hw", 1)

        # Six native 1080p/4MP/5MP sources are scaled to a 960x540 tracking frame.
        # Bilinear is intentionally used here: Lanczos on every live source was
        # consuming the GPU budget needed by NvDCF and the legacy TRT8.6 sidecar.
        self._set_if(self.mux, "interpolation-method", 1)
        self._set_if(self.mux, "compute-hw", 1)

        # The visible grid is only 640x360 per tile. Bilinear tiling avoids a
        # second expensive Lanczos pass on every wall frame. Source/detail quality
        # for YOLO is unaffected because inference branches before nvstreammux.
        self._set_if(self.tiler, "interpolation-method", 1)
        self._set_if(self.tiler, "compute-hw", 1)

        # GPU nvdsosd accepts NV12 directly. Remove the old full-wall RGBA pass.
        self._enable_nv12_direct_osd()

        # NvDCF—not a Python predictor—owns motion between detector observations.
        self.latency_compensator.max_projection_s = 0.08
        self.latency_compensator.projection_gain = 0.35

        print(
            "RESTORE_STABLE_TRT_ARCH detector=YOLO26s/TRT8.6/B1/672x384 "
            "capture=JIT-no-prefetch detector_preprocess=gpu-bilinear "
            f"mux={self.frame_width}x{self.frame_height}/gpu-bilinear "
            f"tracker={self.tracker_width}x{self.tracker_height}/nvdcf-max-perf "
            "tiler=gpu-bilinear osd=gpu-nv12-direct rgba_wall_convert=0 "
            "bbox_owner=NvDCF external_nms=0 geometry_dedup=0 sticky=0",
            flush=True,
        )

    def _enable_nv12_direct_osd(self) -> None:
        convert = self.pipeline.get_by_name("track_wall_convert")
        caps = self.pipeline.get_by_name("track_wall_caps")
        osd = getattr(self, "osd", None)
        if convert is None or caps is None or osd is None:
            raise RuntimeError(
                "RESTORE_STABLE_OSD expected legacy track_wall_convert/caps/osd"
            )

        # Construction is still in NULL state, so this relink is deterministic.
        self.wall_queue.unlink(convert)
        convert.unlink(caps)
        caps.unlink(osd)
        osd.unlink(self.sink)

        if not self.wall_queue.link(osd):
            raise RuntimeError("RESTORE_STABLE_OSD wall_queue -> nvdsosd NV12 link failed")
        if not osd.link(self.sink):
            raise RuntimeError("RESTORE_STABLE_OSD nvdsosd -> EGL link failed")

        try:
            self.pipeline.remove(convert)
            self.pipeline.remove(caps)
        except Exception as exc:
            raise RuntimeError(f"RESTORE_STABLE_OSD could not remove RGBA path: {exc}") from exc

        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", True)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", self.gpu_id)
        print(
            "RESTORE_STABLE_OSD path=wall_queue->nvdsosd->EGL "
            "format=NV12-direct process_mode=GPU rgba_convert=0 verified=1",
            flush=True,
        )

    def _dedup_and_expand(self, rows):
        """Map YOLO26 E2E detections to the 5-scalar NvDCF seed contract."""
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
        _set_key(lines, "minIouDiff4NewTarget", "0.72")
        _set_key(lines, "minTrackerConfidence", "0.10")
        _set_key(lines, "probationAge", "1")
        _set_key(lines, "maxShadowTrackingAge", "100")
        _set_key(lines, "earlyTerminationAge", "4")
        _set_key(lines, "minTrackingConfidenceDuringInactive", "0.08", required=False)
        _set_key(lines, "minIou4TargetDuplicate", "0.90", required=False)
        _set_key(lines, "targetDuplicateRunInterval", "1", required=False)
        # Reassert the true lightweight visual-tracker contract after all generic
        # profile generation. This prevents a future inherited config from silently
        # turning HOG/high-resolution features back on.
        _set_key(lines, "useColorNames", "1", required=False)
        _set_key(lines, "useHog", "0", required=False)
        _set_key(lines, "useHighPrecisionFeature", "0", required=False)
        _set_key(lines, "featureImgSizeLevel", "2", required=False)
        _insert_target_management_key(lines, "outputShadowTracks", "1")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        for required in (
            "minIouDiff4NewTarget: 0.72",
            "probationAge: 1",
            "maxShadowTrackingAge: 100",
            "outputShadowTracks: 1",
            "useColorNames: 1",
            "useHog: 0",
            "featureImgSizeLevel: 2",
        ):
            if required not in text:
                raise RuntimeError(f"RESTORE_STABLE_NVDCF missing {required}")
        print(
            "RESTORE_STABLE_NVDCF minIouDiff4NewTarget=0.72 probationAge=1 "
            "maxShadowTrackingAge=100 duplicateIoU=0.90 duplicateInterval=1 "
            "colorNames=1 hog=0 featureLevel=2 bbox_owner=NvDCF verified=1",
            flush=True,
        )
        return path


def main() -> int:
    return CameraPersonTrackingTRT86RestoreStable().run()


if __name__ == "__main__":
    raise SystemExit(main())
