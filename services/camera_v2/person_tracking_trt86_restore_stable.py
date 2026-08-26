from __future__ import annotations

import os
from pathlib import Path

# Resolve these before detection/tracking modules import their constants.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "672")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "1")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.08")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault(
    "CAMERA_V2_DETECT_ACTIVE_CAMERAS",
    "CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06",
)
# Sparse detector + visual tracker.  Start conservatively so the detector cannot
# starve the six-camera display; CameraPersonTrackingFinal adapts within this range
# from measured wall latency.
os.environ.setdefault("CAMERA_V2_DETECT_TARGET_HZ", "0.30")
os.environ.setdefault("CAMERA_V2_DETECT_MIN_HZ", "0.20")
os.environ.setdefault("CAMERA_V2_DETECT_MAX_HZ", "0.45")
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
    """Final low-latency local tracking baseline.

    YOLO26s/TensorRT 8.6 only supplies fresh detector corrections. NvDCF owns
    camera-local IDs and bbox motion on every live frame. The native smoother may
    change presentation geometry only; it never creates another track. The wall
    stays NVMM/NV12 through GPU OSD so there is no full-wall RGBA conversion.
    """

    def __init__(self) -> None:
        super().__init__()

        # 1) Detector parity: CAM-02 tests proved GPU bilinear matches the
        # Ultralytics reference substantially better than the default nearest path.
        for index in range(len(self.cameras)):
            converter = self.pipeline.get_by_name(f"detect_convert_{index}")
            if converter is not None:
                self._set_if(converter, "interpolation-method", 1)
                self._set_if(converter, "compute-hw", 1)

        # 2) The NVR sources are 2560/3200 wide while nvstreammux is 1280x720.
        # Lanczos on all six live inputs consumed GPU headroom needed by NvDCF/TRT.
        # Bilinear is the low-latency mux downscale; the final 1920x720 tiler keeps
        # the high-quality presentation interpolation configured by DynamicWall.
        self._set_if(self.mux, "interpolation-method", 1)
        self._set_if(self.mux, "compute-hw", 1)

        # 3) GPU nvdsosd accepts NV12 directly. The legacy tracking graph inserted
        # NV12 -> RGBA on every 1920x720 wall frame solely for OSD. Remove that
        # conversion before PLAYING while preserving the existing OSD probes.
        self._enable_nv12_direct_osd()

        # Keep detector projection small: NvDCF, not a Python motion predictor,
        # remains authoritative between detector observations.
        self.latency_compensator.max_projection_s = 0.10
        self.latency_compensator.projection_gain = 0.42

        print(
            "RESTORE_STABLE_TRT_ARCH detector=YOLO26s/TRT8.6/B1/672x384 "
            "capture=JIT-no-prefetch detector_preprocess=gpu-bilinear "
            "mux_scale=gpu-bilinear tiler_scale=quality "
            "osd=gpu-nv12-direct rgba_wall_convert=0 "
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

        # Construction is still in NULL state, so the wall chain can be relinked
        # deterministically without pad blocking or transient frames.
        self.wall_queue.unlink(convert)
        convert.unlink(caps)
        caps.unlink(osd)
        osd.unlink(self.sink)

        if not self.wall_queue.link(osd):
            raise RuntimeError("RESTORE_STABLE_OSD wall_queue -> nvdsosd NV12 link failed")
        if not osd.link(self.sink):
            raise RuntimeError("RESTORE_STABLE_OSD nvdsosd -> EGL link failed")

        # They are now completely detached; remove them so no accidental state or
        # allocation is carried by the pipeline.
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
        """Map YOLO26 E2E detections to the 5-scalar NvDCF seed contract.

        YOLO26 one-to-one E2E already returns final detections, so no external
        geometry dedup/NMS is applied here. ``_scaled_detections`` intentionally
        returns ``((x1, y1, x2, y2), conf)`` for legacy callers, while the
        DetectorLatencyCompensator consumes flat ``(x1, y1, x2, y2, conf)`` rows.
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

        # Do not use an aggressive 0.45 new-target gate here: the production logs
        # showed YOLO detecting several nearby people while NvDCF instantiated only
        # one. YOLO26 E2E already removes detector duplicates; 0.72 preserves close
        # real people, while NvDCF's dedicated duplicate-target pass handles nearly
        # identical re-spawns.
        _set_key(lines, "minIouDiff4NewTarget", "0.72")
        _set_key(lines, "minTrackerConfidence", "0.10")
        _set_key(lines, "probationAge", "1")

        # At the low-latency detector floor (0.20 Hz/cam), a detector refresh can
        # be ~5 s apart. NvDCF visual localization must be allowed to bridge normal
        # sparse-detector gaps; shadow output is the tracker’s real current-frame
        # state, not a fabricated sticky rectangle.
        _set_key(lines, "maxShadowTrackingAge", "100")
        _set_key(lines, "earlyTerminationAge", "4")
        _set_key(lines, "minTrackingConfidenceDuringInactive", "0.08", required=False)

        # Strict duplicate cleanup, without suppressing two genuinely close people.
        _set_key(lines, "minIou4TargetDuplicate", "0.90", required=False)
        _set_key(lines, "targetDuplicateRunInterval", "1", required=False)
        _insert_target_management_key(lines, "outputShadowTracks", "1")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        for required in (
            "minIouDiff4NewTarget: 0.72",
            "probationAge: 1",
            "maxShadowTrackingAge: 100",
            "outputShadowTracks: 1",
        ):
            if required not in text:
                raise RuntimeError(f"RESTORE_STABLE_NVDCF missing {required}")
        print(
            "RESTORE_STABLE_NVDCF minIouDiff4NewTarget=0.72 probationAge=1 "
            "maxShadowTrackingAge=100 duplicateIoU=0.90 duplicateInterval=1 "
            "bbox_owner=NvDCF verified=1",
            flush=True,
        )
        return path


def main() -> int:
    return CameraPersonTrackingTRT86RestoreStable().run()


if __name__ == "__main__":
    raise SystemExit(main())
