from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _force_runtime_profile() -> None:
    """Install the production CAM-01 low-latency profile before camera modules import.

    This deliberately avoids the Pascal-safe pose monkey-patch and the old
    YOLO26m defaults. The detector is YOLO26s on CUDA, CAM-01 only, while NvDCF
    owns per-frame motion between detector refreshes.
    """

    model = ROOT / "yolo26s.pt"
    if not model.is_file():
        raise RuntimeError(f"required detector model not found: {model}")

    forced = {
        # Keep the live wall bounded and latest-frame-only. 80 ms is enough for
        # the local-LAN RTSP jitter buffer without deliberately adding 250 ms.
        "CAMERA_V2_RTSP_TRANSPORT": "tcp",
        "CAMERA_V2_RTSP_LATENCY_MS": "80",
        "CAMERA_V2_LOW_LATENCY_MODE": "1",
        "CAMERA_V2_MUX_TIMEOUT_US": "25000",
        "CAMERA_V2_SOURCE_FPS": "20",
        "CAMERA_V2_EXTRA_SURFACES": "4",
        # Do not carry native 2560/3200-wide surfaces through mux/tiler/tracker.
        # Detection taps the decoded source before mux, so detector quality is
        # independent of this display/tracker downscale.
        "CAMERA_V2_FRAME_WIDTH": "1280",
        "CAMERA_V2_FRAME_HEIGHT": "720",
        "CAMERA_V2_WALL_WIDTH": "1920",
        "CAMERA_V2_WALL_HEIGHT": "720",
        # Fast CUDA person detector. Explicitly one camera and one image per job.
        "CAMERA_V2_YOLO_MODEL": str(model),
        "CAMERA_V2_DETECT_WIDTH": "672",
        "CAMERA_V2_DETECT_HEIGHT": "384",
        "CAMERA_V2_MICRO_BATCH": "1",
        "CAMERA_V2_DETECT_ACTIVE_CAMERAS": "CAM-01",
        "CAMERA_V2_DETECT_CONF": "0.05",
        "CAMERA_V2_DETECT_IOU": "0.70",
        "CAMERA_V2_MAX_DET": "50",
        "CAMERA_V2_DETECT_STARTUP_DELAY": "1.0",
        # Keep detector cadence high enough to refresh NvDCF while leaving headroom
        # for decode/display/tracker/ReID on the GTX 1050 Ti.
        "CAMERA_V2_DETECT_TARGET_HZ": "3.0",
        "CAMERA_V2_DETECT_MIN_HZ": "2.5",
        "CAMERA_V2_DETECT_MAX_HZ": "4.0",
        "CAMERA_V2_DETECT_GPU_DUTY": "0.34",
        "CAMERA_V2_DETECT_GPU_DUTY_MIN": "0.30",
        "CAMERA_V2_DETECT_GPU_DUTY_MAX": "0.38",
        # Never accept half-second-old boxes. Slightly above the old 280 ms gate
        # gives the measured YOLO26s path room without turning latency into truth.
        "CAMERA_V2_MAX_DETECT_RESULT_AGE_MS": "320",
        # NvDCF remains the sticky display/tracking authority between detections.
        "CAMERA_V2_TRACKER_WIDTH": "512",
        "CAMERA_V2_TRACKER_HEIGHT": "288",
        "CAMERA_V2_MIN_DISPLAY_TRACK_CONF": "0.05",
        "CAMERA_V2_BOX_RENDER_AGE": "0.45",
        # Qwen is not allowed to steal GPU/CPU time from the live path.
        "QWEN_REID_ENABLED": "0",
    }

    for key, value in forced.items():
        os.environ[key] = value

    # Explicitly remove experiment selectors that could silently replace the
    # YOLO26s CUDA worker or re-enable a stale TensorRT/pose path.
    for key in (
        "CAMERA_V2_POSE_MODEL",
        "CAMERA_V2_POSE_IMGSZ",
        "CAMERA_V2_POSE_CONF",
        "CAMERA_V2_POSE_IOU",
        "CAMERA_V2_YOLO_TRT86_ENGINE",
        "CAMERA_V2_YOLO_TRT86_PYTHON",
        "CAMERA_V2_YOLO_TRT86_WORKER",
    ):
        os.environ.pop(key, None)


_force_runtime_profile()

# These imports MUST stay below _force_runtime_profile(). detection.py snapshots
# model/geometry/batch constants at import time.
from . import detection as _det  # noqa: E402
from .person_tracking_reid import CameraPersonTrackingReID  # noqa: E402


def _validate_profile() -> None:
    active = os.environ.get("CAMERA_V2_DETECT_ACTIVE_CAMERAS", "")
    expected_model = str(ROOT / "yolo26s.pt")

    errors: list[str] = []
    if str(_det.MODEL_SPEC) != expected_model:
        errors.append(f"model={_det.MODEL_SPEC!r} expected={expected_model!r}")
    if int(_det.INFER_WIDTH) != 672 or int(_det.INFER_HEIGHT) != 384:
        errors.append(
            f"detector_shape={_det.INFER_WIDTH}x{_det.INFER_HEIGHT} expected=672x384"
        )
    if int(_det.MICRO_BATCH) != 1:
        errors.append(f"micro_batch={_det.MICRO_BATCH} expected=1")
    if active != "CAM-01":
        errors.append(f"active={active!r} expected='CAM-01'")

    if errors:
        raise RuntimeError("CAM01_LOWLAT profile invalid: " + "; ".join(errors))

    print(
        "CAM01_LOWLAT_PROFILE "
        f"model={Path(str(_det.MODEL_SPEC)).name} device=cuda:0 "
        f"active={active} detector={_det.INFER_WIDTH}x{_det.INFER_HEIGHT}/micro{_det.MICRO_BATCH} "
        "rtsp=80ms mux_timeout=25ms frame=1280x720 wall=1920x720 "
        "tracker=512x288 max_result_age=320ms qwen=0",
        flush=True,
    )


def main() -> int:
    _validate_profile()
    return CameraPersonTrackingReID().run()


if __name__ == "__main__":
    raise SystemExit(main())
