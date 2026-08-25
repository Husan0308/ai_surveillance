from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.camera_wall_runtime import (  # noqa: E402
    CAMERA_COUNT,
    GRID_COLUMNS,
    GRID_ROWS,
    WALL_HEIGHT,
    WALL_WIDTH,
)
from services.camera_v2.detection import INFER_HEIGHT, INFER_WIDTH, MICRO_BATCH  # noqa: E402
from services.camera_v2.native_bridge import ensure_bridge  # noqa: E402
from services.camera_v2.pascal_safe_pipeline import CameraPascalSafeRuntime  # noqa: E402
from services.ml_service.app.config import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    camera_count = len(settings.cameras)
    if camera_count != CAMERA_COUNT:
        raise RuntimeError(
            f"production camera wall requires exactly {CAMERA_COUNT} enabled cameras, got {camera_count}"
        )
    if os.environ.get("CAMERA_V2_PASCAL_SAFE", "0") != "1":
        raise RuntimeError("production launcher must set CAMERA_V2_PASCAL_SAFE=1")

    transport = os.environ.get("CAMERA_V2_RTSP_TRANSPORT", "").strip().lower()
    latency = int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "0"))
    if transport != "tcp":
        raise RuntimeError(f"production RTSP transport must be tcp, got {transport!r}")
    if latency < 200:
        raise RuntimeError(f"production RTSP latency must be >=200ms, got {latency}ms")

    mux_width = int(os.environ.get("CAMERA_V2_FRAME_WIDTH", "0"))
    mux_height = int(os.environ.get("CAMERA_V2_FRAME_HEIGHT", "0"))
    if (mux_width, mux_height) != (2560, 1440):
        raise RuntimeError(
            f"source-preserving mux must be 2560x1440, got {mux_width}x{mux_height}"
        )
    if (WALL_WIDTH, WALL_HEIGHT) != (1072, 904):
        raise RuntimeError(
            f"monitoring wall must be display-matched NVMM-safe 1072x904, got {WALL_WIDTH}x{WALL_HEIGHT}"
        )
    if (GRID_COLUMNS, GRID_ROWS) != (2, 3):
        raise RuntimeError(f"monitoring grid must be 2x3, got {GRID_COLUMNS}x{GRID_ROWS}")
    if (INFER_WIDTH, INFER_HEIGHT, MICRO_BATCH) != (672, 384, 1):
        raise RuntimeError(
            "detector geometry must be 672x384 micro-batch=1, got "
            f"{INFER_WIDTH}x{INFER_HEIGHT} micro={MICRO_BATCH}"
        )

    detector_backend = os.environ.get(
        "CAMERA_V2_DETECTOR_BACKEND",
        "",
    ).strip().lower()
    detector_model = Path(
        os.environ.get("CAMERA_V2_YOLO_MODEL", "")
    ).name
    detector_cameras = [
        value.strip()
        for value in os.environ.get(
            "CAMERA_V2_DETECT_ACTIVE_CAMERAS",
            "",
        ).split(",")
        if value.strip()
    ]
    if detector_backend != "yolo":
        raise RuntimeError(
            f"production detector backend must be yolo, got {detector_backend!r}"
        )
    if detector_model != "yolo26s.pt":
        raise RuntimeError(
            f"production detector model must be yolo26s.pt, got {detector_model!r}"
        )
    if detector_cameras != ["CAM-01"]:
        raise RuntimeError(
            f"CAM-01 tuning profile must detect only CAM-01, got {detector_cameras!r}"
        )
    if os.environ.get("CAMERA_V2_SINGLE_SOURCE_ANALYSIS", "0") != "1":
        raise RuntimeError("CAM-01 tuning profile must enable single-source analysis")

    runtime_source = (ROOT / "services/camera_v2/pascal_safe_pipeline.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "class CameraPascalSafeRuntime(CameraDetectionV2)",
        "SecureCameraWallV2._add_camera(self, index, camera)",
        "def _install_analysis_inference(self)",
        "pascal_mux_tee",
        "pascal_analysis_tiler",
        "def _analysis_gate_probe",
        "def _on_analysis_sample",
        "CAMERA_DETECT_PATH mode=analysis-tiler",
        "source_path=direct-to-nvstreammux",
        "demux=disabled",
        "mux_batch_retention=bounded",
        "self.wall_queue.unlink(self.sink)",
        "safe_mux_batches",
        "safe_wall_frames",
        "safe_sink_buffers",
        "analysis_frames",
        "CAMERA_STARTUP_STALL",
        "tracker=motion-predictor",
        "nvtracker=disabled",
    ):
        if required not in runtime_source:
            raise RuntimeError(f"Pascal-safe runtime guard missing: {required}")

    controller_source = (
        ROOT / "services/camera_v2/camera_wall_runtime.py"
    ).read_text(encoding="utf-8")
    for required in (
        "CAMERA_DETECTOR_POLICY",
        "base_yolo_worker = detection_module._yolo_worker",
        "detection_module._yolo_worker = base_yolo_worker",
        "CameraPascalSafeRuntime.ANALYSIS_COLUMNS = 1",
        "CameraPascalSafeRuntime.ANALYSIS_ROWS = 1",
        'analysis_tiler.set_property("show-source", analysis_source)',
    ):
        if required not in controller_source:
            raise RuntimeError(f"CAM-01 detector fast-path guard missing: {required}")

    nvdcf_ab_enabled = (
        os.environ.get("CAMERA_V2_NVDCF_AB", "0")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )

    forbidden_dependencies = [
        "nvstreamdemux",
        "pascal_infer_demux",
        "CameraPersonTrackingV2",
        "CameraPersonTrackingFinal",
    ]

    # Normal Pascal-safe production mode still forbids NvDCF.
    # Explicit A/B mode may reference the isolated NvDCF branch.
    if not nvdcf_ab_enabled:
        forbidden_dependencies.extend([
            "libnvds_nvmultiobjecttracker",
            "config_tracker_NvDCF",
        ])

    for forbidden in forbidden_dependencies:
        if forbidden in runtime_source:
            raise RuntimeError(
                f"Pascal-safe runtime leaked forbidden dependency: {forbidden}"
            )

    secure_source = (ROOT / "services/camera_v2/secure.py").read_text(encoding="utf-8")
    for required in (
        'CAMERA_V2_RTSP_TRANSPORT',
        'self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)',
        'self._set_if(element, "protocols", 4)',
        "def _source_pad_added",
        "caps.is_any()",
    ):
        if required not in secure_source:
            raise RuntimeError(f"RTSP ingest guard missing: {required}")

    sample_source = inspect.getsource(CameraPascalSafeRuntime._on_analysis_sample)
    for required in (
        "mapped_size",
        "row_stride",
        "ANALYSIS_TILE_WIDTH",
        "ANALYSIS_TILE_HEIGHT",
        "CAMERA_INFER_LAYOUT",
    ):
        if required not in sample_source:
            raise RuntimeError(f"analysis capture is missing: {required}")

    display_source = (ROOT / "services/camera_v2/dynamic_wall.py").read_text(
        encoding="utf-8"
    )
    for required in (
        'self._set_if(self.mux, "interpolation-method", 4)',
        'self._set_if(self.tiler, "interpolation-method", 4)',
        'self._set_if(self.mux, "compute-hw", 1)',
        'self._set_if(self.tiler, "compute-hw", 1)',
    ):
        if required not in display_source:
            raise RuntimeError(f"GPU display scaling guard missing: {required}")

    main_source = (ROOT / "services/camera_v2/main.py").read_text(encoding="utf-8")
    for required in (
        'self._set_if(self.sink, "sync", False)',
        'self._set_if(self.sink, "qos", False)',
        'self._set_if(self.sink, "async", False)',
        'self._set_if(self.sink, "force-aspect-ratio", True)',
    ):
        if required not in main_source:
            raise RuntimeError(f"live display sink guard missing: {required}")

    ensure_bridge()

    print(
        f"CAMERA_PREFLIGHT cameras={camera_count} core=PASS heatmap=OFF "
        f"rtsp={transport} latency={latency}ms mux={mux_width}x{mux_height} "
        f"grid={WALL_WIDTH}x{WALL_HEIGHT} "
        f"tile={WALL_WIDTH // GRID_COLUMNS}x{WALL_HEIGHT // GRID_ROWS} "
        f"detector=YOLO26s@{INFER_WIDTH}x{INFER_HEIGHT}/micro{MICRO_BATCH} "
        "active=CAM-01 detector_path=analysis-tiler(single-source-fastpath) "
        "demux=disabled mux_retention=bounded source_ingest=isolated "
        "dynamic_pad=late-caps-safe tracker=motion-predictor nvtracker=disabled "
        "capture=single-source-analysis stage_counters=source+mux+wall+sink+analysis"
    )
    print("CAMERA_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
