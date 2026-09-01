from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_camera_v11_deepstream_yolo_cam01_cam02_cam03_v1.sh"
CHECKER = ROOT / "scripts/check_camera_v11_deepstream_yolo_cam01_cam02_v1_log.py"
RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_v1.py"

ARCH = (
    "CAMERA_V11_DS_YOLO_MULTI_ARCH cameras=3 camera_ids=CAM-01,CAM-02,CAM-03 "
    "rtsp_sources=3 rtsp_sessions=3 rtsp_per_camera=1 decode=deepstream-nvdec "
    "source=nvurisrcbin tee_per_camera=1 display=independent-nvstreammux+nvdsosd "
    "detector=shared-trt86-sidecar detector_workers=1 detector_rtsp=0 "
    "detector_queue=latest1-per-camera detector_thread=dedicated scheduler=round-robin "
    "gst_nvinfer=0 second_rtsp=0 opencv=0 ffmpeg=0 tracker=0 reid=0 face=0 ui=0\n"
)
THREAD = "CAMERA_V11_DS_YOLO_MULTI_DETECTOR_THREAD state=START workers=1 cameras=3 scheduler=round-robin\n"
START = "CAMERA_V11_DS_YOLO_MULTI_START cameras=CAM-01,CAM-02,CAM-03 state=async\n"


def stat(camera: str, infer_hz: float = 1.6, render_fps: float = 16.0) -> str:
    return (
        "CAMERA_V11_DS_YOLO_MULTI "
        f"camera={camera} source_fps=20.00 render_fps={render_fps:.2f} infer_hz={infer_hz:.2f} "
        "queue=0 infer_count=100 infer_admitted=100 detector_drops=1000 positive_inferences=0 "
        "detections_total=0 max_objects=0 latest_boxes=0 result_clears=0 stale_expirations=0 "
        "metadata_added=0 result_age_ms=300.0 infer_p95_ms=22.0 render_gap_p95_ms=140.0 "
        "detector_thread_alive=1 worker_alive=1 copy_errors=0 infer_errors=0 meta_errors=0 "
        "warnings=0 pipeline_errors=0\n"
    )


def body(infer_cam03: float = 1.6) -> str:
    text = ARCH + THREAD + START
    for _ in range(5):
        text += stat("CAM-01") + stat("CAM-02") + stat("CAM-03", infer_hz=infer_cam03)
    return text


def run_checker(tmp_path: Path, text: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "multi3.log"
    path.write_text(text)
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--log",
            str(path),
            "--required-cameras",
            "CAM-01,CAM-02,CAM-03",
            "--min-runtime-stats",
            "5",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_locks_exact_three_camera_increment() -> None:
    source = LAUNCHER.read_text()
    assert 'BRANCH_EXPECTED="rebuild/service-architecture-v11-deepstream-yolo-cam01-cam02-cam03-v1-20260901"' in source
    assert 'CAMERAS="${V11_DS_YOLO_CAMERAS:-CAM-01,CAM-02,CAM-03}"' in source
    assert 'camera_set=$CAMERAS expected=CAM-01,CAM-02,CAM-03' in source
    assert 'detector_workers=1' in source
    assert 'detector_rtsp=0' in source
    assert 'second_rtsp=0' in source


def test_runtime_remains_generic_and_single_worker() -> None:
    source = RUNTIME.read_text()
    assert source.count("Step2TRT86Client()") == 1
    assert 'scheduler=round-robin' in source
    assert 'detector_workers=1' in source
    assert '"max-size-buffers", 1' in source
    assert '"leaky", 2' in source
    assert '"max-buffers", 1' in source


def test_checker_accepts_three_fair_cameras(tmp_path: Path) -> None:
    result = run_checker(tmp_path, body())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT=PASS" in result.stdout


def test_checker_rejects_cam03_starvation(tmp_path: Path) -> None:
    result = run_checker(tmp_path, body(infer_cam03=0.5))
    assert result.returncode == 1
    assert "CAM-03:infer_hz_min=0.50<1.20" in result.stdout
