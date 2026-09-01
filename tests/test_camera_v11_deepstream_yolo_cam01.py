from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from services.camera_v11.deepstream_trt86_cam01_v2 import (
    DetectorSnapshot,
    MODEL_PAD_TOP,
    map_detector_boxes_to_display,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_cam01_v2.py"
LAUNCHER = ROOT / "scripts/run_camera_v11_deepstream_yolo_cam01_v1.sh"
CHECKER = ROOT / "scripts/check_camera_v11_deepstream_yolo_cam01_v1_log.py"
CONFIG = ROOT / "services/ml_service/app/config.py"

ARCH = (
    "CAMERA_V11_DS_YOLO_CAM01_ARCH camera=CAM-01 rtsp_sources=1 rtsp_sessions=1 "
    "decode=deepstream-nvdec source=nvurisrcbin tee=1 display=nvstreammux+nvdsosd "
    "detector=trt86-sidecar detector_rtsp=0 detector_queue=latest1 "
    "detector_thread=dedicated gst_nvinfer=0 second_rtsp=0 opencv=0 ffmpeg=0\n"
)
POLICY = (
    "CAMERA_V11_DS_YOLO_CAM01_POLICY transport=tcp detector=672x384 content_h=378 "
    "enabled=1 model_pad_top=3\n"
)
START = "CAMERA_V11_DS_YOLO_CAM01_START camera=CAM-01 state=async\n"
THREAD = "CAMERA_V11_DS_YOLO_CAM01_DETECTOR_THREAD state=START\n"


def stat(**overrides) -> str:
    values = {
        "source_fps": 20.0,
        "render_fps": 20.0,
        "infer_hz": 2.0,
        "queue": 0,
        "source_queue": 0,
        "display_queue": 0,
        "detector_queue": 0,
        "infer_buffers": 20,
        "positive_buffers": 5,
        "detections_total": 5,
        "max_objects": 1,
        "latest_boxes": 1,
        "result_clears": 0,
        "stale_expirations": 0,
        "metadata_added": 50,
        "detector_thread_alive": 1,
        "worker_alive": 1,
        "warnings": 0,
        "copy_errors": 0,
        "infer_errors": 0,
        "meta_errors": 0,
        "pipeline_errors": 0,
    }
    values.update(overrides)
    return "CAMERA_V11_DS_YOLO_CAM01 " + " ".join(f"{k}={v}" for k, v in values.items()) + "\n"


def run_checker(tmp_path: Path, body: str, *args: str) -> subprocess.CompletedProcess[str]:
    log = tmp_path / "camera.log"
    log.write_text(body)
    return subprocess.run(
        [sys.executable, str(CHECKER), "--log", str(log), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_trt_padding_coordinates_map_to_display_edges() -> None:
    assert MODEL_PAD_TOP == 3
    rows = [
        [0.0, 3.0, 672.0, 381.0, 0.75],
        [-5.0, 0.0, 700.0, 384.0, 1.5],
        [10.0, 0.0, 20.0, 2.0, 0.9],
    ]
    mapped = map_detector_boxes_to_display(rows, 640, 360)
    assert mapped[0] == pytest.approx([0.0, 0.0, 640.0, 360.0, 0.75])
    assert mapped[1] == pytest.approx([0.0, 0.0, 640.0, 360.0, 1.0])
    assert len(mapped) == 2  # the box entirely inside top padding has no video area


def test_detector_snapshot_is_immutable() -> None:
    snapshot = DetectorSnapshot(boxes=((1.0, 2.0, 3.0, 4.0, 0.9),), sequence=7)
    with pytest.raises(Exception):
        snapshot.sequence = 8  # type: ignore[misc]


def test_runtime_has_one_source_and_no_streaming_inference_callback() -> None:
    source = RUNTIME.read_text()
    assert source.count('self._make("nvurisrcbin"') == 1
    assert 'appsink.connect("new-sample"' not in source
    assert 'self._set_if(appsink, "emit-signals", False)' in source
    assert 'self.appsink.emit("try-pull-sample"' in source
    detector_loop = source[source.index("    def _detector_loop"):source.index("    def _display_meta_probe")]
    assert "infer_preloaded" in detector_loop
    for callback in ("_source_probe", "_infer_gate_probe", "_display_meta_probe", "_render_probe"):
        section = source[source.index(f"    def {callback}"):]
        section = section.split("\n    def ", 1)[0]
        assert "infer_preloaded" not in section
    assert '"max-size-buffers", 1' in source
    assert '"leaky", 2' in source
    assert '"max-buffers", 1' in source
    assert "cv2.VideoCapture" not in source
    assert "av.open(" not in source


def test_launcher_is_branch_locked_and_uses_canonical_env_and_current_runtime() -> None:
    source = LAUNCHER.read_text()
    assert 'CURRENT_BRANCH="$(git branch --show-current' in source
    assert 'ENV_FILE="${V11_DS_YOLO_ENV_FILE:-$TRT_HOME/.env}"' in source
    assert 'V11_ENV_FILE="$ENV_FILE"' in source
    assert "nvv4l2decoder" in source
    assert "services.camera_v11.deepstream_trt86_cam01_v2" in source
    launch_line = source.splitlines()[-1]
    assert "deepstream_yolo_cam01_v1" not in launch_line
    assert ".pt" not in source
    assert "gst-nvinfer" not in source
    assert "V11_ENV_FILE" in CONFIG.read_text()
    assert "start_new_session=True" in (ROOT / "services/camera_v11/step2_trt86.py").read_text()


def test_checker_accepts_valid_person_and_expiry_log(tmp_path: Path) -> None:
    body = ARCH + POLICY + THREAD + START + stat() + stat(latest_boxes=0, result_clears=1)
    result = run_checker(
        tmp_path,
        body,
        "--require-person",
        "--require-stale-expiry",
        "--min-runtime-stats",
        "2",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT=PASS" in result.stdout


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            ARCH.replace("second_rtsp=0", "second_rtsp=1") + POLICY + THREAD + START + stat() + stat(),
            "arch_second_rtsp",
        ),
        (ARCH + POLICY + THREAD + START + stat() + stat(detector_queue=2, queue=2), "detector_queue=2>1"),
        (ARCH + POLICY + THREAD + START + stat() + stat(infer_errors=1), "infer_errors=1"),
        (ARCH + POLICY + THREAD + START + stat() + stat(warnings=1), "warnings=1"),
        (ARCH + POLICY + THREAD + START + "Traceback (most recent call last)\n" + stat() + stat(), "runtime_traceback"),
    ],
)
def test_checker_rejects_architecture_queue_and_runtime_failures(
    tmp_path: Path, body: str, reason: str
) -> None:
    result = run_checker(tmp_path, body)
    assert result.returncode == 1
    assert reason in result.stdout


def test_checker_rejects_missing_expiry(tmp_path: Path) -> None:
    body = ARCH + POLICY + THREAD + START + stat() + stat(latest_boxes=0)
    result = run_checker(tmp_path, body, "--require-stale-expiry")
    assert result.returncode == 1
    assert "stale_expiry_missing" in result.stdout
