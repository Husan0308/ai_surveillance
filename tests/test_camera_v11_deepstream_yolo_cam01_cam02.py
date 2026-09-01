from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from services.camera_v11.deepstream_trt86_multi_v1 import (
    DetectorSnapshot,
    MODEL_PAD_TOP,
    map_detector_boxes_to_display,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_v1.py"
LAUNCHER = ROOT / "scripts/run_camera_v11_deepstream_yolo_cam01_cam02_v1.sh"
CHECKER = ROOT / "scripts/check_camera_v11_deepstream_yolo_cam01_cam02_v1_log.py"

ARCH = (
    "CAMERA_V11_DS_YOLO_MULTI_ARCH cameras=2 camera_ids=CAM-01,CAM-02 "
    "rtsp_sources=2 rtsp_sessions=2 rtsp_per_camera=1 decode=deepstream-nvdec "
    "source=nvurisrcbin tee_per_camera=1 display=independent-nvstreammux+nvdsosd "
    "detector=shared-trt86-sidecar detector_workers=1 detector_rtsp=0 "
    "detector_queue=latest1-per-camera detector_thread=dedicated scheduler=round-robin "
    "gst_nvinfer=0 second_rtsp=0 opencv=0 ffmpeg=0 tracker=0 reid=0 face=0 ui=0\n"
)
START = "CAMERA_V11_DS_YOLO_MULTI_START cameras=CAM-01,CAM-02 state=async\n"
THREAD = "CAMERA_V11_DS_YOLO_MULTI_DETECTOR_THREAD state=START workers=1 cameras=2 scheduler=round-robin\n"


def stat(camera: str, **overrides) -> str:
    values = {
        "source_fps": 20.0,
        "render_fps": 18.5,
        "infer_hz": 2.0,
        "queue": 0,
        "infer_count": 20,
        "infer_admitted": 20,
        "detector_drops": 180,
        "positive_inferences": 0,
        "detections_total": 0,
        "max_objects": 0,
        "latest_boxes": 0,
        "result_clears": 0,
        "stale_expirations": 0,
        "metadata_added": 0,
        "result_age_ms": 100.0,
        "infer_p95_ms": 20.0,
        "render_gap_p95_ms": 60.0,
        "detector_thread_alive": 1,
        "worker_alive": 1,
        "copy_errors": 0,
        "infer_errors": 0,
        "meta_errors": 0,
        "warnings": 0,
        "pipeline_errors": 0,
    }
    values.update(overrides)
    return "CAMERA_V11_DS_YOLO_MULTI camera=" + camera + " " + " ".join(
        f"{key}={value}" for key, value in values.items()
    ) + "\n"


def valid_body(rows: int = 5) -> str:
    body = ARCH + THREAD + START
    for _ in range(rows):
        body += stat("CAM-01") + stat("CAM-02")
    return body


def run_checker(tmp_path: Path, body: str, *args: str) -> subprocess.CompletedProcess[str]:
    log = tmp_path / "multi.log"
    log.write_text(body)
    return subprocess.run(
        [sys.executable, str(CHECKER), "--log", str(log), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_padding_mapping_matches_cam01_contract() -> None:
    assert MODEL_PAD_TOP == 3
    mapped = map_detector_boxes_to_display(
        [[0.0, 3.0, 672.0, 381.0, 0.8], [10.0, 0.0, 20.0, 2.0, 0.9]],
        640,
        360,
    )
    assert len(mapped) == 1
    assert mapped[0] == pytest.approx([0.0, 0.0, 640.0, 360.0, 0.8])


def test_snapshot_is_immutable() -> None:
    snapshot = DetectorSnapshot(boxes=((1.0, 2.0, 3.0, 4.0, 0.9),), sequence=1)
    with pytest.raises(Exception):
        snapshot.sequence = 2  # type: ignore[misc]


def test_runtime_uses_one_source_per_camera_and_one_shared_worker() -> None:
    source = RUNTIME.read_text()
    assert 'for state in self.states.values():\n            self._build_camera(state)' in source
    assert source.count('self._make("nvurisrcbin"') == 1
    assert source.count("Step2TRT86Client()") == 1
    assert 'detector=shared-trt86-sidecar' in source
    assert 'detector_workers=1' in source
    assert 'scheduler=round-robin' in source
    assert '"max-size-buffers", 1' in source
    assert '"leaky", 2' in source
    assert '"max-buffers", 1' in source
    assert 'appsink.connect("new-sample"' not in source
    assert "cv2.VideoCapture" not in source
    assert "ffmpeg" in source
    assert "subprocess.Popen" not in source


def test_inference_only_occurs_in_dedicated_detector_loop() -> None:
    source = RUNTIME.read_text()
    detector = source[source.index("    def _detector_loop"):source.index("    def _display_meta_probe")]
    assert "infer_preloaded" in detector
    for callback in ("_source_probe", "_infer_gate_probe", "_display_meta_probe", "_render_probe"):
        section = source[source.index(f"    def {callback}"):]
        section = section.split("\n    def ", 1)[0]
        assert "infer_preloaded" not in section


def test_launcher_is_locked_to_incremental_branch_and_two_cameras() -> None:
    source = LAUNCHER.read_text()
    assert 'BRANCH_EXPECTED="rebuild/service-architecture-v11-deepstream-yolo-cam01-cam02-v1-20260901"' in source
    assert 'CAMERAS="${V11_DS_YOLO_CAMERAS:-CAM-01,CAM-02}"' in source
    assert 'detector_workers=1' in source
    assert 'detector_rtsp=0' in source
    assert 'deepstream_trt86_multi_v1' in source
    assert ".pt" not in source
    assert "gst-nvinfer" not in source


def test_checker_accepts_stable_two_camera_log(tmp_path: Path) -> None:
    result = run_checker(tmp_path, valid_body())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT=PASS" in result.stdout


def test_checker_accepts_later_person_and_expiry_for_one_camera(tmp_path: Path) -> None:
    body = ARCH + THREAD + START
    for index in range(5):
        if index == 4:
            body += stat(
                "CAM-01",
                positive_inferences=2,
                detections_total=2,
                metadata_added=12,
                result_clears=1,
            )
        else:
            body += stat("CAM-01")
        body += stat("CAM-02")
    result = run_checker(
        tmp_path,
        body,
        "--require-person-camera",
        "CAM-01",
        "--require-stale-expiry-camera",
        "CAM-01",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (valid_body().replace("second_rtsp=0", "second_rtsp=1"), "arch_second_rtsp"),
        (valid_body().replace("detector_workers=1", "detector_workers=2", 1), "arch_detector_workers"),
        (valid_body().replace("queue=0", "queue=2", 1), "queue=2>1"),
        (valid_body().replace("infer_errors=0", "infer_errors=1", 1), "infer_errors=1"),
        (valid_body() + "Traceback (most recent call last)\n", "runtime_traceback"),
    ],
)
def test_checker_rejects_architecture_and_runtime_regressions(
    tmp_path: Path, body: str, reason: str
) -> None:
    result = run_checker(tmp_path, body)
    assert result.returncode == 1
    assert reason in result.stdout


def test_checker_rejects_missing_second_camera_stats(tmp_path: Path) -> None:
    body = ARCH + THREAD + START + "".join(stat("CAM-01") for _ in range(5))
    result = run_checker(tmp_path, body)
    assert result.returncode == 1
    assert "CAM-02:stats=0<5" in result.stdout


def test_checker_rejects_unfair_shared_scheduler(tmp_path: Path) -> None:
    body = ARCH + THREAD + START
    for _ in range(5):
        body += stat("CAM-01", infer_hz=2.0) + stat("CAM-02", infer_hz=0.5)
    result = run_checker(tmp_path, body)
    assert result.returncode == 1
    assert "CAM-02:infer_hz_min=0.50<1.20" in result.stdout
