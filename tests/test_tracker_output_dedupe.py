from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native/test_tracker_output_dedupe.cpp"
TRACKER = ROOT / "services/camera_v11/native/six_camera_v13/tracker.hpp"


def test_native_tracker_duplicate_output_policy(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler, "g++ is required to validate native tracker output filtering"
    executable = tmp_path / "test-tracker-output-dedupe"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT}", str(SOURCE), "-o", str(executable)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_tracker_output_filter_is_before_canonical_association_and_publication() -> None:
    source = TRACKER.read_text()
    tracker_exit = source.split("static GstPadProbeReturn tracker_exit(", 1)[1]
    assert tracker_exit.index("dedupe_tracker_outputs_by_id_continuity") < tracker_exit.index(
        "// Match detections to active objects"
    )
    assert tracker_exit.index("dedupe_tracker_outputs_by_id_continuity") < tracker_exit.index(
        "tracker_atomic_snapshot"
    )
    assert "duplicate_track_outputs_suppressed" in source
