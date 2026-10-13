from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_camera_recovery_state_machine(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler, "g++ is required to validate native recovery policy"
    executable = tmp_path / "test-camera-recovery-policy"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT}", str(ROOT / "tests/native/test_camera_recovery_policy.cpp"),
         "-o", str(executable)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)
