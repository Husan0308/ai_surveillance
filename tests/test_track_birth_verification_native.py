from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native/test_track_birth_verification.cpp"
NATIVE = ROOT / "services/camera_v11/native/six_camera_v13"


def test_native_track_birth_verification_state_machine(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler, "g++ is required to validate the native birth-verification ledger"
    executable = tmp_path / "test-track-birth-verification"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{NATIVE}", f"-I{ROOT}", str(SOURCE), "-o", str(executable)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)
