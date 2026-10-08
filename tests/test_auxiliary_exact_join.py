from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_auxiliary_exact_join_is_bounded_and_exact(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler, "g++ is required to test the native exact auxiliary join"
    executable = tmp_path / "test-auxiliary-exact-join"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT}", str(ROOT / "tests/native/test_auxiliary_exact_join.cpp"),
         "-pthread", "-o", str(executable)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)
