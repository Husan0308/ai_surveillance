from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_compact_global_id_overlay_fixture_in_qt_subprocess(tmp_path):
    """Keep Qt teardown isolated from pytest/plugin worker threads."""
    project_root = Path(__file__).resolve().parents[1]
    renderer = project_root / "scripts" / "render_entrance_global_id_overlay_fixture.py"
    screenshot = tmp_path / "synthetic_global_id_video_overlay.png"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, str(renderer), "--output", str(screenshot)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "UI_FIXTURE_ASSERTIONS=PASS" in result.stdout
    assert "CAM05_COMPACT_PROVISIONAL_LABEL=PASS" in result.stdout
    assert "CANDIDATE_WITHDRAWAL_LABEL_REMOVAL=PASS" in result.stdout
    assert screenshot.is_file() and screenshot.stat().st_size > 10_000
