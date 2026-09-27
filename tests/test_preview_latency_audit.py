import json
import subprocess
import sys
from pathlib import Path


AUDITOR = Path(__file__).resolve().parents[1] / "scripts/audit_six_camera_preview_latency.py"


def run_audit(tmp_path, total_ms, cameras=None):
    rows = []
    for frame in range(240):
        t0 = 1_000_000_000 + frame * 50_000_000
        rows.append({
            "camera_id": "CAM-01", "sequence": frame + 1,
            "t0_decoder_reference_monotonic_ns": t0,
            "t1_decoder_out_monotonic_ns": t0 + 2_000_000,
            "t6_ipc_publish_monotonic_ns": t0 + 4_000_000,
            "t7_ui_receive_monotonic_ns": t0 + 6_000_000,
            "t8_ui_paint_monotonic_ns": t0 + int(total_ms * 1_000_000),
        })
    log = tmp_path / "timing.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    command = [sys.executable, str(AUDITOR), "--log", str(log)]
    if cameras:
        command += ["--cameras", *cameras]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout)


def test_individual_gate_uses_only_explicit_camera(tmp_path):
    code, report = run_audit(tmp_path, 20, ["CAM-01"])
    assert code == 0
    assert report["status"] == "PASS"
    assert list(report["cameras"]) == ["CAM-01"]


def test_default_gate_still_requires_all_six(tmp_path):
    code, report = run_audit(tmp_path, 20)
    assert code == 2
    assert len(report["failures"]) == 5
    assert len(report["cameras"]) == 6


def test_individual_gate_measures_reference_to_actual_paint(tmp_path):
    code, report = run_audit(tmp_path, 40, ["CAM-01"])
    assert code == 2
    assert report["cameras"]["CAM-01"]["decoder_stage"]["p95_ms"] == 2
    assert report["cameras"]["CAM-01"]["decoder_reference_to_ui_paint"]["p95_ms"] == 40
