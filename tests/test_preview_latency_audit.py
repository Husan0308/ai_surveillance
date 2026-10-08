import json
import sys

import pytest

from scripts import audit_six_camera_preview_latency as auditor


def run_audit(tmp_path, monkeypatch, capsys, total_ms=20, frames=240, invalid=False):
    rows = []
    for camera in auditor.CAMERAS:
        for frame in range(frames):
            t0 = 1_000_000_000 + frame * 50_000_000
            rows.append({
                "camera_id": camera,
                "sequence": frame + 1,
                "t0_decoder_reference_monotonic_ns": t0,
                "t1_decoder_out_monotonic_ns": t0 + 2_000_000,
                "t2_preview_prepare_complete_monotonic_ns": t0 + 3_000_000,
                "t3_tracker_metadata_ready_monotonic_ns": t0 + 3_500_000,
                "t4_shared_publish_monotonic_ns": t0 + 4_000_000,
                "t6_ipc_publish_monotonic_ns": t0 + 4_000_000,
                "t7a_reader_poll_begin_monotonic_ns": t0 + 5_000_000,
                "t7b_reader_return_monotonic_ns": t0 + 5_500_000,
                "t7_ui_receive_monotonic_ns": t0 + 6_000_000,
                "t8_ui_paint_monotonic_ns": t0 + int(total_ms * 1_000_000),
                "t6_ui_paint_monotonic_ns": t0 + int(total_ms * 1_000_000),
            })
    if invalid:
        rows[0]["t1_decoder_out_monotonic_ns"] = rows[0]["t0_decoder_reference_monotonic_ns"] - 1
    path = tmp_path / "timing.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", ["audit", "--log", str(path)])
    code = auditor.main()
    return code, json.loads(capsys.readouterr().out)


def test_all_six_actual_paint_timings_pass(tmp_path, monkeypatch, capsys):
    code, report = run_audit(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert report["status"] == "PASS"
    assert set(report["cameras"]) == set(auditor.CAMERAS)
    assert report["failures"] == []
    camera = report["cameras"]["CAM-01"]
    assert camera["decoder_reference_to_preview_prepared"]["p99_ms"] == 3
    assert camera["decoder_reference_to_tracker_metadata"]["p99_ms"] == 3.5
    assert camera["max_preview_or_tracker_to_shared_publish"]["p99_ms"] == 0.5
    assert camera["shared_publish_to_ui_paint"]["p99_ms"] == 16


@pytest.mark.parametrize("total_ms", [40, 41])
def test_40ms_limit_is_strict_and_includes_ui_paint(tmp_path, monkeypatch, capsys, total_ms):
    code, report = run_audit(tmp_path, monkeypatch, capsys, total_ms=total_ms)
    assert code == 2
    assert len(report["failures"]) == 6
    camera = report["cameras"]["CAM-01"]
    assert camera["decoder_stage"]["p95_ms"] == 2
    assert camera["decoder_reference_to_ui_paint"]["p95_ms"] == total_ms


def test_insufficient_samples_cannot_pass(tmp_path, monkeypatch, capsys):
    code, report = run_audit(tmp_path, monkeypatch, capsys, frames=199)
    assert code == 2
    assert len(report["failures"]) == 6
    assert all("insufficient timing samples" in item for item in report["failures"])


def test_invalid_timestamp_order_cannot_be_silently_excluded(tmp_path, monkeypatch, capsys):
    code, report = run_audit(tmp_path, monkeypatch, capsys, invalid=True)
    assert code == 2
    assert report["cameras"]["CAM-01"]["invalid_timing_rows"] == 1
    assert report["failures"] == ["CAM-01: invalid timing rows=1"]


def test_reader_poll_may_overlap_publish_when_return_is_serialized(tmp_path, monkeypatch, capsys):
    code, report = run_audit(tmp_path, monkeypatch, capsys)
    assert code == 0

    # A blocking reader can start under the writer's exclusive lock, but must
    # return only after the complete frame+metadata publication is committed.
    path = tmp_path / "timing.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["t7a_reader_poll_begin_monotonic_ns"] = (
        rows[0]["t4_shared_publish_monotonic_ns"] - 100_000
    )
    rows[0]["t7b_reader_return_monotonic_ns"] = (
        rows[0]["t4_shared_publish_monotonic_ns"] + 100_000
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(sys, "argv", ["audit", "--log", str(path)])
    code = auditor.main()
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["status"] == "PASS"
    assert report["cameras"]["CAM-01"]["invalid_timing_rows"] == 0


def test_reader_return_before_publish_is_rejected(tmp_path, monkeypatch, capsys):
    code, report = run_audit(tmp_path, monkeypatch, capsys)
    assert code == 0
    path = tmp_path / "timing.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["t7a_reader_poll_begin_monotonic_ns"] = (
        rows[0]["t4_shared_publish_monotonic_ns"] - 100_000
    )
    rows[0]["t7b_reader_return_monotonic_ns"] = (
        rows[0]["t4_shared_publish_monotonic_ns"] - 1
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(sys, "argv", ["audit", "--log", str(path)])
    code = auditor.main()
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["cameras"]["CAM-01"]["invalid_timing_rows"] == 1
    assert report["failures"] == ["CAM-01: invalid timing rows=1"]
