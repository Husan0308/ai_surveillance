"""Offline contract tests for the isolated Fix 6 upstream progress collector."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / ".runtime/deepstream-fix6-upstream-20261031/host/services/gpu/runtime/live_launcher.py"
SPEC = importlib.util.spec_from_file_location("fix6_live_launcher", CANDIDATE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def progress_line(*, generation: int = 0, sequence: int = 1,
                  upstream: dict | None = None) -> str:
    stages = [
        {"count": sequence, "last_ns": 1000 + sequence, "pts": 2000 + sequence},
        {"count": sequence, "last_ns": 1000 + sequence, "pts": 2000 + sequence},
        {"count": sequence, "last_ns": 1000 + sequence, "pts": 2000 + sequence},
        {"count": sequence, "last_ns": 1000 + sequence, "pts": 2000 + sequence},
    ]
    record = {
        "source_id": 0, "camera_id": "CAM-01", "monotonic_ns": 5000 + sequence,
        "source_generation": generation, "fatal_metadata": False, "stages": stages,
    }
    if upstream is not None:
        record["upstream_diagnostics"] = upstream
    return "SIX_PROGRESS " + json.dumps(record, separators=(",", ":"))


def stage(*, attached: bool = True, valid: bool = True, generation: int = 0,
          buffers: int | None = 0, last_ns: int | None = None) -> dict:
    return {
        "probe_attached": attached, "valid": valid,
        "source_generation": generation, "buffers": buffers,
        "last_progress_monotonic_ns": last_ns,
    }


def upstream(*, generation: int = 0, rtp: dict | None = None,
             compressed: dict | None = None) -> dict:
    return {
        "enabled": True, "sample_monotonic_ns": 9000,
        "rtp_pad": rtp if rtp is not None else stage(generation=generation),
        "compressed_pre_decode": (
            compressed if compressed is not None else stage(generation=generation)
        ),
    }


def test_attached_zero_is_distinct_from_missing_and_retained_bounded() -> None:
    diag = MODULE.ChildOutputDiagnostics(
        source_stall_diagnostics=True,
        upstream_progress_diagnostics=True,
        max_source_progress_samples=2,
    )
    diag.ingest(progress_line(sequence=1, upstream=upstream()))
    diag.ingest(progress_line(sequence=2, upstream=upstream()))
    diag.ingest(progress_line(sequence=3, upstream=upstream()))
    result = diag.snapshot()["source_stall_diagnostics"]
    assert result["progress_samples_observed"] == 3
    assert result["progress_samples_dropped"] == 1
    rows = result["progress_samples"]
    assert [row["decode_count"] for row in rows] == [2, 3]
    rtp = rows[0]["upstream_diagnostics"]["stages"]["rtp_pad"]
    assert rtp["valid"] is True and rtp["probe_attached"] is True
    assert rtp["buffers"] == 0 and rtp["last_progress_monotonic_ns"] is None


def test_missing_probe_is_not_reported_as_zero_progress() -> None:
    missing = stage(attached=False, valid=False, buffers=None)
    diag = MODULE.ChildOutputDiagnostics(
        source_stall_diagnostics=True, upstream_progress_diagnostics=True,
    )
    diag.ingest(progress_line(upstream=upstream(rtp=missing)))
    row = diag.snapshot()["source_stall_diagnostics"]["progress_samples"][0]
    rtp = row["upstream_diagnostics"]["stages"]["rtp_pad"]
    assert rtp["probe_attached"] is False
    assert rtp["valid"] is False
    assert rtp["buffers"] is None


def test_generation_mismatch_fails_closed_without_positive_progress() -> None:
    stale = stage(generation=1, buffers=33, last_ns=8000)
    diag = MODULE.ChildOutputDiagnostics(
        source_stall_diagnostics=True, upstream_progress_diagnostics=True,
    )
    diag.ingest(progress_line(generation=2, upstream=upstream(generation=2, rtp=stale)))
    row = diag.snapshot()["source_stall_diagnostics"]["progress_samples"][0]
    rtp = row["upstream_diagnostics"]["stages"]["rtp_pad"]
    assert rtp["valid"] is False
    assert rtp["source_generation"] is None
    assert rtp["buffers"] is None
    assert rtp["last_progress_monotonic_ns"] is None


def test_old_progress_records_remain_compatible() -> None:
    diag = MODULE.ChildOutputDiagnostics(source_stall_diagnostics=True)
    diag.ingest(progress_line())
    row = diag.snapshot()["source_stall_diagnostics"]["progress_samples"][0]
    assert row["decode_count"] == 1
    assert "upstream_diagnostics" not in row


def test_upstream_payload_is_ignored_when_diagnostics_are_off() -> None:
    diag = MODULE.ChildOutputDiagnostics(source_stall_diagnostics=True)
    diag.ingest(progress_line(upstream=upstream(rtp=stage(buffers=44, last_ns=9000))))
    row = diag.snapshot()["source_stall_diagnostics"]["progress_samples"][0]
    assert "upstream_diagnostics" not in row
    assert diag.snapshot()["source_stall_diagnostics"]["upstream_progress_enabled"] is False


def test_malformed_or_missing_upstream_records_fail_closed() -> None:
    diag = MODULE.ChildOutputDiagnostics(
        source_stall_diagnostics=True, upstream_progress_diagnostics=True,
    )
    diag.ingest(progress_line(sequence=1, upstream={"enabled": True}))
    diag.ingest(progress_line(sequence=2, upstream={
        "enabled": True, "sample_monotonic_ns": "not-a-time",
        "rtp_pad": stage(buffers=99, last_ns=100),
    }))
    first, second = diag.snapshot()["source_stall_diagnostics"]["progress_samples"]
    assert first["upstream_diagnostics"]["stages"]["rtp_pad"]["buffers"] is None
    assert first["upstream_diagnostics"]["stages"]["rtp_pad"]["valid"] is False
    assert second["upstream_diagnostics"]["enabled"] is False
    assert second["upstream_diagnostics"]["stages"]["rtp_pad"]["buffers"] is None

    diag.ingest(progress_line(sequence=3, upstream=upstream(
        rtp=stage(buffers=5, last_ns=None))))
    third = diag.snapshot()["source_stall_diagnostics"]["progress_samples"][-1]
    assert third["upstream_diagnostics"]["stages"]["rtp_pad"]["valid"] is False
    assert third["upstream_diagnostics"]["stages"]["rtp_pad"]["buffers"] is None


def test_combined_stage_records_distinguish_upstream_failure_boundaries() -> None:
    cases = [
        # RTP, post-parser compressed, decoded, tracker
        (0, 0, 0, 0),  # upstream input stalled
        (10, 0, 0, 0),  # RTP delivered, depay/parser output stopped
        (10, 8, 0, 0),  # compressed stage delivered, decoder output stopped
        (10, 8, 6, 0),  # decoded output advanced, tracker stopped
    ]
    observed = []
    for index, (rtp_count, compressed_count, decoded_count, tracker_count) in enumerate(cases):
        diag = MODULE.ChildOutputDiagnostics(
            source_stall_diagnostics=True, upstream_progress_diagnostics=True,
        )
        body = json.loads(progress_line(sequence=index + 1).partition(" ")[2])
        body["stages"][0]["count"] = decoded_count
        body["stages"][3]["count"] = tracker_count
        body["upstream_diagnostics"] = upstream(
            rtp=stage(buffers=rtp_count, last_ns=10 if rtp_count else None),
            compressed=stage(buffers=compressed_count,
                             last_ns=20 if compressed_count else None),
        )
        diag.ingest("SIX_PROGRESS " + json.dumps(body))
        row = diag.snapshot()["source_stall_diagnostics"]["progress_samples"][0]
        stages = row["upstream_diagnostics"]["stages"]
        observed.append((stages["rtp_pad"]["buffers"],
                         stages["compressed_pre_decode"]["buffers"],
                         row["decode_count"], row["tracker_count"]))
    assert observed == cases


def test_container_opt_in_is_explicit_and_default_off() -> None:
    normal = MODULE._docker_args(
        "image", Path("/models"), "bridge", None, None, "normal",
        mosaic_viewer_binary=Path("/viewer"), mosaic_slot_path=Path("/tmp/slot"),
    )
    diagnostic = MODULE._docker_args(
        "image", Path("/models"), "bridge", None, None, "diagnostic",
        mosaic_viewer_binary=Path("/viewer"), mosaic_slot_path=Path("/tmp/slot"),
        upstream_progress_diagnostics=True,
    )
    assert "SIX_LIVE_UPSTREAM_DIAGNOSTICS=1" not in normal
    assert "SIX_LIVE_UPSTREAM_DIAGNOSTICS=1" in diagnostic
