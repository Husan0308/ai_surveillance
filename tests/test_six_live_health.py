"""CPU-only tests for official DeepStream six-source FPS log monitoring."""
import pytest

from services.gpu.runtime.live_health import CAMERA_IDS, SourceFpsMonitor


def perf(values=None):
    values = [20.0] * 6 if values is None else values
    return "**PERF: " + "\t".join(f"{fps:.2f} ({fps:.2f})" for fps in values)


def test_two_complete_samples_enable_readiness():
    mon = SourceFpsMonitor()
    assert not mon.snapshot(now=0)["ready"]
    assert mon.ingest(perf(), now=0)
    assert not mon.snapshot(now=0)["ready"]
    assert mon.ingest(perf(), now=5)
    snap = mon.snapshot(now=6)
    assert snap["ready"]
    assert snap["samples"] == 2
    assert len(snap["camera_fps"]) == 6
    assert tuple(snap["camera_fps"]) == CAMERA_IDS


def test_zero_fps_causes_fail_closed():
    mon = SourceFpsMonitor()
    mon.ingest(perf(), now=0)
    mon.ingest(perf(), now=5)
    assert mon.snapshot(now=6)["ready"]
    values = [20.0] * 6
    values[4] = 0.0
    assert mon.ingest(perf(values), now=10)
    assert not mon.snapshot(now=11)["ready"]
    assert mon.snapshot(now=11)["camera_fps"]["CAM-05"] == 0.0


def test_stale_stats_are_not_live_frame_progress():
    mon = SourceFpsMonitor(max_age_sec=15)
    mon.ingest(perf(), now=0)
    mon.ingest(perf(), now=5)
    assert mon.snapshot(now=20)["ready"]
    assert not mon.snapshot(now=20.01)["ready"]
    assert not mon.snapshot(now=20.01)["fresh"]


def test_bad_or_partial_lines_do_not_advance_counter():
    mon = SourceFpsMonitor()
    bad = [
        "**PERF: FPS 0 (Avg)\t FPS 1 (Avg)",
        "**PERF: 20 (20)\t20 (20)",
        "INFO: " + perf(),
        "**PERF: 20.0 (20.0) " * 8,
        "PERF(0): 20.0 (20.0)",
        "**PERF: 0 (Avg)\t1 (Avg)\t2 (Avg)",
        "",
    ]
    for line in bad:
        assert not mon.ingest(line, now=0)
    assert mon.snapshot(now=0)["samples"] == 0


def test_stale_gap_resets_good_streak():
    mon = SourceFpsMonitor(max_age_sec=15)
    mon.ingest(perf(), now=0)
    mon.ingest(perf(), now=5)
    assert mon.snapshot(now=5)["ready"]
    mon.ingest(perf(), now=25)
    assert not mon.snapshot(now=25)["ready"]
    mon.ingest(perf(), now=30)
    assert mon.snapshot(now=30)["ready"]


def test_threshold_and_instrumentation_limits_are_explicit():
    with pytest.raises(ValueError):
        SourceFpsMonitor(min_fps=0)
    with pytest.raises(ValueError):
        SourceFpsMonitor(max_age_sec=-1)
    assert "not exact frame PTS" in SourceFpsMonitor().snapshot(now=0)["limited_to"]


def test_live_launcher_uses_monitor_before_and_during_timer():
    from pathlib import Path
    launcher = Path(__file__).resolve().parents[1] / "services/gpu/runtime/live_launcher.py"
    code = launcher.read_text()
    assert "_ask_operator(proc, monitor)" in code
    assert "if not monitor.snapshot()[\"ready\"]:" in code
    assert "monitor = SourceFpsMonitor(" in code
    assert "LIVE SOURCE FPS:" in code
