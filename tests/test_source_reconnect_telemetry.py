from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native/test_source_reconnect_telemetry.cpp"
MAIN = ROOT / "services/camera_v11/native/six_camera_v13/main.cpp"
RECOVERY_RUNNER = ROOT / "scripts/run_source_stall_recovery_acceptance.py"
SYNTHETIC_RUNNER = ROOT / "scripts/run_verified_candidate_synthetic_six.py"
TRACKER = ROOT / "services/camera_v11/native/six_camera_v13/tracker.hpp"
LOWCONF_CONTRACT = ROOT / "services/camera_v11/native/six_camera_v13/lowconf_contract.hpp"
LOWCONF_VERIFIER = ROOT / "services/camera_v11/native/six_camera_v13/lowconf_verifier.hpp"
FRAME_RING = ROOT / "services/camera_v11/native/six_camera_v13/exact_frame_ring.hpp"


def test_native_rtsp_reconnect_warning_classifier(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler, "g++ is required to validate the native reconnect classifier"
    executable = tmp_path / "test-source-reconnect-telemetry"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT}", str(SOURCE), "-o", str(executable)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_monitoring_exposes_nvurisrcbin_retries_and_watchdog_recoveries_separately() -> None:
    source = MAIN.read_text()
    assert "rtsp_reconnect_attempts" in source
    assert "source_stall_recoveries" in source
    assert "reconnects" in source
    runner = RECOVERY_RUNNER.read_text()
    assert "source_stall_recoveries" in runner


def test_reconnect_epoch_clears_tracker_state_and_rejects_stale_exact_joins() -> None:
    main = MAIN.read_text()
    tracker = TRACKER.read_text()
    lowconf_contract = LOWCONF_CONTRACT.read_text()
    lowconf = LOWCONF_VERIFIER.read_text()
    frame_ring = FRAME_RING.read_text()
    assert "tracking_epoch" in main
    assert "++target->tracking_epoch" in main
    assert '"rtsp-reconnect-interval", 0u' in main
    assert '"init-rtsp-reconnect-interval", 0u' in main
    assert "reconnect_interval != 0 || init_reconnect_interval != 0" in main
    assert "retry_owner=application_watchdog" in main
    assert "SOURCE_RECOVERY_BEGIN" in main
    assert "SOURCE_STALL_RECOVERY_COOLDOWN_US" not in main
    assert "tracker_reset_temporal_state_for_epoch" in tracker
    assert "tracker_birth_verification.clear_source(source_id)" in tracker
    assert "source_epoch" in lowconf_contract
    assert "queue.front().source_epoch == source_epoch" in lowconf
    assert "frame.key.source_epoch != source_tracking_epoch_current(source_id)" in lowconf
    assert "slot.source_epoch != source_epoch" in frame_ring


def test_synthetic_recovery_audit_normalizes_native_event_and_checks_every_cycle() -> None:
    runner = SYNTHETIC_RUNNER.read_text()
    assert 'event_name == "SOURCE_RECOVERY_BEGIN"' in runner
    assert 'event_name = "SOURCE_STALL_RECOVERY_BEGIN"' in runner
    assert "planned_counts =" in runner
    assert "observed_counts == planned_counts" in runner
    assert "all(cycle == expected_cycle" in runner
