from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / '.runtime/deepstream-rtsp-pause-suppression-reporting-20261030-r1/sources/apps/sample_apps/deepstream-app/six_live_monitor.c')


def test_production_build_source_has_no_800_second_hard_exit():
    text = SOURCE.read_text()
    assert not re.search(r'\balarm\s*\(\s*800\s*\)', text)
    assert not re.search(r'_exit\s*\(\s*124\s*\)', text)
    assert 'six_live_stop_entered' in text and 'six_live_stop_completed' in text


def test_normal_monitor_command_does_not_enable_gnu_timeout():
    launcher = (ROOT / 'services/gpu/runtime/live_launcher.py').read_text()
    monitor = (ROOT / 'scripts/six_camera_live/monitor.sh').read_text()
    assert 'bounded_monitor = args.monitor and args.duration is not None' in launcher
    assert 'runtime_limit = (None if not bounded_run else' in launcher
    start = monitor.split('exec "$monitor_python"', 1)[1].split('\n    ;;', 1)[0]
    assert '--duration' not in start


def test_explicit_bounded_monitor_still_uses_expected_container_timeout():
    launcher = (ROOT / 'services/gpu/runtime/live_launcher.py').read_text()
    assert 'timeout --foreground --verbose --signal=KILL' in launcher
    assert 'container_kill_deadline = stop_deadline + DAILY_STOP_GRACE_SECONDS' in launcher
    assert 'BOUNDED_SHUTDOWN_RESERVE_SECONDS = 30.0' in launcher


def test_failure_evidence_records_exit_124_without_normalizing_it():
    log = ROOT / '.runtime/six-camera-soak/20261030T222852Z-3e5338cec2/launcher.log'
    text = log.read_text(errors='replace')
    assert 'docker_run_exit_code=124' in text
    assert 'DOCKER_RUN_EXIT code=124' in text
    assert 'OOMKilled": false' in text
    assert 'docker_run_client_killed=False' in text
    assert 'CONTAINER_WATCHDOG pid1=timeout' not in text


def test_soak_elapsed_uses_finished_time_for_completed_run():
    source = (ROOT / 'scripts/six_camera_live/soak_13h.py').read_text()
    assert 'finished_epoch' in source
    assert "state in ('running', 'starting')" in source


def test_rebuilt_candidate_has_no_alarm_import_or_hard_exit_symbol():
    candidate = ROOT / '.runtime/deepstream-alarm800-fix-offline-20261031/sources/apps/sample_apps/deepstream-app/deepstream-app'
    symbols = subprocess.run(['nm', '-D', str(candidate)], capture_output=True, text=True, check=True).stdout
    local = subprocess.run(['nm', str(candidate)], capture_output=True, text=True, check=True).stdout
    assert ' alarm@' not in symbols
    assert ' hard_exit' not in local
    assert ' six_live_prepare' in local and ' six_live_stop' in local

