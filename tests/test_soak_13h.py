"""CPU-only soak lifecycle verification. Never opens RTSP or GPU."""
import importlib.util
import json
import os
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

FILE = pathlib.Path(__file__).resolve().parents[1] / 'scripts/six_camera_live/soak_13h.py'
spec = importlib.util.spec_from_file_location('soak_13h_review', FILE)
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    runtime = root / '.runtime/six-camera-soak'
    runtime.mkdir(parents=True)
    monitor = root / 'scripts/six_camera_live/monitor.sh'
    monitor.parent.mkdir(parents=True)
    monitor.write_text('#!/bin/sh\nexit 0\n')
    viewer = root / 'viewer'
    viewer.write_text('test viewer')
    monkeypatch.setattr(soak, 'ROOT', root)
    monkeypatch.setattr(soak, 'RUNTIME', runtime)
    monkeypatch.setattr(soak, 'STATE', runtime / 'active.json')
    monkeypatch.setattr(soak, 'LOCK', runtime / 'control.lock')
    monkeypatch.setattr(soak, 'MONITOR', monitor)
    monkeypatch.setattr(soak, 'VIEWER', viewer)
    monkeypatch.setattr(soak, 'MONITOR_HASH', soak.digest(monitor))
    monkeypatch.setattr(soak, 'VIEWER_HASH', soak.digest(viewer))
    return root, runtime, monitor


def test_state_roundtrip_and_private_mode(env):
    soak.save_state({'run_id': 'test', 'state': 'starting'})
    assert soak.load_state()['run_id'] == 'test'
    assert soak.STATE.stat().st_mode & 0o777 == 0o600


def test_first_append_creates_report(env):
    report = soak.RUNTIME / 'first.jsonl'
    soak.append(report, {'type': 'start'})
    assert soak.records(report) == [{'type': 'start'}]


def test_truncated_jsonl_tail_survives(env):
    report = soak.RUNTIME / 'report.jsonl'
    soak.append(report, {'type': 'ok'})
    with report.open('ab') as f:
        f.write(b'{"type":"incomplete"')
    assert soak.records(report) == [{'type': 'ok'}]


def test_report_size_enforced(env, monkeypatch):
    report = soak.RUNTIME / 'limit.jsonl'
    monkeypatch.setattr(soak, 'REPORT_MAX_BYTES', 3)
    with pytest.raises(RuntimeError, match='size limit'):
        soak.append(report, {'long': True})


def test_ram_stats_are_numeric():
    info = soak.host_stats()
    assert info is not None
    assert info['ram_total_mib'] > 0


def test_process_identity():
    assert soak.proc_ticks(os.getpid()) is not None
    assert soak.proc_ticks(-1) is None


def test_owner_missing_lock_is_not_verified(env):
    state = {'monitor_pid': os.getpid(), 'monitor_ticks': soak.proc_ticks(os.getpid())}
    assert soak.owner_from_launcher(state) is None


def test_owner_bad_container_identity_rejected(env):
    pid = os.getpid()
    state = {'monitor_pid': pid, 'monitor_ticks': soak.proc_ticks(pid)}
    owner_path = soak.ROOT / '.runtime/six-live-preview.lock'
    owner_path.write_text(json.dumps({'mode': 'daily', 'pid': pid,
                                      'start_ticks': state['monitor_ticks'],
                                      'container_name': 'ai-surveillance-six-live-1-abcdef123456'}))
    assert soak.owner_from_launcher(state) is None


def test_owner_exact_pid_ticks_accepted(env):
    pid = os.getpid()
    state = {'monitor_pid': pid, 'monitor_ticks': soak.proc_ticks(pid)}
    owner_path = soak.ROOT / '.runtime/six-live-preview.lock'
    owner_path.write_text(json.dumps({'mode': 'daily', 'pid': pid,
                                      'start_ticks': state['monitor_ticks'],
                                      'container_name': f'ai-surveillance-six-live-{pid}-abcdef123456'}))
    assert soak.owner_from_launcher(state)['pid'] == pid


def test_docker_inspection_failure_blocks(env, monkeypatch):
    monkeypatch.setattr(soak, 'run_cmd',
                        lambda *a, **kw: SimpleNamespace(returncode=1, stdout='', stderr='denied'))
    with pytest.raises(RuntimeError, match='Docker inspection failed'):
        soak.docker_names()


def test_transient_start_arguments_and_report(env, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(soak, 'systemd_state', lambda: 'inactive')
    monkeypatch.setattr(soak, 'preflight', lambda: (':0', '/tmp/test-Xauthority'))
    def fake_run(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='Running as unit', stderr='')
    monkeypatch.setattr(soak, 'run_cmd', fake_run)
    soak.start()
    state = soak.load_state()
    assert state['state'] == 'starting'
    args = calls[0]
    assert '--property=Restart=no' in args
    assert f'--property=RuntimeMaxSec={soak.SYSTEMD_RUNTIME}s' in args
    assert any('ExecStopPost=' in item and 'post-stop' in item for item in args)
    assert args[-2:] == ['run', state['run_id']]
    assert 'Report:' in capsys.readouterr().out


def test_duplicate_start_refused(env, monkeypatch):
    monkeypatch.setattr(soak, 'systemd_state', lambda: 'active')
    with pytest.raises(RuntimeError, match='already active'):
        soak.start()


def test_unresolved_orphan_state_refused(env, monkeypatch):
    soak.save_state({'run_id': 'old', 'state': 'running'})
    monkeypatch.setattr(soak, 'systemd_state', lambda: 'inactive')
    with pytest.raises(RuntimeError, match='unresolved state'):
        soak.start()


def test_stop_signals_only_supervised_unit(env, monkeypatch):
    soak.save_state({'run_id': 'one', 'state': 'running',
                     'report': str(soak.RUNTIME / 'sample.jsonl')})
    monkeypatch.setattr(soak, 'systemd_state', lambda: 'active')
    calls = []
    monkeypatch.setattr(soak, 'run_cmd',
                        lambda args, **kw: (calls.append(args) or
                                            SimpleNamespace(returncode=0, stdout='', stderr='')))
    soak.stop()
    assert calls == [['systemctl', '--user', 'stop', soak.UNIT]]


def test_supervisor_failure_fallback_preserves_failure(env, monkeypatch):
    report = soak.RUNTIME / 'report.jsonl'
    soak.append(report, {'type': 'start'})
    soak.save_state({'run_id': 'one', 'state': 'running', 'report': str(report)})
    observations = []
    monkeypatch.setattr(soak, 'cleanup',
                        lambda data, **kw: (observations.append(kw) or {'ok': True}))
    assert soak.post_stop('one') == 0
    assert observations[0]['fallback']
    assert soak.load_state()['state'] == 'failed'
    assert soak.records(report)[-1]['status'] == 'failed'


def test_clean_post_stop_is_idempotent(env, monkeypatch):
    report = soak.RUNTIME / 'report.jsonl'
    soak.append(report, {'type': 'final', 'status': 'clean'})
    soak.save_state({'run_id': 'one', 'state': 'stopped', 'report': str(report)})
    monkeypatch.setattr(soak, 'cleanup', lambda *a, **kw:
                        pytest.fail('unexpected cleanup'))
    assert soak.post_stop('one') == 0
    assert len(soak.records(report)) == 1


def test_unverified_container_is_never_force_removed(env, monkeypatch):
    pid = os.getpid()
    name = f'ai-surveillance-six-live-{pid}-abcdef123456'
    data = {'monitor_pid': pid, 'monitor_ticks': soak.proc_ticks(pid),
            'container_name': name}
    calls = []
    def fake_run(args, **kw):
        calls.append(args)
        if args[:2] == ['docker', 'ps']:
            return SimpleNamespace(returncode=0, stdout=name, stderr='')
        raise AssertionError(f'unsafe Docker action: {args}')
    monkeypatch.setattr(soak, 'run_cmd', fake_run)
    result = soak.cleanup(data, reason='test')
    assert not result['ok']
    assert result['owned_container_remaining']
    assert all(args[:2] != ['docker', 'rm'] for args in calls)


def test_total_deadline_has_cleanup_reserve():
    assert soak.NORMAL_STOP < soak.SYSTEMD_RUNTIME
    assert soak.SYSTEMD_RUNTIME + soak.SYSTEMD_STOP < soak.LIMIT
    assert soak.SAMPLE_SECONDS == 60


def test_report_reads_single_jsonl(env, capsys):
    report = soak.RUNTIME / 'report.jsonl'
    soak.append(report, {'type': 'start'})
    soak.append(report, {'type': 'health', 'ml': {'http': 200}})
    soak.save_state({'run_id': 'one', 'report': str(report), 'state': 'stopped'})
    soak.report()
    assert json.loads(capsys.readouterr().out)['records'] == 2


def test_status_flags_orphaned_unit(env, monkeypatch, capsys):
    report = soak.RUNTIME / 'report.jsonl'
    soak.append(report, {'type': 'start'})
    soak.save_state({'run_id': 'one', 'state': 'running',
                     'report': str(report), 'started_epoch': 1})
    monkeypatch.setattr(soak, 'systemd_state', lambda: 'inactive')
    soak.status()
    assert json.loads(capsys.readouterr().out)['state'] == 'UNVERIFIED_OR_ORPHANED'


def test_monitor_exit_is_failure(env, monkeypatch):
    report = soak.RUNTIME / 'r.jsonl'
    soak.append(report, {'type': 'start'})
    data = {'run_id': 'one', 'state': 'starting', 'report': str(report),
            'run_dir': str(soak.RUNTIME), 'started_epoch': 1}
    soak.save_state(data)
    class Child:
        pid = os.getpid()
        def poll(self): return 3
        def wait(self, timeout): return 3
    monkeypatch.setattr(soak.subprocess, 'Popen', lambda *a, **kw: Child())
    monkeypatch.setattr(soak, 'snapshot_owner', lambda data: None)
    monkeypatch.setattr(soak, 'cleanup', lambda data, **kw: {'ok': True})
    assert soak.supervise('one') == 1
    assert soak.records(report)[-1]['status'] == 'failed'


def test_disposable_real_systemd_user_deadline(tmp_path):
    """Requires local user manager; does not start six-camera launcher."""
    import secrets
    if not pathlib.Path('/usr/bin/systemd-run').exists():
        pytest.skip('systemd-run unavailable')
    check = subprocess.run(['systemctl', '--user', 'show-environment'], capture_output=True)
    if check.returncode:
        pytest.skip('systemd user manager unavailable in this environment')
    unit = 'soak-cpu-probe-' + secrets.token_hex(4) + '.service'
    marker = tmp_path / 'completed.txt'
    command = ['systemd-run', '--user', '--collect', '--unit=' + unit,
               '--property=Type=exec', '--property=Restart=no',
               '--property=RuntimeMaxSec=2s', '--property=TimeoutStopSec=3s',
               '--property=ExecStopPost=/usr/bin/touch ' + str(marker),
               '/usr/bin/sleep', '30']
    started = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert started.returncode == 0, started.stderr
    try:
        import time
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.1)
        assert marker.exists(), 'real systemd deadline or ExecStopPost failed'
    finally:
        subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=10)


def test_reaped_process_is_not_alive():
    import sys
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    ticks = soak.proc_ticks(child.pid)
    try:
        child.wait(timeout=4)
        assert soak.proc_identity_alive(child.pid, ticks) is False
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
