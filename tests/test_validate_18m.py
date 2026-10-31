import http.server
import importlib.util
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/six_camera_live/validate_18m.py'
spec = importlib.util.spec_from_file_location('validate_18m', HELPER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fake_run(tmp_path, monkeypatch, runid='fixture'):
    runtime = tmp_path / 'runtime'
    run_dir = runtime / runid
    run_dir.mkdir(parents=True, mode=0o700)
    report = run_dir / 'monitor_18m.jsonl'
    report.touch(mode=0o600)
    state = {'run_id': runid, 'report': str(report), 'display': ':fixture',
             'xauthority': '/tmp/fake-xauthority', 'start_mono': time.monotonic()}
    (run_dir / 'state.json').write_text(json.dumps(state))
    os.chmod(run_dir / 'state.json', 0o600)
    monkeypatch.setattr(module, 'RUNTIME', runtime)
    return run_dir, report


def _fake_child(tmp_path, *, graceful=True, exit_code=None):
    child = tmp_path / ('fake_launcher_' + secrets.token_hex(3) + '.py')
    if exit_code is not None:
        source = f'import time\ntime.sleep(0.15)\nraise SystemExit({exit_code})\n'
    elif graceful:
        source = ('import signal,sys,time\n'
                  'signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n'
                  'while True: time.sleep(.05)\n')
    else:
        source = 'import time\nwhile True: time.sleep(.05)\n'
    child.write_text(source)
    return [sys.executable, str(child)]


def _good_health(port, timeout=3):
    del timeout
    if port == 8100:
        body = {'service': 'api_service', 'status': 'ok', 'ml_accessible': True}
    else:
        body = {'service': 'ml_service', 'status': 'ok', 'monitoring_status': 'fresh',
                'analytics_status': 'ready', 'six_camera_readiness': 'ready',
                'shared_pipeline_status': 'healthy', 'cameras': []}
    return module.HealthResult(200, body)


def test_isolated_18m_launcher_explicitly_enables_bounded_stall_diagnostics():
    command = module.launcher_cmd()
    module_index = command.index("services.gpu.runtime.live_launcher")
    assert command[module_index - 2:module_index + 1] == [
        "-B", "-m", "services.gpu.runtime.live_launcher",
    ]
    assert command[-2:] == ["--publish-ml-telemetry", "--source-stall-diagnostics"]


def test_candidate_source_and_binary_are_alarm_free_and_isolated():
    src = ROOT / '.runtime/deepstream-alarm800-fix-offline-20261031/sources/apps/sample_apps/deepstream-app/six_live_monitor.c'
    exe = ROOT / '.runtime/deepstream-alarm800-fix-offline-20261031/sources/apps/sample_apps/deepstream-app/deepstream-app'
    assert 'alarm(800)' not in src.read_text() and '_exit (124)' not in src.read_text()
    assert module.hashlib.sha256(exe.read_bytes()).hexdigest() == module.EXPECTED
    nm = subprocess.run(['nm', '-D', str(exe)], capture_output=True, text=True, check=True).stdout
    assert ' alarm@' not in nm
    assert ' sigaction@' in nm


def test_direct_script_imports_project_with_pythonpath_unset():
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([str(ROOT / '.runtime/full-stack-venv/bin/python'), '-B',
                             str(HELPER), 'import-check'], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'PROJECT_IMPORTS_OK' in result.stdout


def test_transient_worker_imports_project_with_explicit_pythonpath():
    unit = 'ai-surveillance-18m-import-' + secrets.token_hex(4)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([
        'systemd-run', '--user', '--pipe', '--wait', '--collect', f'--unit={unit}',
        '--property=Type=exec', '--property=Restart=no',
        '--property=WorkingDirectory=' + str(ROOT),
        '--setenv=PYTHONPATH=' + str(ROOT),
        str(ROOT / '.runtime/full-stack-venv/bin/python'), '-B', str(HELPER), 'import-check',
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'PROJECT_IMPORTS_OK' in result.stdout


def test_total_deadline_includes_stop_reserve():
    assert module.TOTAL == 1500
    assert module.PRE + module.STOP <= module.TOTAL
    assert module.TOTAL - module.PRE - module.STOP >= 15
    text = HELPER.read_text()
    assert '"--property=Restart=no"' in text
    assert '"--property=KillMode=mixed"' in text
    assert 'RuntimeMaxSec={remaining}s' in text
    assert 'TimeoutStopSec={STOP}s' in text
    assert module.STOP_COMMAND_TIMEOUT + module.STOP_CLEANUP_MARGIN < module.STOP
    assert module.STOP_COMMAND_TIMEOUT + max(1, module.STOP - module.STOP_COMMAND_TIMEOUT - module.STOP_CLEANUP_MARGIN) <= module.STOP


def test_health_helper_uses_one_typed_200_contract():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"status":"ok","ml_accessible":true}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = module.health(server.server_port, timeout=1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert isinstance(result, module.HealthResult)
    assert result.healthy
    assert result.status == 200
    assert result.body['ml_accessible'] is True


def test_health_helper_retains_http_failure_response():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"status":"unavailable"}'
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = module.health(server.server_port, timeout=1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert result.status == 503
    assert result.body == {'status': 'unavailable'}
    assert not result.healthy


def test_health_helper_reports_timeout():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(0.2)
            try:
                self.send_response(200)
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = module.health(server.server_port, timeout=0.03)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert result.status is None
    assert result.error == 'network_timeout'
    assert not result.healthy


def test_health_helper_rejects_malformed_health_json():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'<html>not health json</html>'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = module.health(server.server_port, timeout=1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert result.status == 200
    assert result.body is None
    assert result.error == 'malformed_json'
    assert not result.healthy


def test_preflight_failure_creates_no_report_unit_or_camera_child(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'RUNTIME', tmp_path / 'runtime')
    calls = []
    def fail_preflight():
        raise RuntimeError('api_or_ml_health_unavailable')
    def forbidden_call(*args, **kwargs):
        calls.append(args)
        raise AssertionError('systemd must not be called after failed preflight')
    with pytest.raises(RuntimeError, match='api_or_ml_health_unavailable'):
        module.start_run(preflight_fn=fail_preflight, call_fn=forbidden_call)
    assert calls == []
    assert not module.RUNTIME.exists()


def test_transient_command_explicitly_sets_project_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'RUNTIME', tmp_path / 'runtime')
    calls = []
    def fake_call(args, timeout=8):
        calls.append((list(args), timeout))
        if args[0] == 'systemd-run':
            return subprocess.CompletedProcess(args, 0, '', '')
        return subprocess.CompletedProcess(args, 0, 'active\n', '')
    report = module.start_run(preflight_fn=lambda: {'display': ':fixture', 'xauthority': '/tmp/xauth'},
                              call_fn=fake_call)
    unit_command = calls[0][0]
    assert '--setenv=PYTHONPATH=' + str(ROOT) in unit_command
    assert report.exists()


def test_worker_manual_stop_is_not_validation_pass(tmp_path, monkeypatch):
    run_dir, report = _fake_run(tmp_path, monkeypatch)
    child_script = tmp_path / ('graceful_fake_launcher_' + secrets.token_hex(3) + '.py')
    ready_path = tmp_path / ('fake_launcher_ready_' + secrets.token_hex(3))
    child_script.write_text(
        'import pathlib,signal,sys,time\n'
        'signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n'
        f'pathlib.Path({str(ready_path)!r}).write_text("ready")\n'
        'while True: time.sleep(.05)\n')
    command = [sys.executable, str(child_script)]
    stop_event = threading.Event()
    monkeypatch.setattr(module, 'gpu_snapshot', lambda: {'utilization_percent': 1})
    monkeypatch.setattr(module, 'host_snapshot', lambda: {'cpu_percent': 1})
    def sampled_health(port):
        result = _good_health(port)
        if port == 8001:
            deadline = time.monotonic() + 2
            while not ready_path.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            assert ready_path.exists(), 'fake launcher did not finish its signal setup'
            stop_event.set()
        return result
    child_envs = []
    def launch_child(cmd, **kwargs):
        child_envs.append(kwargs['env'])
        return subprocess.Popen(cmd, **kwargs)
    result = module.worker(
        'fixture', preflight_fn=lambda: {'display': ':fixture', 'xauthority': '/tmp/fake'},
        command_factory=lambda: command, child_factory=launch_child, health_fn=sampled_health,
        session_fn=lambda: (True, ':fixture', '/tmp/fake'),
        cleanup_fn=lambda pid, rc: {'launcher_reaped': rc is not None,
                                    'owned_container_absent': True,
                                    'camera_source_locks_free': True,
                                    'preview_ipc_clear': True,
                                    'telemetry_ipc_clear': True},
        poll_interval=0.01, sample_interval=0.05, stop_event=stop_event)
    rows = _rows(report)
    final = rows[-1]
    assert result == 1
    assert any(row['type'] == 'pipeline_started' for row in rows)
    assert any(row['type'] == 'health' and row['api_http'] == 200 and row['ml_http'] == 200 for row in rows)
    assert final['reason'] == 'manual_or_systemd_stop'
    assert final['launcher_exit_code'] == 0
    assert final['cleanup_ok'] is True
    assert final['success'] is False
    assert child_envs and str(ROOT) in child_envs[0]['PYTHONPATH'].split(os.pathsep)
    assert (run_dir / 'launcher.log').exists()
    assert report.stat().st_mode & 0o777 == 0o600


def test_worker_preflight_failure_spawns_zero_launcher_children(tmp_path, monkeypatch):
    _run_dir, report = _fake_run(tmp_path, monkeypatch)
    spawned = []
    def child_factory(*args, **kwargs):
        spawned.append(args)
        raise AssertionError('camera launcher must not be spawned')
    result = module.worker(
        'fixture', preflight_fn=lambda: (_ for _ in ()).throw(RuntimeError('x11 unavailable')),
        child_factory=child_factory, cleanup_fn=lambda *_: {'no_owned_child': True})
    rows = _rows(report)
    assert result == 1
    assert spawned == []
    assert rows[-1]['reason'] == 'worker_exception'
    assert rows[-1]['exception_category'] == 'RuntimeError'
    assert rows[-1]['cleanup_ok'] is True
    assert rows[-1]['success'] is False


def test_worker_exception_after_spawn_stops_reaps_and_preserves_exit_code(tmp_path, monkeypatch):
    _run_dir, report = _fake_run(tmp_path, monkeypatch)
    command = _fake_child(tmp_path, graceful=False)
    monkeypatch.setattr(module, 'gpu_snapshot', lambda: None)
    monkeypatch.setattr(module, 'host_snapshot', lambda: None)
    def broken_health(_port):
        raise ValueError('synthetic health sampler fault')
    def checks(pid, rc):
        return {'launcher_reaped': rc is not None and not Path(f'/proc/{pid}').exists(),
                'owned_container_absent': True, 'camera_source_locks_free': True,
                'preview_ipc_clear': True, 'telemetry_ipc_clear': True}
    result = module.worker(
        'fixture', preflight_fn=lambda: {'display': ':fixture', 'xauthority': '/tmp/fake'},
        command_factory=lambda: command, health_fn=broken_health,
        session_fn=lambda: (True, ':fixture', '/tmp/fake'), cleanup_fn=checks,
        poll_interval=0.01, install_signal_handlers=False)
    rows = _rows(report)
    final = rows[-1]
    assert result == 1
    assert any(row['type'] == 'pipeline_started' for row in rows)
    assert any(row['type'] == 'worker_exception' and row['category'] == 'ValueError' for row in rows)
    assert final['reason'] == 'worker_exception'
    assert final['exception_category'] == 'ValueError'
    assert final['launcher_exit_code'] == -signal.SIGTERM
    assert final['stop_method'] == 'exact_popen_child_sigterm'
    assert final['cleanup_ok'] is True
    assert final['success'] is False


def test_monitor_child_nonzero_exit_is_failed_and_preserved(tmp_path, monkeypatch):
    _run_dir, report = _fake_run(tmp_path, monkeypatch)
    command = _fake_child(tmp_path, exit_code=23)
    result = module.worker(
        'fixture', preflight_fn=lambda: {'display': ':fixture', 'xauthority': '/tmp/fake'},
        command_factory=lambda: command, health_fn=_good_health,
        session_fn=lambda: (True, ':fixture', '/tmp/fake'),
        cleanup_fn=lambda pid, rc: {'launcher_reaped': rc == 23, 'owned_container_absent': True},
        poll_interval=0.01, sample_interval=0.02, install_signal_handlers=False)
    final = _rows(report)[-1]
    assert result == 1
    assert final['reason'] == 'launcher_failure'
    assert final['termination_source'] == 'launcher_nonzero_exit'
    assert final['launcher_exit_code'] == 23
    assert final['success'] is False




def test_planned_native_observation_boundary_and_budget(tmp_path):
    timeline = module.NativeTimeline(tmp_path / "unused.log")

    required = module.REQUIRED_NATIVE_OBSERVATION_SECONDS
    assert required == 1080
    assert module.PRE >= (
        required + 2 * module.PREFLIGHT_BUDGET + 120 + 15
    )
    assert module.TOTAL >= module.PRE + module.STOP + 15

    start = 1_000_000_000_000
    playing = start + 1_000_000_000

    # Missing native markers: fail closed.
    assert not timeline.planned_observation_complete(
        playing + required * 1_000_000_000
    )

    timeline.native_start_ns = start
    timeline.playing_ns = playing

    assert not timeline.planned_observation_complete(
        playing + required * 1_000_000_000 - 1
    )

    assert timeline.planned_observation_complete(
        playing + required * 1_000_000_000
    )

    # Invalid marker ordering: fail closed.
    timeline.native_start_ns = playing + 1
    assert not timeline.planned_observation_complete(
        playing + required * 1_000_000_000
    )


def test_native_observation_must_reach_required_duration(tmp_path):
    timeline = module.NativeTimeline(tmp_path / "missing.log")
    begin = 1_000_000_000_000
    required = module.REQUIRED_NATIVE_OBSERVATION_SECONDS

    timeline.playing_ns = begin

    # No shutdown marker: cannot establish valid observation.
    assert timeline.native_observation_seconds() is None
    assert timeline.observation_met(required) is False

    # One second short must fail.
    timeline.shutdown_requested_ns = begin + (required - 1) * 1_000_000_000
    assert timeline.observation_met(required) is False

    # Exact threshold qualifies.
    timeline.shutdown_requested_ns = begin + required * 1_000_000_000
    assert timeline.observation_met(required) is True

    # Invalid timestamp ordering must fail closed.
    timeline.shutdown_requested_ns = begin - 1
    assert timeline.native_observation_seconds() is None
    assert timeline.observation_met(required) is False


def test_native_timeline_records_playing_800_null_and_distinct_exit_codes(tmp_path):
    log = tmp_path / 'launcher.log'
    start_ns = 1_000_000_000_000
    exit_ns = start_ns + 801_250_000_000
    log.write_text('\n'.join([
        f'SIX_PIPELINE {{"status":"starting","monotonic_ns":{start_ns}}}',
        f'SIX_PIPELINE {{"status":"playing","monotonic_ns":{start_ns + 1_000_000_000}}}',
        f'SIX_PIPELINE {{"status":"playing","monotonic_ns":{start_ns + 800_500_000_000}}}',
        'LIVE WINDOW COMPLETE; using reserved shutdown budget.',
        'DOCKER_SIGNAL event=stop_request_started signal=SIGINT monotonic_ns=1800000000000',
        'SIX_SHUTDOWN event=sigint_received signal=2',
        'SIX_SHUTDOWN event=pipeline_null_observed set_result=1 query_result=1 current_state=NULL pending_state=VOID_PENDING',
        f'SIX_SHUTDOWN event=process_exit_initiated code=0 monotonic_ns={exit_ns}',
        'DOCKER_RUN_EXIT code=0',
        'DOCKER_STATE {"Status":"exited","Running":false,"OOMKilled":false,"ExitCode":0,"Dead":false}',
        'CLEANUP_PREDICATES {"cleanup_ok":true}',
    ]) + '\n')
    timeline = module.NativeTimeline(log)
    events = timeline.poll()
    names = [event['milestone'] for event in events]
    assert names == [
        'native_viewer_started', 'viewer_reached_playing',
        'native_800_second_boundary_crossed', 'shutdown_initiated',
        'native_sigint_received', 'graceful_null_state_verified',
        'native_process_exit_initiated', 'docker_run_exited',
        'container_exited', 'exact_owned_cleanup_verified',
    ]
    assert timeline.null_state_completed is True
    assert timeline.native_pipeline_uptime_seconds() == 801.25
    assert timeline.docker_run_exit_code == 0
    assert timeline.container_exit_code == 0
    assert timeline.cleanup_ok is True
    assert timeline.final_reason(0, None) == ('deadline', 'planned_live_window_completion')
    early = module.NativeTimeline(tmp_path / 'no_planned_end.log')
    assert early.final_reason(0, None) == ('monitor_exit', 'zero_exit_before_planned_completion')


def test_null_state_requires_a_successful_native_state_query(tmp_path):
    log = tmp_path / 'launcher.log'
    log.write_text('SIX_SHUTDOWN event=pipeline_null_observed set_result=0 query_result=0 '
                   'current_state=PLAYING pending_state=NULL\n')
    timeline = module.NativeTimeline(log)
    events = timeline.poll()
    assert timeline.null_state_completed is False
    assert [event['milestone'] for event in events] == ['null_state_marker_incomplete']


def test_exit_124_is_preserved_as_abnormal_even_if_cleanup_succeeds(tmp_path):
    log = tmp_path / 'launcher.log'
    log.write_text('DOCKER_RUN_EXIT code=124\n'
                   'DOCKER_STATE {"Status":"exited","Running":false,"OOMKilled":false,'
                   '"ExitCode":124,"Dead":false}\n'
                   'CLEANUP_PREDICATES {"cleanup_ok":true}\n')
    timeline = module.NativeTimeline(log)
    timeline.poll()
    assert timeline.docker_run_exit_code == 124
    assert timeline.container_exit_code == 124
    assert timeline.cleanup_ok is True
    assert timeline.final_reason(124, None) == ('launcher_failure', 'launcher_nonzero_exit')


def test_failure_source_summary_uses_last_pre_shutdown_sample_only():
    timeline = module.NativeTimeline(Path('/unused'))
    timeline.shutdown_requested_ns = 1_000
    def message(sid, event, decode, tracker):
        sample = {'source_id': sid, 'camera_id': f'CAM-{sid + 1:02d}',
                  'monotonic_ns': event, 'stages': [
                      {'count': decode, 'last_ns': event}, {}, {},
                      {'count': tracker, 'last_ns': event}]}
        return 'SIX_PROGRESS ' + json.dumps(sample)
    summary = timeline._summarize_source_progress([
        message(0, 900, 10, 10), message(0, 1_000, 10, 10),
        message(0, 1_100, 20, 20),  # After stop request: must not mask the stall.
        message(1, 900, 10, 10), message(1, 1_000, 20, 20),
    ])
    by_id = {row['source_id']: row for row in summary}
    assert by_id[0]['decode_delta_since_previous_sample'] == 0
    assert by_id[0]['tracker_delta_since_previous_sample'] == 0
    assert by_id[1]['decode_delta_since_previous_sample'] == 10
    assert by_id[1]['tracker_delta_since_previous_sample'] == 10
    assert by_id[0]['event_monotonic_ns'] == 1_000


def test_health_records_use_native_monotonic_elapsed_seconds(tmp_path, monkeypatch):
    _run_dir, report = _fake_run(tmp_path, monkeypatch)
    native_child = tmp_path / 'fake_native_launcher.py'
    ready_path = tmp_path / 'native_marker_ready'
    native_child.write_text(
        'import json,pathlib,signal,sys,time\n'
        'print("SIX_PIPELINE " + json.dumps({"status":"starting", "monotonic_ns":time.monotonic_ns()}), flush=True)\n'
        'print("SIX_PIPELINE " + json.dumps({"status":"playing", "monotonic_ns":time.monotonic_ns()}), flush=True)\n'
        f'pathlib.Path({str(ready_path)!r}).write_text("ready")\n'
        'signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n'
        'while True: time.sleep(.02)\n')
    stop_event = threading.Event()
    monkeypatch.setattr(module, 'gpu_snapshot', lambda: None)
    monkeypatch.setattr(module, 'host_snapshot', lambda: None)
    def sampled_health(port):
        deadline = time.monotonic() + 1
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(.002)
        if port == 8001:
            stop_event.set()
        return _good_health(port)
    result = module.worker(
        'fixture', preflight_fn=lambda: {'display': ':fixture', 'xauthority': '/tmp/fake'},
        command_factory=lambda: [sys.executable, str(native_child)], health_fn=sampled_health,
        session_fn=lambda: (True, ':fixture', '/tmp/fake'),
        cleanup_fn=lambda _pid, rc: {'launcher_reaped': rc == 0},
        poll_interval=.01, sample_interval=.02, stop_event=stop_event)
    rows = _rows(report)
    health = next(row for row in rows if row.get('type') == 'health')
    assert result == 1, rows
    assert health['elapsed_clock'] == 'CLOCK_MONOTONIC'
    assert isinstance(health['elapsed_seconds'], float)
    assert health['elapsed_seconds'] >= 0
    assert isinstance(health['native_viewer_elapsed_seconds'], float)
    assert health['native_viewer_elapsed_seconds'] >= 0
    final = rows[-1]
    assert final['native_viewer_started_monotonic_ns'] is not None
    assert final['reason'] == 'manual_or_systemd_stop'
    assert final['success'] is False
    assert final['cleanup_ok'] is True


def test_nonzero_launcher_failure_diagnostic_is_not_misclassified_as_monitor_exit(tmp_path, monkeypatch):
    _run_dir, report = _fake_run(tmp_path, monkeypatch)
    private_runtime = tmp_path / 'project' / '.runtime'
    private_runtime.mkdir(parents=True)
    monkeypatch.setattr(module, 'ROOT', tmp_path / 'project')
    diagnostic = private_runtime / 'failure.json'
    diagnostic.write_text(json.dumps({
        'failure_stage': 'observation_window',
        'failure': 'RuntimeError: one or more DeepStream source FPS streams stalled or became unmeasurable',
        'exit_status': {'container_state_before_cleanup': {'Status': 'exited', 'Running': False,
                                                            'ExitCode': 0, 'OOMKilled': False}},
    }))
    child = tmp_path / 'fake_failure_launcher.py'
    child.write_text(
        'import json,time\n'
        'start=time.monotonic_ns()\n'
        'print("SIX_PIPELINE " + json.dumps({"status":"starting", "monotonic_ns":start}), flush=True)\n'
        'print("SIX_PIPELINE " + json.dumps({"status":"playing", "monotonic_ns":start+1_000_000_000}), flush=True)\n'
        'print("DOCKER_SIGNAL event=stop_request_started signal=SIGINT monotonic_ns=" + str(start+2_000_000_000), flush=True)\n'
        'print("SIX_SHUTDOWN event=pipeline_null_observed set_result=1 query_result=1 current_state=NULL pending_state=VOID_PENDING", flush=True)\n'
        'print("SIX_SHUTDOWN event=process_exit_initiated code=0 monotonic_ns=" + str(start+3_000_000_000), flush=True)\n'
        'print("DOCKER_RUN_EXIT code=0", flush=True)\n'
        'print("DOCKER_STATE {\\"Status\\":\\"exited\\",\\"Running\\":false,\\"OOMKilled\\":false,\\"ExitCode\\":0,\\"Dead\\":false}", flush=True)\n'
        'print("CLEANUP_PREDICATES {\\"cleanup_ok\\":true}", flush=True)\n'
        f'print("LIVE FAILURE DIAGNOSTICS: {diagnostic}", flush=True)\n'
        'raise SystemExit(2)\n')
    monkeypatch.setattr(module, 'gpu_snapshot', lambda: None)
    monkeypatch.setattr(module, 'host_snapshot', lambda: None)
    result = module.worker(
        'fixture', preflight_fn=lambda: {'display': ':fixture', 'xauthority': '/tmp/fake'},
        command_factory=lambda: [sys.executable, str(child)], health_fn=_good_health,
        session_fn=lambda: (True, ':fixture', '/tmp/fake'),
        cleanup_fn=lambda _pid, _rc: {'launcher_reaped': True, 'owned_container_absent': True},
        poll_interval=.01, sample_interval=60, install_signal_handlers=False)
    final = _rows(report)[-1]
    assert result == 1
    assert final['reason'] == 'source_progress_failure'
    assert final['termination_source'] == 'launcher_nonzero_exit'
    assert final['launcher_exit_code'] == 2
    assert final['docker_run_exit_code'] == 0
    assert final['container_exit_code'] == 0
    assert final['null_state_completed'] is True
    assert final['launcher_failure_stage'] == 'observation_window'
    assert 'source FPS streams stalled' in final['launcher_failure']
    assert final['native_pipeline_uptime_seconds'] == 3.0
    assert final['crossed_800_seconds'] is False
    assert final['cleanup_ok'] is True
    assert final['success'] is False


def test_owned_container_matching_is_exact_and_does_not_include_other_runs():
    names = ('ai-surveillance-six-live-123-abcdef123456',
             'ai-surveillance-six-live-123-abcdef123456-extra',
             'ai-surveillance-six-live-124-abcdef123456',
             'unrelated-deepstream')
    assert module._owned_container_names(123, names) == (names[0],)
    assert module._owned_container_names(999, names) == ()


def test_partial_jsonl_is_skipped_and_fallback_remains_failed(tmp_path, monkeypatch):
    run_dir, report = _fake_run(tmp_path, monkeypatch)
    with report.open('a') as stream:
        stream.write('{"type":"partial')
    state_path = run_dir / 'state.json'
    state = json.loads(state_path.read_text())
    state['child_identity'] = None
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(module, '_cleanup_checks', lambda *_: {'should_not_be_called': True})
    result = module.poststop('fixture')
    rows = _rows(report)
    assert result != 0
    assert any(row['type'] == 'final' and row['reason'] == 'supervisor_exit_fallback'
               and row['success'] is False and row['cleanup_ok'] is False for row in rows)


def test_start_rejects_duplicate_from_preflight_before_systemd(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'RUNTIME', tmp_path / 'runtime')
    calls = []
    def duplicate():
        raise RuntimeError('diagnostic_unit_already_active')
    with pytest.raises(RuntimeError, match='diagnostic_unit_already_active'):
        module.start_run(preflight_fn=duplicate,
                         call_fn=lambda *args, **kwargs: calls.append(args))
    assert calls == []
    assert not module.RUNTIME.exists()


def test_exact_existing_six_camera_container_blocks_preflight(monkeypatch):
    monkeypatch.setattr(module, 'session', lambda timeout=5: (True, ':fixture', '/tmp/xauth'))
    monkeypatch.setattr(module, '_container_names', lambda timeout=7: ('ai-surveillance-six-live-88-abcdef123456',))
    monkeypatch.setattr(module, 'gpu_snapshot', lambda timeout=5: {'vram_free_mib': 9000})
    monkeypatch.setattr(module, '_camera_locks_free', lambda: True)
    monkeypatch.setattr(module, '_monitor_lock_clear', lambda: True)
    monkeypatch.setattr(module, '_ipc_clear', lambda: {'preview_slot': True, 'telemetry_lock': True})
    monkeypatch.setattr(module, '_rtsp_rows', lambda timeout=4: ())
    monkeypatch.setattr(module, 'health', _good_health)
    monkeypatch.setattr(module, '_active_unit', lambda: pytest.fail('unit check must not run before orphan check'))
    with pytest.raises(RuntimeError, match='existing_orphan_six_camera_container'):
        module.preflight()


def test_health_preflight_precedes_duplicate_owner_check(monkeypatch):
    observed = []
    monkeypatch.setattr(module, 'session', lambda timeout=5: (observed.append('x11') or True, ':fixture', '/tmp/xauth'))
    monkeypatch.setattr(module, '_container_names', lambda timeout=7: (observed.append('docker') or ()))
    monkeypatch.setattr(module, 'gpu_snapshot', lambda timeout=5: (observed.append('gpu') or {'vram_free_mib': 9000}))
    monkeypatch.setattr(module, '_camera_locks_free', lambda: (observed.append('camera_locks') or True))
    monkeypatch.setattr(module, '_monitor_lock_clear', lambda: (observed.append('owner_lock') or True))
    monkeypatch.setattr(module, '_ipc_clear', lambda: (observed.append('ipc') or {'preview_slot': True}))
    monkeypatch.setattr(module, '_rtsp_rows', lambda timeout=4: (observed.append('rtsp') or ()))
    def health(port, timeout=3):
        observed.append(f'health_{port}')
        return _good_health(port)
    monkeypatch.setattr(module, 'health', health)
    monkeypatch.setattr(module, '_active_unit', lambda timeout=4: (observed.append('unit') or False))
    monkeypatch.setattr(module, 'call', lambda args, timeout=8: subprocess.CompletedProcess(args, 0, '', ''))
    module.preflight()
    assert observed.index('health_8100') < observed.index('owner_lock') < observed.index('unit')


def test_real_transient_unit_rejects_duplicate_name_and_is_stopped():
    unit = 'ai-surveillance-18m-duplicate-' + secrets.token_hex(4)
    first = subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}',
                            '--property=Type=exec', '--property=Restart=no',
                            '/usr/bin/sleep', '30'], capture_output=True, text=True, timeout=10)
    assert first.returncode == 0, first.stderr
    try:
        duplicate = subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}',
                                    '--property=Type=exec', '--property=Restart=no',
                                    '/usr/bin/sleep', '30'], capture_output=True, text=True, timeout=10)
        assert duplicate.returncode != 0
    finally:
        stopped = subprocess.run(['systemctl', '--user', 'stop', unit],
                                 capture_output=True, text=True, timeout=8)
        assert stopped.returncode == 0, stopped.stderr


def test_real_transient_service_runtime_deadline_and_cleanup(tmp_path):
    unit = 'ai-surveillance-18m-fixture-' + secrets.token_hex(4)
    run = subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}',
                          '--property=Type=exec', '--property=Restart=no', '--property=KillMode=mixed',
                          '--property=RuntimeMaxSec=2s', '--property=TimeoutStopSec=2s',
                          '/usr/bin/sleep', '30'], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    (tmp_path / 'systemd-run.txt').write_text(run.stdout)
    deadline = time.monotonic() + 8
    state = 'active'
    props = {}
    while time.monotonic() < deadline:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
                                 '--property=ActiveState', '--property=Result', '--property=MainPID'],
                                capture_output=True, text=True, timeout=4)
        assert result.returncode == 0
        (tmp_path / 'systemd-status.txt').write_text(result.stdout)
        props = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        state = props.get('ActiveState')
        if state == 'inactive':
            break
        time.sleep(.1)
    assert state == 'inactive', result.stdout
    assert props.get('MainPID') in ('0', '')
    (tmp_path / 'systemd-result.txt').write_text(props.get('Result', ''))


def test_real_transient_service_manual_stop_and_abnormal_exit(tmp_path):
    unit = 'ai-surveillance-18m-manual-' + secrets.token_hex(4)
    start = subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}',
                            '--property=Type=exec', '--property=Restart=no', '--property=KillMode=mixed',
                            '--property=TimeoutStopSec=3s', '/usr/bin/sleep', '30'],
                           capture_output=True, text=True, timeout=10)
    assert start.returncode == 0, start.stderr
    time.sleep(.2)
    stopped = subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True,
                             text=True, timeout=8)
    assert stopped.returncode == 0, stopped.stderr
    state = subprocess.run(['systemctl', '--user', 'show', unit,
                             '--property=ActiveState', '--value'], capture_output=True,
                            text=True, timeout=5)
    assert state.stdout.strip() == 'inactive'
    (tmp_path / 'manual-stop.txt').write_text(state.stdout)

    failed_unit = 'ai-surveillance-18m-crash-' + secrets.token_hex(4)
    bad = subprocess.run(['systemd-run', '--user', '--wait', '--collect', f'--unit={failed_unit}',
                          '--property=Type=exec', '--property=Restart=no', '/bin/sh', '-c', 'exit 42'],
                         capture_output=True, text=True, timeout=10)
    assert bad.returncode == 42, f'expected child exit 42, got {bad.returncode}: {bad.stderr} {bad.stdout}'
    st = subprocess.run(['systemctl', '--user', 'is-active', failed_unit],
                        capture_output=True, text=True, timeout=4)
    assert st.returncode != 0
    (tmp_path / 'abnormal-result.txt').write_text(f'exit={bad.returncode}; unit={st.stdout.strip()}')


def test_planned_observation_full_shutdown_integration(tmp_path, monkeypatch):
    """Exercise planned observation, stop, NULL evidence and acceptance."""
    _run_dir, report = _fake_run(tmp_path, monkeypatch)

    # Accelerate only the test; production stays at 1080 seconds.
    monkeypatch.setattr(
        module, "REQUIRED_NATIVE_OBSERVATION_SECONDS", 0.3
    )
    monkeypatch.setattr(module, "gpu_snapshot", lambda: None)
    monkeypatch.setattr(module, "host_snapshot", lambda: None)

    child = tmp_path / "fake_planned_launcher.py"
    child.write_text(
        "import json, signal, sys, time\n"
        "def mark(status):\n"
        "    print('SIX_PIPELINE ' + json.dumps({"
        "'status': status, 'monotonic_ns': time.monotonic_ns()"
        "}), flush=True)\n"
        "def stop(*_):\n"
        "    print('DOCKER_SIGNAL event=stop_request_started "
        "signal=SIGTERM monotonic_ns=%d' % time.monotonic_ns(), "
        "flush=True)\n"
        "    print('SIX_SHUTDOWN event=pipeline_null_observed "
        "set_result=1 query_result=1 current_state=NULL "
        "pending_state=VOID_PENDING', flush=True)\n"
        "    mark('stopped')\n"
        "    print('SIX_SHUTDOWN event=process_exit_initiated "
        "code=0 monotonic_ns=%d' % time.monotonic_ns(), flush=True)\n"
        "    print('DOCKER_RUN_EXIT code=0', flush=True)\n"
        "    print('DOCKER_STATE {\"Status\":\"exited\","
        "\"Running\":false,\"OOMKilled\":false,"
        "\"ExitCode\":0,\"Dead\":false}', flush=True)\n"
        "    print('CLEANUP_PREDICATES {\"cleanup_ok\":true}', "
        "flush=True)\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "mark('starting')\n"
        "mark('playing')\n"
        "while True: time.sleep(.01)\n"
    )

    result = module.worker(
        "fixture",
        preflight_fn=lambda: {
            "display": ":fixture",
            "xauthority": "/tmp/fake",
        },
        command_factory=lambda: [sys.executable, str(child)],
        health_fn=_good_health,
        session_fn=lambda: (True, ":fixture", "/tmp/fake"),
        cleanup_fn=lambda pid, rc: {
            "launcher_reaped": rc == 0,
            "owned_container_absent": True,
        },
        poll_interval=0.01,
        sample_interval=0.1,
        install_signal_handlers=False,
    )

    rows = _rows(report)
    final = rows[-1]

    assert result == 0, final
    assert final["reason"] == "deadline"
    assert final["termination_source"] == (
        "planned_native_observation_completed"
    )
    assert final["native_observation_met"] is True
    assert final["native_observation_seconds"] >= 0.3
    assert final["null_state_completed"] is True
    assert final["launcher_exit_code"] == 0
    assert final["cleanup_ok"] is True
    assert final["success"] is True
