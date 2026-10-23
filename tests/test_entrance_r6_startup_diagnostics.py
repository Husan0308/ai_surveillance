import io
import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from services.mv3dt_staging.live_tracking.live_bridge import (
    DeepStreamProcess,
    StartupTrace,
)


class _Joinable:
    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


def _process_shell(tmp_path, process, trace):
    shell = DeepStreamProcess.__new__(DeepStreamProcess)
    shell.startup_trace = trace
    shell.process = process
    shell._stdout_thread = _Joinable()
    shell._stderr_thread = _Joinable()
    shell._stdout_file = io.StringIO()
    shell._stderr_file = io.StringIO()
    shell.ready = __import__("threading").Event()
    shell.failed = __import__("threading").Event()
    shell.stdout_path = tmp_path / "stdout.log"
    shell.stderr_path = tmp_path / "stderr.log"
    shell.done = threading.Event()
    shell.done_message_seen = threading.Event()
    shell._input_lock = threading.Lock()
    shell._input_end_recorded = False
    return shell


def _trace_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_startup_trace_is_private_jsonl_and_redacts_credentials(tmp_path):
    trace_path = tmp_path / "private" / "startup.jsonl"
    trace = StartupTrace(trace_path)
    trace.record("UNIT_TEST", argument="rtsp://user:secret@camera.invalid/path")

    row = _trace_rows(trace_path)[0]
    assert row["stage"] == "UNIT_TEST"
    assert row["host_pid"] == os.getpid()
    assert row["monotonic_ns"] > 0
    content = trace_path.read_text()
    assert "secret" not in content
    assert trace_path.stat().st_mode & 0o777 == 0o600
    assert trace_path.parent.stat().st_mode & 0o777 == 0o700


def test_ready_wait_reports_process_exit_before_deadline(tmp_path):
    trace_path = tmp_path / "startup.jsonl"
    trace = StartupTrace(trace_path)

    class ExitedProcess:
        pid = 4321
        stdin = None
        returncode = 3

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    shell = _process_shell(tmp_path, ExitedProcess(), trace)
    with pytest.raises(RuntimeError, match="exited with code 3 before READY"):
        shell.wait_ready(timeout_sec=5)

    stages = [row["stage"] for row in _trace_rows(trace_path)]
    assert "READY_WAIT_STARTED" in stages
    assert "PROCESS_EXITED_BEFORE_READY" in stages
    assert "DEEPSTREAM_PROCESS_STOPPED" in stages
    assert "READY_TIMEOUT" not in stages


def test_shutdown_trace_identifies_term_and_kill_target(tmp_path):
    trace_path = tmp_path / "startup.jsonl"
    trace = StartupTrace(trace_path)

    class EscalatingProcess:
        pid = 8765
        stdin = None
        returncode = None

        def __init__(self):
            self.wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls <= 2:
                raise subprocess.TimeoutExpired("docker", timeout)
            return self.returncode

        def terminate(self):
            self.term_sent = True

        def kill(self):
            self.kill_sent = True
            self.returncode = -9

    process = EscalatingProcess()
    shell = _process_shell(tmp_path, process, trace)
    shell.stop(timeout_sec=0.01)

    rows = _trace_rows(trace_path)
    term = next(row for row in rows if row["stage"] == "HOST_SIGTERM_SENT")
    kill = next(row for row in rows if row["stage"] == "HOST_SIGKILL_SENT")
    stopped = next(row for row in rows if row["stage"] == "DEEPSTREAM_PROCESS_STOPPED")
    assert term["target"] == kill["target"] == "docker-cli-process"
    assert term["target_pid"] == kill["target_pid"] == 8765
    assert stopped["exit_code"] == -9
