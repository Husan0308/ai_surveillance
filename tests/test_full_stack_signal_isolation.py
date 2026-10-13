"""Regression for the full-stack supervisor's ordered signal shutdown."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import textwrap
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("_cycle", range(3))
def test_terminal_sigint_reaches_supervisor_before_owned_child(_cycle: int) -> None:
    """A terminal Ctrl-C must be forwarded once as the supervisor's TERM."""
    child_code = textwrap.dedent(
        """
        import signal, sys, time
        def record(signum, _frame):
            print("CHILD_SIGNAL", signum, flush=True)
            if signum == signal.SIGTERM:
                raise SystemExit(0)
        signal.signal(signal.SIGINT, record)
        signal.signal(signal.SIGTERM, record)
        print("CHILD_READY", flush=True)
        while True:
            time.sleep(0.05)
        """
    )
    harness = textwrap.dedent(
        f"""
        import os, signal, subprocess, sys, time
        from pathlib import Path
        from scripts.full_stack_runtime import spawn_owned_child
        child = spawn_owned_child(
            [sys.executable, "-u", "-c", {child_code!r}],
            cwd=Path.cwd(), env=os.environ.copy(), stdout=sys.stdout)
        state = {{"running": True}}
        def stop(_signum, _frame):
            state["running"] = False
        signal.signal(signal.SIGINT, stop)
        print("SUPERVISOR_READY", child.pid, os.getpgid(child.pid), flush=True)
        try:
            while state["running"]:
                time.sleep(0.02)
        finally:
            child.terminate()
            child.wait(timeout=3)
            print("SUPERVISOR_DONE", child.returncode, flush=True)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    supervisor = subprocess.Popen(
        [sys.executable, "-u", "-c", harness],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    output: list[str] = []
    child_pid: int | None = None
    try:
        assert supervisor.stdout is not None
        ready_line = ""
        deadline = time.monotonic() + 5
        while "SUPERVISOR_READY" not in "\n".join(output) or "CHILD_READY" not in "\n".join(output):
            timeout = deadline - time.monotonic()
            assert timeout > 0, f"isolated supervisor did not become ready: {output}"
            ready, _, _ = select.select([supervisor.stdout], [], [], timeout)
            assert ready, f"isolated supervisor did not become ready: {output}"
            line = supervisor.stdout.readline().strip()
            output.append(line)
            if line.startswith("SUPERVISOR_READY"):
                ready_line = line
        fields = ready_line.split()
        child_pid = int(fields[1])
        child_pgid = int(fields[2])
        assert child_pgid == child_pid, "owned child shares the terminal signal group"

        os.killpg(supervisor.pid, signal.SIGINT)
        remaining, _ = supervisor.communicate(timeout=5)
        output.append(remaining)
        assert supervisor.returncode == 0
        joined = "\n".join(output)
        assert "CHILD_SIGNAL 15" in joined, joined
        assert "CHILD_SIGNAL 2" not in joined, joined
        assert "SUPERVISOR_DONE 0" in joined, joined
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=3)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_expected_docker_term_exit_requires_complete_native_cleanup(tmp_path: Path) -> None:
    from scripts.full_stack_runtime import verified_analytics_sigterm_shutdown

    expected_cameras = {f"CAM-{index:02d}" for index in range(1, 7)}

    def fixture(name: str, *, code: int, signals: list[int],
                missing: set[str] = frozenset(), fatal: int = 0,
                shared_errors: int = 0) -> Path:
        output = tmp_path / name
        analytics = output / "analytics"
        analytics.mkdir(parents=True)
        (analytics / "exit.json").write_text(json.dumps({
            "return_code": code,
            "signal_received": signals,
            "native_exit_monotonic_ns": 123,
        }))
        ended = [f"{camera} END ingress=100 input=100 hardware_decoder=1 errors=0 warnings=0"
                 for camera in sorted(expected_cameras - missing)]
        (analytics / "native.log").write_text("\n".join([
            *ended,
            f"GROUP END output=100 fatal={fatal} shared_errors={shared_errors} "
            "shared_warnings=0 isolation=not-requested",
        ]))
        return output

    assert verified_analytics_sigterm_shutdown(
        fixture("clean", code=143, signals=[signal.SIGTERM]), 143, signal.SIGTERM)
    assert not verified_analytics_sigterm_shutdown(
        fixture("double-signal", code=130, signals=[signal.SIGINT, signal.SIGTERM]),
        130, signal.SIGTERM)
    assert not verified_analytics_sigterm_shutdown(
        fixture("incomplete", code=143, signals=[signal.SIGTERM], missing={"CAM-06"}),
        143, signal.SIGTERM)
    assert not verified_analytics_sigterm_shutdown(
        fixture("fatal", code=143, signals=[signal.SIGTERM], fatal=1), 143, signal.SIGTERM)
    assert not verified_analytics_sigterm_shutdown(
        fixture("shared-error", code=143, signals=[signal.SIGTERM], shared_errors=1),
        143, signal.SIGTERM)
