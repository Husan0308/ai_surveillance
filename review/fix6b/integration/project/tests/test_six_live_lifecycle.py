"""CPU-only lifecycle tests: all Docker/GPU operations and time are simulated."""
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest

from services.gpu.runtime import live_launcher as live
from services.gpu.runtime.live_config import CAMERAS, CHANNELS, Camera

ASK_OPERATOR = live._ask_operator


@pytest.fixture
def runner(tmp_path, monkeypatch):
    class Clock:
        now = 0.0

        def __init__(self):
            self.timers = []

        def advance(self, seconds):
            target = self.now + seconds
            for timer in list(self.timers):
                if not timer.cancelled and not timer.fired and timer.at <= target:
                    self.now = timer.at
                    timer.fired = True
                    timer.function(*timer.args)
            self.now = target

    clock = Clock()
    state = SimpleNamespace(
        calls=[], launched=[], graceful=True, remove_ok=True, confirm_delay=1,
        ready=lambda: clock.now >= 1.3, block_write=False, exit_at=None, container_present=True,
        container_exit_code=None, graceful_delay_seconds=0.0,
        graceful_exit_at=None, timeout_kill_at=None, timeout_start_offset=1.3,
    )

    class Timer:
        def __init__(self, delay, function, args):
            self.at = clock.now + delay
            self.function, self.args = function, args
            self.cancelled = self.fired = False
            clock.timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    class Thread:
        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

        def join(self, timeout):
            assert timeout <= 0.25

    class Sink:
        def write(self, payload):
            assert payload  # Real tar delivery, with synthetic credentials only.
            if state.block_write:
                watchdog_at = 60 - min(2.0, 60 / 4)
                clock.advance(watchdog_at - clock.now)
                assert state.proc.returncode == 137  # Independent watchdog ran.
                raise BrokenPipeError("synthetic blocked stdin")

    class Process:
        stdout = None
        returncode = None

        def __init__(self, cmd, **kwargs):
            state.launched.append((clock.now, cmd))
            state.launch_options = kwargs
            timeout_match = re.search(r"--signal=KILL ([0-9.]+)s", cmd[-1])
            if timeout_match:
                state.timeout_kill_at = (
                    clock.now + float(timeout_match.group(1)) + state.timeout_start_offset
                )
            self.stdin = SimpleNamespace(buffer=Sink(), close=lambda: None)
            state.proc = self

        def poll(self):
            if (state.graceful_exit_at is not None
                    and clock.now >= state.graceful_exit_at
                    and self.returncode is None):
                self.returncode = 0
                state.container_exit_code = 0
            if (state.timeout_kill_at is not None
                    and clock.now >= state.timeout_kill_at
                    and self.returncode is None):
                self.returncode = 137
                state.container_exit_code = 137
            if state.exit_at is not None and clock.now >= state.exit_at:
                self.returncode = 139
            return self.returncode

        def wait(self, timeout):
            if self.poll() is None:
                events = [event for event in
                          (state.graceful_exit_at, state.timeout_kill_at)
                          if event is not None and clock.now < event <= clock.now + timeout]
                if events:
                    clock.advance(min(events) - clock.now)
                    self.poll()
                    return self.returncode
                clock.advance(timeout)
                raise live.subprocess.TimeoutExpired("synthetic-child", timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    def run(cmd, **kwargs):
        state.calls.append((clock.now, cmd, kwargs.get("timeout")))
        if cmd[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="8000\n", stderr="")
        assert cmd[:2] in (["docker", "kill"], ["docker", "rm"], ["docker", "inspect"], ["docker", "ps"])
        if cmd[1] == "ps":
            name = cmd[cmd.index("--filter") + 1].removeprefix("name=")
            return SimpleNamespace(returncode=0, stderr="", stdout=name + "\n" if state.container_present else "")
        if cmd[1] == "kill" and state.graceful:
            if state.graceful_delay_seconds:
                state.graceful_exit_at = clock.now + state.graceful_delay_seconds
            else:
                state.proc.returncode = 0
                if state.container_exit_code is None:
                    state.container_exit_code = 0
        if cmd[1] == "rm":
            if not state.remove_ok:
                return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")
            if state.proc.poll() is None:
                state.proc.returncode = 137
            state.container_present = False
        if cmd[1] == "inspect" and state.container_exit_code is not None:
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "Running": False, "ExitCode": state.container_exit_code,
                "OOMKilled": False, "Status": "exited",
            }))
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "Running": state.proc.poll() is None, "ExitCode": state.proc.poll() or 0,
        }))

    def confirm(*args, **kwargs):
        assert args == (None, None)
        assert not state.launched
        assert kwargs["deadline"] == 60 - live.BOUNDED_SHUTDOWN_RESERVE_SECONDS
        clock.advance(state.confirm_delay)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ("engine", "onnx", "labels"):
        (model_dir / name).write_bytes(b"synthetic")
    infer = tmp_path / "infer.txt"
    infer.write_text("""[property]
batch-size=6
network-mode=2
network-type=0
num-detected-classes=3
filter-out-class-ids=1;2
model-engine-file=/models/engine
onnx-file=/models/onnx
labelfile-path=/models/labels
""")
    platform = tmp_path / "platform.json"
    platform.write_text(json.dumps({"image": "synthetic-pinned-image", "image_tag": "9.1"}))
    binary = tmp_path / ".runtime/viewer"
    binary.parent.mkdir()
    binary.write_text("synthetic non-executable fixture contents")
    binary.chmod(0o700)
    camera_list = [Camera(cam, f"rtsp://fixture.invalid/Streaming/Channels/{channel}",
                          "fixture-user", "fixture-password")
                   for cam, channel in zip(CAMERAS, CHANNELS)]
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "PLATFORM", platform)
    monkeypatch.setattr(live, "MOSAIC_PREVIEW_PATH", str(tmp_path / "ipc/preview.bin"))
    monkeypatch.setattr(live, "MOSAIC_FULLSCREEN_PREVIEW_PATH", str(tmp_path / "ipc/fullscreen.bin"))
    monkeypatch.setattr(live, "MOSAIC_FULLSCREEN_REQUEST_PATH", str(tmp_path / "ipc/fullscreen-request"))
    monkeypatch.setattr(live, "load_settings", lambda unused: None)
    monkeypatch.setattr(live, "configured_cameras", lambda unused: camera_list)
    monkeypatch.setattr(live, "nvr_sessions_preflight", lambda *args: None)
    monkeypatch.setattr(live, "checked_network", lambda *args: None)
    monkeypatch.setattr(live, "_ask_operator", confirm)
    monkeypatch.setattr(live.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(live.time, "sleep", clock.advance)
    monkeypatch.setattr(live.threading, "Timer", Timer)
    monkeypatch.setattr(live.threading, "Thread", Thread)
    monkeypatch.setattr(live.subprocess, "Popen", Process)
    monkeypatch.setattr(live.subprocess, "run", run)
    monkeypatch.setattr(live, "SourceFpsMonitor", lambda **kwargs: SimpleNamespace(
        snapshot=lambda: {"ready": state.ready()}))
    args = ["--run", "--authorize-physical-rtsp", "--duration", "60",
            "--infer-config", str(infer), "--models-root", str(model_dir),
            "--expected-engine-sha256", hashlib.sha256(b"synthetic").hexdigest(),
            "--mosaic-preview", "--mosaic-viewer-binary", str(binary)]
    state.clock, state.args, state.root = clock, args, tmp_path
    return state


def assert_released(state):
    from services.shared.source_ownership import SourceOwnership
    assert not (state.root / "ipc/preview.bin").exists()
    with SourceOwnership(CAMERAS, state.root / ".runtime/camera-owner-locks"):
        pass
    assert state.clock.now <= 60


def test_confirmation_precedes_camera_creation_and_cleanup_is_inside_budget(runner):
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    assert live.main(runner.args) == 0
    assert len(runner.launched) == 1
    launched_at, cmd = runner.launched[0]
    assert launched_at == 1
    assert "timeout --foreground --verbose --signal=KILL 44.000s" in cmd[-1]
    assert "--kill-after" not in cmd[-1]
    assert "--tiledtext" in cmd[-1]
    assert runner.clock.now == pytest.approx(30)
    assert any(cmd[:4] == ["docker", "kill", "--signal", "SIGINT"]
               for _, cmd, _ in runner.calls)
    assert runner.clock.timers[0].cancelled
    assert original == {sig: signal.getsignal(sig) for sig in original}
    assert_released(runner)


def test_time_spent_confirming_does_not_extend_deadline(runner):
    runner.confirm_delay = 6
    runner.ready = lambda: runner.clock.now >= 6.2
    assert live.main(runner.args) == 0
    assert "39.000s /bin/sh" in runner.launched[0][1][-1]
    assert runner.clock.now == pytest.approx(30)
    assert_released(runner)


def test_ml_telemetry_is_explicit_opt_in_and_uses_test_owned_ipc(runner, monkeypatch):
    from services.gpu.runtime.telemetry_staging import StagedTelemetry

    telemetry_dir = runner.root / "private-telemetry"
    telemetry_dir.mkdir(mode=0o700)
    monkeypatch.setattr(live.StagedTelemetry, "host_contracts",
                        classmethod(lambda cls: StagedTelemetry(telemetry_dir)))
    monkeypatch.setattr(live, "_telemetry_ipc_checks", lambda *, enabled=True: {
        "monitoring_telemetry_absent": (not (telemetry_dir / "v11_monitoring_telemetry_v1.json").exists()),
        "tracked_object_telemetry_absent": all(
            not (telemetry_dir / f"v11_tracked_objects_cam{i:02d}_v1.json").exists()
            for i in range(1, 7)),
        "telemetry_owner_lock_absent": not (telemetry_dir / StagedTelemetry.LOCK_NAME).exists(),
    } if enabled else {"monitoring_telemetry_absent": True,
                       "tracked_object_telemetry_absent": True,
                       "telemetry_owner_lock_absent": True})

    class Loop:
        def __init__(self, publisher):
            self.publisher = publisher
            self.failure_code = None

        def start(self):
            assert self.publisher.publish_once()

        def stop(self):
            self.publisher.observe_pipeline(
                "stopped", timestamp_ns=__import__("time").monotonic_ns())
            return self.publisher.close()

    monkeypatch.setattr(live, "TelemetryPublisherLoop", Loop)
    runner.args.append("--publish-ml-telemetry")
    assert live.main(runner.args) == 0
    _, command = runner.launched[0]
    assert "SIX_ML_TELEMETRY_ENABLED=1" in command
    assert not any(path.exists() for path in telemetry_dir.iterdir())


def test_late_confirmation_cannot_start_any_camera(runner):
    runner.confirm_delay = 34
    assert live.main(runner.args) == 2
    assert not runner.launched
    assert_released(runner)


def test_missing_start_expires_without_creating_a_container(runner, monkeypatch):
    monkeypatch.setattr(live, "_ask_operator", ASK_OPERATOR)
    monkeypatch.setattr(live.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    def no_input(inputs, outputs, errors, timeout):
        runner.clock.advance(timeout)
        return [], [], []
    monkeypatch.setattr(live.select, "select", no_input)
    assert live.main(runner.args) == 2
    assert runner.clock.now == 30
    assert not runner.launched
    assert not runner.clock.timers
    assert_released(runner)


def test_unresponsive_sigint_reports_gnu_kill_status_without_docker_race(runner, capsys):
    runner.graceful = False
    assert live.main(runner.args) == 2
    assert runner.proc.returncode == 137
    assert runner.clock.now == pytest.approx(46.3)
    assert all(timeout <= 1 for _, cmd, timeout in runner.calls if cmd[0] == "docker")
    assert "container_exit_code=137" in capsys.readouterr().err
    assert_released(runner)


def test_watchdog_stops_container_while_parent_is_blocked_on_handoff(runner):
    runner.block_write = True
    assert live.main(runner.args) == 2
    assert runner.clock.timers[0].fired
    assert runner.proc.returncode == 137
    assert_released(runner)


def test_readiness_loss_aborts_and_releases_all_owners(runner):
    runner.ready = lambda: 1.3 <= runner.clock.now < 2
    assert live.main(runner.args) == 2
    assert_released(runner)
    payload = json.loads(next((runner.root / ".runtime").glob("six-live-failure-*.json")).read_text())
    assert payload["failure_stage"] == "observation_window"


def test_no_source_readiness_cannot_be_reported_as_success(runner):
    runner.ready = lambda: False
    assert live.main(runner.args) == 2
    assert_released(runner)


def test_child_crash_keeps_exact_exit_status_and_stops_without_retry(runner):
    runner.exit_at = 2
    assert live.main(runner.args) == 2
    assert len(runner.launched) == 1
    payload = json.loads(next((runner.root / ".runtime").glob("six-live-failure-*.json")).read_text())
    assert payload["exit_status"]["docker_run_returncode_at_failure_detection"] == 139
    assert "fixture-password" not in json.dumps(payload)
    assert_released(runner)


def test_manual_cancellation_runs_cleanup(runner, monkeypatch):
    def interrupt(seconds):
        runner.clock.advance(seconds)
        raise KeyboardInterrupt("synthetic cancellation")
    monkeypatch.setattr(live.time, "sleep", interrupt)
    assert live.main(runner.args) == 2
    assert_released(runner)


def test_cleanup_failure_is_never_success(runner):
    runner.remove_ok = False
    assert live.main(runner.args) == 2
    assert runner.clock.now <= 60


@pytest.mark.parametrize("payload,expected", [(b"START\n", None), (b"NO\n", "did not confirm"),
                                             (b"", "terminal closed"), (b"STA", "deadline expired")])
def test_terminal_input_is_bounded_even_for_partial_lines(monkeypatch, payload, expected):
    now = [0.0]
    input_count = [0]
    monkeypatch.setattr(live.sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 123))
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    def select(inputs, *args):
        now[0] += args[-1]
        input_count[0] += 1
        return (inputs if input_count[0] == 1 else [], [], [])
    monkeypatch.setattr(live.select, "select", select)
    monkeypatch.setattr(live.os, "read", lambda fd, size: payload)
    if expected:
        with pytest.raises(RuntimeError, match=expected):
            live._ask_operator(None, None, deadline=5)
    else:
        live._ask_operator(None, None, deadline=5)
    assert now[0] <= 5


def test_expired_cleanup_budget_never_starts_another_docker_command(monkeypatch):
    monkeypatch.setattr(live.time, "monotonic", lambda: 30)
    def forbidden(*args, **kwargs):
        raise AssertionError("Docker must not start a new cleanup wait after deadline")
    monkeypatch.setattr(live.subprocess, "run", forbidden)
    with pytest.raises(TimeoutError):
        live._remove_owned_container("owned-test", deadline=30)


@pytest.fixture
def daily(runner, monkeypatch):
    runner.args.remove("--duration")
    runner.args.remove("60")
    runner.args.append("--monitor")
    runner.ready = lambda: True
    monkeypatch.setattr(live, "display_preflight", lambda: (":0", runner.root / "private-auth"))

    class GUI:
        returncode = None
        close_at = None

        def poll(self):
            if self.close_at is not None and runner.clock.now >= self.close_at:
                self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        def wait(self, timeout):
            return self.returncode

    runner.gui = GUI()
    monkeypatch.setattr(live, "_start_camera_wall", lambda: runner.gui)
    runner.cancel_after = 2

    def cancel(seconds):
        runner.clock.advance(runner.cancel_after)
        raise KeyboardInterrupt("synthetic normal stop")
    monkeypatch.setattr(live.time, "sleep", cancel)
    return runner


def assert_daily_released(state):
    from services.shared.source_ownership import SourceOwnership
    assert not (state.root / "ipc/preview.bin").exists()
    assert state.gui.poll() is not None
    with SourceOwnership(CAMERAS, state.root / ".runtime/camera-owner-locks"):
        pass
    owner = json.loads((state.root / ".runtime/six-live-preview.lock").read_text())
    assert owner["mode"] == "daily" and owner["cleanup_ok"] is True
    assert owner["cleanup_checks"]["cleanup_ok"] is True
    assert all(owner["cleanup_checks"]["predicates"].values())


def test_daily_mode_runs_past_test_limit_without_timer_or_start_prompt(daily):
    previous_hup = signal.getsignal(signal.SIGHUP)
    daily.cancel_after = 700
    assert live.main(daily.args) == 0
    assert len(daily.launched) == 1
    assert "exec timeout" not in daily.launched[0][1][-1]
    assert daily.launch_options["start_new_session"] is True
    assert signal.getsignal(signal.SIGHUP) == previous_hup
    assert daily.clock.now == 700
    # Only the bounded shutdown watchdog was scheduled, after the stop request.
    assert len(daily.clock.timers) == 1
    assert daily.clock.timers[0].at == 718
    assert daily.clock.timers[0].cancelled
    assert_daily_released(daily)


def test_daily_graceful_stop_allows_native_teardown_longer_than_one_second(daily, monkeypatch, capsys):
    daily.graceful = False
    make_process = live.subprocess.Popen

    def create(*args, **kwargs):
        process = make_process(*args, **kwargs)
        def drain(timeout):
            assert timeout == 15
            daily.clock.advance(2)  # Reference teardown sleeps 1s, plus pipeline drain.
            process.returncode = 0
            return 0
        process.wait = drain
        return process
    monkeypatch.setattr(live.subprocess, "Popen", create)
    assert live.main(daily.args) == 0
    assert daily.proc.returncode == 0
    assert daily.clock.now == 4
    assert "Graceful DeepStream exit confirmed" in capsys.readouterr().out
    assert_daily_released(daily)


def test_daily_forced_removal_only_after_graceful_wait_expires(daily, capsys):
    daily.graceful = False
    assert live.main(daily.args) == 0
    assert daily.proc.returncode == 137
    removals = [at for at, cmd, _ in daily.calls if cmd[:2] == ["docker", "rm"]]
    assert removals == [17]  # Stop at 2s, then the full 15s grace period.
    assert daily.clock.now < 22  # Total cleanup budget is 20 seconds.
    assert "forcing owned container after grace period" in capsys.readouterr().out
    assert_daily_released(daily)


def test_bounded_slow_graceful_shutdown_finishes_before_kill_fallback(
        daily, monkeypatch, capsys):
    daily.args.extend(("--duration", "60"))
    daily.graceful_delay_seconds = 12
    daily.cancel_after = 30
    monkeypatch.setattr(live.time, "sleep", daily.clock.advance)
    assert live.main(daily.args) == 0
    assert daily.proc.returncode == 0
    assert daily.clock.now == pytest.approx(42)  # 30s observation + 12s drain.
    assert "timeout --foreground --verbose --signal=KILL 45.000s" in daily.launched[0][1][-1]
    output = capsys.readouterr().out
    assert "CONTAINER_WATCHDOG pid1=timeout mode=SIGKILL_FALLBACK" in output
    assert "internal_kill_deadline_monotonic=" in output
    assert "host_stop_deadline_monotonic=" in output
    assert "DOCKER_SIGNAL event=stop_request_started signal=SIGINT monotonic_ns=" in output
    assert "DOCKER_SIGNAL event=stop_request_completed signal=SIGINT" in output
    assert "waiting_for_kill_fallback" not in output
    assert daily.clock.timers[0].cancelled
    assert_daily_released(daily)


def test_bounded_unresponsive_child_reports_kill_fallback_before_host_removal(
        daily, monkeypatch, capsys):
    daily.args.extend(("--duration", "60"))
    daily.graceful = False
    monkeypatch.setattr(live.time, "sleep", daily.clock.advance)
    assert live.main(daily.args) == 2
    assert daily.proc.returncode == 137
    assert daily.container_exit_code == 137
    assert daily.clock.now == pytest.approx(46.3)
    sigints = [call for call in daily.calls
               if call[1][:4] == ["docker", "kill", "--signal", "SIGINT"]]
    assert len(sigints) == 1
    removals = [call for call in daily.calls if call[1][:2] == ["docker", "rm"]]
    assert len(removals) == 1 and removals[0][0] == pytest.approx(46.3)
    output = capsys.readouterr()
    assert "DOCKER_RUN_EXIT waiting_for_kill_fallback" in output.out
    assert "DeepStream container exit code: 137" in output.out
    assert ("RUN_STATUS FAILED stage=container_shutdown shutdown status: "
            "container_exit_code=137") in output.err
    assert_daily_released(daily)


def test_repeated_sigint_during_shutdown_is_ignored_and_sent_once(
        daily, monkeypatch):
    registered = {}
    real_signal = live.signal.signal

    def capture_handler(sig, handler):
        previous = real_signal(sig, handler)
        if callable(handler):
            registered[sig] = handler
        return previous

    monkeypatch.setattr(live.signal, "signal", capture_handler)
    def first_interrupt(seconds):
        daily.clock.advance(daily.cancel_after)
        registered[signal.SIGINT](signal.SIGINT, None)

    monkeypatch.setattr(live.time, "sleep", first_interrupt)
    stop = live._stop_owned_container
    second_signal_observed = []

    def stop_with_duplicate_signal(name, *, deadline=None):
        # The first SIGINT entered cleanup. Further user signals are ignored
        # until cleanup completes, so they cannot interrupt or duplicate stop.
        assert live.signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        os.kill(os.getpid(), signal.SIGINT)
        second_signal_observed.append(True)
        return stop(name, deadline=deadline)

    monkeypatch.setattr(live, "_stop_owned_container", stop_with_duplicate_signal)
    assert live.main(daily.args) == 0
    assert second_signal_observed == [True]
    sent = [call for call in daily.calls
            if call[1][:4] == ["docker", "kill", "--signal", "SIGINT"]]
    assert len(sent) == 1
    assert_daily_released(daily)


def test_gnu_timeout_forwards_one_sigint_then_kills_only_child():
    import shutil

    timeout = shutil.which("timeout")
    assert timeout is not None
    child = "trap '' INT; exec sleep 5"
    process = subprocess.Popen(
        [timeout, "--foreground", "--verbose", "--signal=KILL", "0.5s",
         "sh", "-c", child],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        time.sleep(0.05)
        os.kill(process.pid, signal.SIGINT)
        assert process.wait(timeout=2) == 137
        output = process.stdout.read() if process.stdout else ""
        assert output.count("sending signal INT") == 1
        assert output.count("sending signal KILL") == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)


def test_bounded_monitor_preserves_nonzero_container_exit_after_clean_resource_cleanup(
    daily, capsys,
):
    daily.args.extend(("--duration", "60"))
    daily.container_exit_code = 137
    result = live.main(daily.args)
    output = capsys.readouterr()
    assert result == 2
    assert "DeepStream container exit code: 137" in output.out
    assert ("RUN_STATUS FAILED stage=container_shutdown shutdown status: "
            "container_exit_code=137") in output.err
    assert_daily_released(daily)


def test_shutdown_diagnostics_cover_native_teardown_order():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    stage = root / ".runtime/deepstream-ml-telemetry-offline-20261028-r1/sources/apps/sample_apps/deepstream-app"
    main = (stage / "deepstream_app_main.c").read_text()
    app = (stage / "deepstream_app.c").read_text()
    monitor = (stage / "six_live_monitor.c").read_text()
    preview = (stage / "mosaic_preview_branch.c").read_text()
    for marker in (
        "event=sigint_received", "event=process_teardown_started",
        "event=instance_teardown_completed", "event=process_exit_initiated",
    ):
        assert marker in main
    teardown = app[app.index("void\ndestroy_pipeline ("):]
    events = (
        "event=eos_send_started", "event=eos_bus_drain_completed",
        "event=pipeline_null_started", "event=pipeline_null_observed",
        "event=mosaic_preview_stop_started", "event=mosaic_preview_stop_completed",
        "event=six_live_stop_started", "event=six_live_stop_completed",
    )
    assert [teardown.index(event) for event in events] == sorted(
        teardown.index(event) for event in events
    )
    assert "event=telemetry_thread_join_started" in monitor
    assert "event=telemetry_thread_join_completed" in monitor
    assert "event=mosaic_preview_resources_release_completed" in preview


def test_bounded_stop_wait_obeys_deadline_persisted_in_owner(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(live, "ROOT", tmp_path)
    path = tmp_path / ".runtime/six-live-preview.lock"
    path.parent.mkdir()
    deadline = 3.0
    path.write_text(json.dumps({"mode": "daily", "pid": os.getpid(),
                               "start_ticks": live._process_start_ticks(os.getpid()),
                               "cleanup_ok": None, "run_deadline_monotonic": deadline}))
    now = [0.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(live.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(live.os, "kill", lambda *_: None)
    monkeypatch.setattr(live.fcntl, "flock", lambda *_: (_ for _ in ()).throw(BlockingIOError()))
    rc = live.stop_monitoring()
    assert rc == 2
    assert now[0] <= deadline


def test_daily_closing_camera_wall_stops_same_pipeline(daily, monkeypatch):
    daily.gui.close_at = 2
    monkeypatch.setattr(live.time, "sleep", daily.clock.advance)
    assert live.main(daily.args) == 0
    assert len(daily.launched) == 1
    assert_daily_released(daily)


def test_daily_source_startup_remains_bounded(daily, monkeypatch):
    daily.ready = lambda: False
    monkeypatch.setattr(live.time, "sleep", daily.clock.advance)
    assert live.main(daily.args) == 2
    assert 120 <= daily.clock.now < 121
    assert len(daily.launched) == 1
    assert_daily_released(daily)


def test_daily_gui_uses_supported_main_wall_without_camera_secrets(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("CAM_02_RTSP_PASSWORD", "synthetic-hidden-password")
    monkeypatch.delenv("SURVEILLANCE_API_TOKEN", raising=False)
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("ML_HOST", "0.0.0.0")
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "https://api.example.invalid")
    monkeypatch.setenv("FRONTEND_ML_VIDEO_BASE_URL", "https://ml.example.invalid")
    monkeypatch.setenv("ML_SERVICE_URL", "https://ml-service.example.invalid")
    monkeypatch.setenv("V11_UI_FULLSCREEN_SEQUENCE_OUTPUT", "/tmp/ambient-sequence.jsonl")
    monkeypatch.setenv("V11_UI_FULLSCREEN_CAPTURE_MODE", "x11-window")
    monkeypatch.setattr(live, "runtime_python", lambda root: tmp_path / "venv/bin/python")
    monkeypatch.setattr(live.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))
    live._start_camera_wall()
    argv, options = calls[0]
    assert argv == [str(tmp_path / "venv/bin/python"), "-m", "services.frontend.app.main"]
    assert options["env"]["V11_UI_MOSAIC_WALL_ENABLED"] == "1"
    assert "V11_UI_FULLSCREEN_SEQUENCE_OUTPUT" not in options["env"]
    assert "V11_UI_FULLSCREEN_CAPTURE_MODE" not in options["env"]
    assert "CAM_02_RTSP_PASSWORD" not in options["env"]
    for key in ("API_HOST", "ML_HOST", "FRONTEND_API_BASE_URL",
                "FRONTEND_ML_VIDEO_BASE_URL", "ML_SERVICE_URL", "SURVEILLANCE_API_TOKEN"):
        assert key not in options["env"]
    assert options["env"]["PYTHON_DOTENV_DISABLED"] == "1"
    assert options["stdout"] == live.subprocess.PIPE
    assert options["stderr"] == live.subprocess.STDOUT
    assert options["start_new_session"] is True


def test_opt_in_fullscreen_verifier_replaces_the_single_wall_process(tmp_path, monkeypatch):
    calls = []
    output_dir = tmp_path / "fullscreen-private"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    output = output_dir / "fullscreen.jsonl"
    monkeypatch.delenv("SURVEILLANCE_API_TOKEN", raising=False)
    monkeypatch.setattr(live, "runtime_python", lambda root: tmp_path / "venv/bin/python")
    monkeypatch.setattr(live.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))

    live._start_camera_wall(fullscreen_sequence_output=output)

    assert len(calls) == 1
    argv, options = calls[0]
    assert argv == [str(tmp_path / "venv/bin/python"), "-m",
                    "scripts.validate_production_wall", "--output", str(output),
                    "--fullscreen-sequence", "--capture-mode", "qt-widget"]
    assert "services.frontend.app.main" not in argv
    assert options["env"]["V11_UI_MOSAIC_WALL_ENABLED"] == "1"
    assert options["env"]["V11_UI_FULLSCREEN_SEQUENCE_OUTPUT"] == str(output)
    assert options["env"]["V11_UI_FULLSCREEN_CAPTURE_MODE"] == "qt-widget"
    assert "V11_UI_MOSAIC_PIXEL_PROVENANCE" not in options["env"]


def test_pixel_provenance_opt_in_reaches_only_verifier_ui(tmp_path, monkeypatch):
    calls = []
    output_dir = tmp_path / "fullscreen-private"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    output = output_dir / "fullscreen.jsonl"
    monkeypatch.delenv("SURVEILLANCE_API_TOKEN", raising=False)
    monkeypatch.setenv("V11_UI_MOSAIC_PIXEL_PROVENANCE", "1")
    monkeypatch.setattr(live, "runtime_python", lambda root: tmp_path / "venv/bin/python")
    monkeypatch.setattr(live.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))

    live._start_camera_wall(fullscreen_sequence_output=output, pixel_provenance=True,
                            grid_only=True)

    argv, options = calls[0]
    assert options["env"]["V11_UI_MOSAIC_PIXEL_PROVENANCE"] == "1"
    assert "--grid-only" in argv
    assert "--fullscreen-sequence" not in argv


def test_verifier_environment_clears_ambient_pixel_provenance_by_default(monkeypatch):
    monkeypatch.setenv("V11_UI_MOSAIC_PIXEL_PROVENANCE", "1")
    assert "V11_UI_MOSAIC_PIXEL_PROVENANCE" not in live._camera_wall_environment()


def test_targeted_fullscreen_cameras_reach_the_existing_verifier(tmp_path, monkeypatch):
    calls = []
    output_dir = tmp_path / "fullscreen-private"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    output = output_dir / "fullscreen.jsonl"
    monkeypatch.delenv("SURVEILLANCE_API_TOKEN", raising=False)
    monkeypatch.setattr(live, "runtime_python", lambda root: tmp_path / "venv/bin/python")
    monkeypatch.setattr(live.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))

    live._start_camera_wall(
        fullscreen_sequence_output=output,
        fullscreen_camera_ids=["CAM-03", "CAM-06"],
    )

    argv, _ = calls[0]
    assert argv[-4:] == ["--fullscreen-camera", "CAM-03",
                         "--fullscreen-camera", "CAM-06"]


def test_fullscreen_verifier_preflight_starts_exact_verifier_without_camera_network(tmp_path, monkeypatch):
    output_dir = tmp_path / "fullscreen-private"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    output = output_dir / "fullscreen.jsonl"
    calls = []
    marker = "FULLSCREEN_VERIFIER_PREFLIGHT_PASS camera_source_ids=" + ",".join(
        f"{camera_id}:{source_id}"
        for source_id, camera_id in enumerate(live.MOSAIC_CAMERA_IDS)
    )
    monkeypatch.setenv("CAM_02_RTSP_PASSWORD", "synthetic-secret-value")
    monkeypatch.delenv("SURVEILLANCE_API_TOKEN", raising=False)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "runtime_python", lambda root: tmp_path / "venv/bin/python")
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return live.subprocess.CompletedProcess(argv, 0, stdout=marker + "\n", stderr="")
    monkeypatch.setattr(live.subprocess, "run", run)

    live._fullscreen_verifier_preflight(output, ["CAM-03", "CAM-06"])

    assert len(calls) == 1
    argv, options = calls[0]
    assert argv == [str(tmp_path / "venv/bin/python"), "-m",
                    "scripts.validate_production_wall", "--preflight-only",
                    "--fullscreen-camera", "CAM-03", "--fullscreen-camera", "CAM-06"]
    assert options["timeout"] == live.FULLSCREEN_VERIFIER_PREFLIGHT_TIMEOUT_SECONDS
    assert options["env"]["V11_UI_FULLSCREEN_SEQUENCE_OUTPUT"] == str(output)
    assert "CAM_02_RTSP_PASSWORD" not in options["env"]
    assert "nvr_sessions_preflight" not in " ".join(argv)


def test_pixel_provenance_flag_is_propagated_during_verifier_preflight(tmp_path, monkeypatch):
    output_dir = tmp_path / "fullscreen-private"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    output = output_dir / "fullscreen.jsonl"
    marker = "FULLSCREEN_VERIFIER_PREFLIGHT_PASS camera_source_ids=" + ",".join(
        f"{camera_id}:{source_id}"
        for source_id, camera_id in enumerate(live.MOSAIC_CAMERA_IDS)
    )
    calls = []
    monkeypatch.delenv("SURVEILLANCE_API_TOKEN", raising=False)
    monkeypatch.setattr(live, "runtime_python", lambda root: tmp_path / "venv/bin/python")
    monkeypatch.setattr(live.subprocess, "run", lambda argv, **kwargs: (
        calls.append((argv, kwargs)) or
        live.subprocess.CompletedProcess(argv, 0, stdout=marker + "\n", stderr="")
    ))

    live._fullscreen_verifier_preflight(output, ["CAM-05"], pixel_provenance=True)

    assert calls[0][1]["env"]["V11_UI_MOSAIC_PIXEL_PROVENANCE"] == "1"


def test_fullscreen_verifier_preflight_fails_closed_without_ready_marker(tmp_path, monkeypatch):
    output_dir = tmp_path / "fullscreen-private"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    output = output_dir / "fullscreen.jsonl"
    monkeypatch.setattr(live, "runtime_python", lambda root: tmp_path / "venv/bin/python")
    monkeypatch.setattr(live.subprocess, "run", lambda argv, **kwargs:
                        live.subprocess.CompletedProcess(argv, 1, stdout="", stderr="UI startup failed"))
    with pytest.raises(RuntimeError, match="fullscreen verifier preflight failed before camera startup"):
        live._fullscreen_verifier_preflight(output)


def test_verify_fullscreen_cli_missing_path_fails_before_camera_preflight(runner, monkeypatch):
    calls = []
    monkeypatch.delenv("V11_UI_FULLSCREEN_SEQUENCE_OUTPUT", raising=False)
    monkeypatch.setattr(live, "nvr_sessions_preflight", lambda *args: calls.append("nvr"))
    with pytest.raises(SystemExit):
        live.main(runner.args + ["--monitor", "--verify-fullscreen"])
    assert calls == []
    assert not runner.launched


def test_pixel_provenance_grid_cli_requires_private_output_before_network_preflight(monkeypatch):
    calls = []
    monkeypatch.setattr(live, "nvr_sessions_preflight", lambda *_args: calls.append("nvr"))
    with pytest.raises(SystemExit):
        live.main([
            "--infer-config", "/private/offline/infer.txt",
            "--models-root", "/private/offline/models",
            "--run", "--monitor", "--mosaic-preview",
            "--mosaic-viewer-binary", "/private/offline/viewer",
            "--mosaic-pixel-provenance",
        ])
    assert calls == []


def test_camera_wall_keeps_authenticated_deployment_endpoints_when_token_exists(monkeypatch):
    monkeypatch.setenv("SURVEILLANCE_API_TOKEN", "synthetic-token-value-that-is-longer-than-32-chars")
    monkeypatch.setenv("API_HOST", "10.0.0.8")
    monkeypatch.setenv("ML_HOST", "10.0.0.9")
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "https://api.example.invalid")
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)

    env = live._camera_wall_environment()

    assert env["SURVEILLANCE_API_TOKEN"].startswith("synthetic-token-")
    assert env["API_HOST"] == "10.0.0.8"
    assert env["ML_HOST"] == "10.0.0.9"
    assert env["FRONTEND_API_BASE_URL"] == "https://api.example.invalid"
    assert "PYTHON_DOTENV_DISABLED" not in env


def test_stop_cli_needs_no_model_configuration_or_camera_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROOT", tmp_path)
    def forbidden(*args):
        raise AssertionError("stop must not read cameras or launch a process")
    monkeypatch.setattr(live, "load_settings", forbidden)
    monkeypatch.setattr(live.subprocess, "Popen", forbidden)
    assert live.main(["--stop-monitoring"]) == 0


@pytest.mark.parametrize("field,value", [("start_ticks", "stale"), ("mode", "bounded")])
def test_stop_refuses_stale_pid_or_non_daily_owner(tmp_path, monkeypatch, field, value):
    import fcntl
    import os
    monkeypatch.setattr(live, "ROOT", tmp_path)
    path = tmp_path / ".runtime/six-live-preview.lock"
    path.parent.mkdir()
    with path.open("w+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = {"mode": "daily", "pid": os.getpid(), "start_ticks": live._process_start_ticks(os.getpid())}
        owner[field] = value
        json.dump(owner, lock); lock.flush()
        def forbidden(*args):
            raise AssertionError("must never signal an unverified owner")
        monkeypatch.setattr(live.os, "kill", forbidden)
        assert live.stop_monitoring() == 2


@pytest.mark.parametrize("cleanup_ok", [False, None])
def test_stop_reports_previous_cleanup_failure(tmp_path, monkeypatch, cleanup_ok):
    monkeypatch.setattr(live, "ROOT", tmp_path)
    path = tmp_path / ".runtime/six-live-preview.lock"
    path.parent.mkdir()
    path.write_text(json.dumps({"mode": "daily", "cleanup_ok": cleanup_ok}))
    assert live.stop_monitoring() == 2


def test_stop_signals_verified_cpu_owner_and_waits_for_resource_lock_release(tmp_path, monkeypatch):
    import os
    import select
    import subprocess
    import sys
    monkeypatch.setattr(live, "ROOT", tmp_path)
    path = tmp_path / ".runtime/six-live-preview.lock"
    path.parent.mkdir()
    code = '''import fcntl,json,os,signal,sys
lock=open(sys.argv[1],"w+")
fcntl.flock(lock,fcntl.LOCK_EX)
owner={"mode":"daily","pid":os.getpid(),"start_ticks":open(f"/proc/{os.getpid()}/stat").read().rsplit(")",1)[1].split()[19],"cleanup_ok":None}
def write():
 lock.seek(0);lock.truncate();json.dump(owner,lock);lock.flush()
def stop(*args):
 owner["cleanup_ok"]=True;write();lock.close();sys.exit(0)
signal.signal(signal.SIGTERM,stop)
write();print("READY",flush=True)
while True:signal.pause()
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([child.stdout], [], [], 2)[0]
        assert child.stdout.readline().strip() == "READY"
        assert live.stop_monitoring() == 0
        assert child.wait(timeout=2) == 0
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=2)


def test_remove_timeout_can_still_mean_daemon_completed_removal(monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "rm":
            raise live.subprocess.TimeoutExpired(cmd, 1)
        assert cmd[:2] == ["docker", "ps"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(live.subprocess, "run", run)
    assert live._remove_owned_container("owned-test") is True
    assert [cmd[1] for cmd in calls] == ["rm", "ps"]


def test_remove_waits_boundedly_for_async_daemon_completion(monkeypatch):
    now = [0.0]
    queries = []
    def run(cmd, **kwargs):
        if cmd[1] == "rm":
            now[0] += 1
            raise live.subprocess.TimeoutExpired(cmd, 1)
        queries.append(now[0])
        return SimpleNamespace(returncode=0, stdout="owned-test\n" if len(queries) == 1 else "", stderr="")
    monkeypatch.setattr(live.subprocess, "run", run)
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(live.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert live._remove_owned_container("owned-test", deadline=3) is True
    assert len(queries) == 2 and now[0] == pytest.approx(1.1)


def test_rm_success_does_not_override_a_still_present_container(monkeypatch):
    now = [0.0]
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="owned-test\n" if cmd[1] == "ps" else "", stderr="")
    monkeypatch.setattr(live.subprocess, "run", run)
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(live.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert live._remove_owned_container("owned-test", deadline=0.25) is False
    assert now[0] <= 0.25
    assert sum(cmd[1] == "rm" for cmd in calls) == 1


@pytest.mark.parametrize("rm_exit,rm_error", [(0, ""), (1, "No such container: owned-test")])
def test_daemon_query_failure_never_proves_removal(monkeypatch, rm_exit, rm_error):
    def run(cmd, **kwargs):
        if cmd[1] == "rm":
            return SimpleNamespace(returncode=rm_exit, stdout="", stderr=rm_error)
        return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")
    monkeypatch.setattr(live.subprocess, "run", run)
    assert live._remove_owned_container("owned-test") is False


def test_absence_check_matches_exact_container_name(monkeypatch):
    monkeypatch.setattr(live.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="owned-test-extra\nunrelated\n", stderr=""))
    assert live._container_absent("owned-test") is True
    assert live._container_absent("owned-test-", prefix=True) is False


def test_daily_final_state_recheck_corrects_stale_removal_result(daily, monkeypatch, capsys):
    def delayed_removal(name, **kwargs):
        daily.container_present = False
        return False  # Simulate a CLI timeout/uncertain immediate response.
    monkeypatch.setattr(live, "_remove_owned_container", delayed_removal)
    assert live.main(daily.args) == 0
    output = capsys.readouterr()
    assert "OWNED DEEPSTREAM CONTAINER REMOVED" in output.out
    assert "removal could not be verified" not in output.err
    assert_daily_released(daily)
    owner = json.loads((daily.root / ".runtime/six-live-preview.lock").read_text())
    assert owner["container_name"].startswith("ai-surveillance-six-live-")


def test_daily_native_exit_is_not_inferred_from_hung_attach_client(daily, monkeypatch, capsys):
    daily.graceful = False  # Attach CLI wait times out, independently of native state.
    run = live.subprocess.run
    def native_already_exited(cmd, **kwargs):
        if cmd[1] == "inspect":
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "Status": "exited", "Running": False, "ExitCode": 0}))
        return run(cmd, **kwargs)
    monkeypatch.setattr(live.subprocess, "run", native_already_exited)
    assert live.main(daily.args) == 0
    output = capsys.readouterr().out
    assert "Graceful DeepStream exit confirmed" in output
    assert "container is still running" not in output
    assert_daily_released(daily)


@pytest.mark.parametrize("daemon_result,expected", [("absent", 0), ("present", 2), ("unavailable", 2)])
def test_stop_rechecks_old_cleanup_flag_without_signalling_any_process(
        tmp_path, monkeypatch, daemon_result, expected):
    import os
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "MOSAIC_PREVIEW_PATH", str(tmp_path / "absent-preview"))
    path = tmp_path / ".runtime/six-live-preview.lock"
    path.parent.mkdir()
    path.write_text(json.dumps({"mode": "daily", "pid": 12345, "cleanup_ok": False}))
    original = path.read_bytes()
    calls = []
    def query(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[:2] == ["docker", "ps"]
        return SimpleNamespace(returncode=1 if daemon_result == "unavailable" else 0,
            stdout="ai-surveillance-six-live-12345-example\n" if daemon_result == "present" else "", stderr="")
    def forbidden(*args):
        raise AssertionError("historical recheck must never signal or restart anything")
    monkeypatch.setattr(live.subprocess, "run", query)
    monkeypatch.setattr(live.os, "kill", forbidden)
    assert live.stop_monitoring() == expected
    assert len(calls) == 1
    assert path.read_bytes() == original  # Preserve the historical shutdown record.


def test_stale_status_cannot_be_reconciled_while_preview_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROOT", tmp_path)
    preview = tmp_path / "preview"
    preview.write_bytes(b"still present")
    monkeypatch.setattr(live, "MOSAIC_PREVIEW_PATH", str(preview))
    def forbidden(*args, **kwargs):
        raise AssertionError("preview is still present; cannot declare cleanup")
    monkeypatch.setattr(live.subprocess, "run", forbidden)
    assert live._recheck_monitor_cleanup({"pid": 12345}) is False


def test_cleanup_snapshot_records_each_successful_predicate(tmp_path, monkeypatch):
    preview = tmp_path / "preview-v1.bin"
    fullscreen = tmp_path / "fullscreen-v1.bin"
    request = tmp_path / "fullscreen-request-v1"
    request.write_bytes(b"\n")
    request.chmod(0o600)
    monkeypatch.setattr(live, "MOSAIC_PREVIEW_PATH", str(preview))
    monkeypatch.setattr(live, "MOSAIC_FULLSCREEN_PREVIEW_PATH", str(fullscreen))
    monkeypatch.setattr(live, "MOSAIC_FULLSCREEN_REQUEST_PATH", str(request))
    monkeypatch.setattr(live, "_camera_locks_free", lambda: True)
    monkeypatch.setattr(live.time, "monotonic", lambda: 40.0)
    proc = SimpleNamespace(poll=lambda: 0)
    wall = SimpleNamespace(poll=lambda: 0)

    checks = live._cleanup_check_snapshot(
        container_owned=True, container_removed=True, proc=proc, camera_wall=wall,
        camera_locks_owned=True, preview_enabled=True, run_deadline=50.0,
    )

    assert checks["cleanup_ok"] is True
    assert checks["deadline_remaining_seconds"] == 10.0
    assert all(checks["predicates"].values())
    assert checks["predicates"]["owned_rtsp_sessions_closed"] is True
    assert checks["predicates"]["fullscreen_request_cleared"] is True


def test_cleanup_snapshot_identifies_each_failed_predicate(tmp_path, monkeypatch):
    preview = tmp_path / "preview-v1.bin"
    fullscreen = tmp_path / "fullscreen-v1.bin"
    request = tmp_path / "fullscreen-request-v1"
    fullscreen.write_bytes(b"leftover fullscreen slot")
    request.write_text("CAM-03\n")
    request.chmod(0o600)
    monkeypatch.setattr(live, "MOSAIC_PREVIEW_PATH", str(preview))
    monkeypatch.setattr(live, "MOSAIC_FULLSCREEN_PREVIEW_PATH", str(fullscreen))
    monkeypatch.setattr(live, "MOSAIC_FULLSCREEN_REQUEST_PATH", str(request))
    monkeypatch.setattr(live, "_camera_locks_free", lambda: False)
    monkeypatch.setattr(live.time, "monotonic", lambda: 102.0)
    proc = SimpleNamespace(poll=lambda: None)
    wall = SimpleNamespace(poll=lambda: None)

    checks = live._cleanup_check_snapshot(
        container_owned=True, container_removed=False, proc=proc, camera_wall=wall,
        camera_locks_owned=True, preview_enabled=True, run_deadline=100.0,
    )
    failed = {name for name, passed in checks["predicates"].items() if not passed}

    assert checks["cleanup_ok"] is False
    assert checks["deadline_remaining_seconds"] == -2.0
    assert failed == {
        "container_absent", "owned_rtsp_sessions_closed", "docker_run_client_reaped",
        "camera_wall_reaped", "camera_source_locks_free",
        "fullscreen_preview_slot_absent", "fullscreen_request_cleared",
        "within_total_deadline",
    }


def test_camera_lock_check_does_not_replace_or_delete_owned_lock(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(live, "ROOT", tmp_path)
    path = tmp_path / ".runtime/camera-owner-locks/CAM-02.lock"
    path.parent.mkdir(parents=True)
    with path.open("w+") as lock:
        lock.write("preserve owner evidence"); lock.flush()
        original_inode = path.stat().st_ino
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert live._camera_locks_free() is False
    assert live._camera_locks_free() is True
    assert path.stat().st_ino == original_inode
    assert path.read_text() == "preserve owner evidence"


def test_monitor_script_pins_the_fullscreen_source_selection_build():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "scripts/six_camera_live/monitor.sh")
    text = script.read_text(encoding="utf-8")
    assert ".runtime/deepstream-rtsp-pause-suppression-reporting-20261030-r1/" in text
    assert "2602a28a191e3b8671e5e9e4d4fda1fc98d428b70b2d4235bfea8684c269483e" in text
    assert ".runtime/fullscreen-quality-fix-20261026-r2/" not in text
    assert "--verify-fullscreen" in text
    assert 'V11_UI_FULLSCREEN_SEQUENCE_OUTPUT="$sequence_dir/fullscreen-sequence.jsonl"' in text
    assert 'fullscreen_args=(--verify-fullscreen)' in text
    assert 'unset V11_UI_FULLSCREEN_SEQUENCE_OUTPUT V11_UI_FULLSCREEN_CAPTURE_MODE' in text

@pytest.mark.parametrize('capture_ok', [True, False])
def test_opt_in_gdb_reaches_actual_launcher_and_preserves_failure(runner, monkeypatch, capture_ok):
    from scripts import native_shutdown_diagnostics as diag
    bundle = runner.root / 'diagnostic-tools'
    bundle.mkdir()
    for name in ('gdb', 'capture.py', 'target.py', 'manifest.json', 'lib/ld-linux-x86-64.so.2'):
        path = bundle / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    validated = []
    monkeypatch.setattr(diag, 'validate_six_live_bundle', lambda *a: validated.append(a))
    captures = []
    def start(*args):
        captures.append(args)
        args[-1].update(ok=capture_ok, capture_exit=0 if capture_ok else 2)
        return SimpleNamespace(join=lambda timeout:None, is_alive=lambda:False)
    monkeypatch.setattr(diag, 'start_shutdown_capture', start)
    runner.args.extend(('--shutdown-gdb-dir', str(bundle)))
    assert live.main(runner.args) == (0 if capture_ok else 2)
    assert len(validated) == 1 and len(captures) == 1
    assert '--cap-add=SYS_PTRACE' in runner.launched[0][1]
    assert '/diag/target.py' in runner.launched[0][1][-1]
    assert_released(runner)
