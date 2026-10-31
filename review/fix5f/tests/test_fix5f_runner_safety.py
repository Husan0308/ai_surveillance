from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/six_camera_live/run_fix5f_plugin_test.py"
SPEC = importlib.util.spec_from_file_location("fix5f_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_explicit_docker_not_found_is_absent(monkeypatch):
    monkeypatch.setattr(runner, "run_readonly", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="Error: No such object: fix5f-test"))
    assert runner.inspect_owned_container("fix5f-test", "run-id") is None


@pytest.mark.parametrize("stderr", [
    "Cannot connect to the Docker daemon",
    "permission denied while trying to connect",
    "",
])
def test_inspection_errors_fail_closed(monkeypatch, stderr):
    monkeypatch.setattr(runner, "run_readonly", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr=stderr))
    with pytest.raises(runner.Blocked, match="could not verify"):
        runner.inspect_owned_container("fix5f-test", "run-id")


def test_malformed_inspection_is_not_absence(monkeypatch):
    monkeypatch.setattr(runner, "run_readonly", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="not-json", stderr=""))
    with pytest.raises(runner.Blocked, match="malformed"):
        runner.inspect_owned_container("fix5f-test", "run-id")


def test_container_label_and_immutable_id_are_required(monkeypatch):
    payload = [{"Id": "immutable-container-id", "Name": "/fix5f-test",
                "Config": {"Labels": {runner.OWNER_LABEL: "run-id"}},
                "State": {"Status": "running", "ExitCode": 0, "OOMKilled": False}}]
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "inspect"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner, "run_readonly", fake_run)
    assert runner.docker_owner_action("stop", "fix5f-test", "run-id", 3)
    assert calls[-1][-1] == "immutable-container-id"


def test_unowned_container_cannot_be_stopped(monkeypatch):
    payload = [{"Id": "unowned-id", "Name": "/fix5f-test",
                "Config": {"Labels": {runner.OWNER_LABEL: "somebody-else"}},
                "State": {"Status": "running"}}]
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(runner, "run_readonly", fake_run)
    with pytest.raises(runner.Blocked, match="without this run's ownership label"):
        runner.docker_owner_action("stop", "fix5f-test", "run-id", 3)
    assert len(calls) == 1


def test_cli_without_execution_latch_only_preflights(monkeypatch, capsys):
    monkeypatch.setattr(runner, "preflight", lambda: {"ready": True})
    monkeypatch.setattr(runner, "execute", lambda: pytest.fail("execution must not start"))
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "--preflight-only"])
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out) == {"ready": True}


def test_failed_preflight_never_calls_execute(monkeypatch):
    def fail_preflight():
        raise runner.Blocked("fixture preflight block")

    monkeypatch.setattr(runner, "preflight", fail_preflight)
    monkeypatch.setattr(runner, "execute", lambda: pytest.fail("execution must not start"))
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "--preflight-only"])
    assert runner.main() == 2
