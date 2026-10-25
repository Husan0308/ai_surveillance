import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.cam_six_validation.credential_redaction import CredentialRedactor
from scripts.cam_six_validation.stage_capture_manifest import finalize_stage_capture


ROOT = Path(__file__).resolve().parents[1]


def _redactor():
    redactor = CredentialRedactor()
    redactor.add_credentials(
        username="synthetic-user/blue",
        password="synthetic-pass+marker",
        uri="rtsp://synthetic-user%2Fblue:synthetic-pass%2Bmarker@example.invalid/secret-channel",
    )
    return redactor


def _assert_markers_absent(text):
    for marker in (
        "synthetic-user/blue",
        "synthetic-user%2Fblue",
        "synthetic-pass+marker",
        "synthetic-pass%2Bmarker",
        "synthetic-user%2Fblue:synthetic-pass%2Bmarker",
        "secret-channel",
        "example.invalid",
    ):
        assert marker.lower() not in text.lower()


def test_standalone_username_and_password_are_redacted_without_losing_context():
    redactor = _redactor()
    output = redactor.sanitize(
        "camera username=synthetic-user/blue auth failed for password=synthetic-pass+marker"
    )
    _assert_markers_absent(output)
    assert "camera username=" in output
    assert "auth failed for password=" in output
    assert output.count("<redacted>") >= 2


def test_full_rtsp_uri_is_redacted_even_without_known_userinfo_format():
    redactor = _redactor()
    output = redactor.sanitize(
        "source error: RTSP://synthetic-user%2Fblue:synthetic-pass%2Bmarker@"
        "example.invalid/secret-channel?token=synthetic-pass+marker (code 401)"
    )
    _assert_markers_absent(output)
    assert "source error:" in output and "(code 401)" in output
    assert "<redacted-rtsp-uri>" in output


def test_uri_encoded_credentials_redacted_case_insensitively():
    redactor = CredentialRedactor(["operator/name", "pw+value"])
    output = redactor.sanitize("user=operator%2fname password=pw%2bvalue")
    assert "operator%2f" not in output.lower()
    assert "pw%2bvalue" not in output.lower()
    assert "user=" in output and "password=" in output
    encoded_metadata = CredentialRedactor()
    encoded_metadata.add_credentials(username="configured%2Fname", password="pw%2Bvalue")
    decoded = encoded_metadata.sanitize("user=configured/name password=pw+value")
    assert "configured/name" not in decoded
    assert "pw+value" not in decoded


def test_formatted_messages_arguments_and_exception_output_are_sanitized():
    redactor = _redactor()
    message = "warning: %s failed with %s" % (
        "synthetic-user/blue", "synthetic-pass+marker"
    )
    output = redactor.sanitize(message)
    _assert_markers_absent(output)
    args = redactor.sanitize_args([
        "--source", "rtsp://synthetic-user/blue:synthetic-pass+marker@host/path"
    ])
    _assert_markers_absent(" ".join(args))
    exception = RuntimeError(
        "child failed: rtsp://synthetic-user/blue:synthetic-pass+marker@host/path"
    )
    rendered = redactor.sanitize_exception(exception)
    _assert_markers_absent(rendered)
    assert rendered.startswith("RuntimeError:")


def test_launcher_exception_boundary_uses_credentials_loaded_before_failure(monkeypatch, capsys):
    import scripts.validate_cam_six as launcher

    def fail_after_credentials(*, person_detection, person_tracking, redactor):
        del person_detection, person_tracking
        redactor.add_credentials(
            username="synthetic-user/blue", password="synthetic-pass+marker"
        )
        raise RuntimeError("source auth failed for synthetic-user/blue")

    monkeypatch.setattr(launcher, "_main", fail_after_credentials)
    assert launcher.main() == 1
    output = capsys.readouterr().err
    _assert_markers_absent(output)
    assert "RuntimeError:" in output


def test_argument_errors_are_redacted(capsys):
    import pytest
    from scripts.validate_cam_six import _RedactingArgumentParser

    parser = _RedactingArgumentParser(redactor=_redactor(), prog="validator")
    parser.add_argument("--mode", choices=["safe"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "rtsp://synthetic-user/blue:synthetic-pass+marker@host/path"])
    output = capsys.readouterr().err
    _assert_markers_absent(output)
    assert "invalid choice" in output


def test_child_stdout_and_stderr_are_sanitized_before_display_or_persistence():
    redactor = _redactor()
    script = (
        "import sys; "
        "print('stdout user=synthetic-user/blue'); "
        "print('stderr password=synthetic-pass+marker', file=sys.stderr)"
    )
    child = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    sanitized = redactor.sanitize_child_output(child.stdout + child.stderr)
    _assert_markers_absent(sanitized)
    assert "stdout user=" in sanitized and "stderr password=" in sanitized


def test_launcher_subprocess_boundary_filters_auth_env_and_sanitizes_both_streams(
    monkeypatch, capsys
):
    import scripts.validate_cam_six as launcher

    monkeypatch.setenv("CAM_02_RTSP_USERNAME", "synthetic-user/blue")
    monkeypatch.setenv("CAM_02_RTSP_PASSWORD", "synthetic-pass+marker")
    redactor = _redactor()
    script = (
        "import os,sys; "
        "print('stdout username=synthetic-user/blue'); "
        "print('stderr password=synthetic-pass+marker', file=sys.stderr); "
        "print('credential env=' + str(any('RTSP_USERNAME' in k or 'RTSP_PASSWORD' in k "
        "for k in os.environ)))"
    )
    result = launcher._run_sanitized([sys.executable, "-c", script], redactor)
    shown = capsys.readouterr()
    _assert_markers_absent(result.stdout + result.stderr + shown.out + shown.err)
    assert "credential env=False" in shown.out
    assert "stdout username=" in shown.out
    assert "stderr password=" in shown.err


def test_manifest_serialization_sanitizes_records_and_ffprobe_stderr(tmp_path, monkeypatch):
    import scripts.cam_six_validation.stage_capture_manifest as module

    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    (capture_dir / "stage-frames.jsonl").write_text(json.dumps({
        "kind": "diagnostic",
        "detail": "rtsp://synthetic-user/blue:synthetic-pass+marker@host/secret-channel",
    }) + "\n")
    recording = tmp_path / "input.mkv"
    recording.write_bytes(b"synthetic")
    monkeypatch.setattr(module.shutil, "which", lambda name: "/synthetic/" + name)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=("probe error rtsp://synthetic-user/blue:synthetic-pass+marker@"
                    "host/secret-channel"),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    path = finalize_stage_capture(capture_dir, recording, redact_text=_redactor().sanitize)
    serialized = path.read_text()
    _assert_markers_absent(serialized)
    assert "probe error" in serialized
    assert json.loads(serialized)["decoded_mkv_status"]["status"] == "ffprobe_failed"


def test_manifest_default_sanitizer_masks_rtsp_uri(tmp_path, monkeypatch):
    import scripts.cam_six_validation.stage_capture_manifest as module

    capture_dir = tmp_path / "captures-default"
    capture_dir.mkdir()
    (capture_dir / "stage-frames.jsonl").write_text(json.dumps({
        "kind": "diagnostic",
        "detail": "connection error rtsp://default-user:default-pass@host/private",
    }) + "\n")
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    result = finalize_stage_capture(capture_dir, tmp_path / "missing.mkv")
    serialized = result.read_text()
    for marker in ("default-user", "default-pass", "host/private"):
        assert marker not in serialized
    assert "connection error" in serialized


def test_empty_and_malformed_metadata_are_safe_and_do_not_remove_unrelated_text():
    redactor = CredentialRedactor([None, "", 7, object()])
    assert redactor.sanitize("decoder warning: frame timeout 17") == (
        "decoder warning: frame timeout 17"
    )
    output = redactor.sanitize("bad URI rtsp://broken-%ZZ/chan error=EINVAL")
    assert "<redacted-rtsp-uri>" in output
    assert "error=EINVAL" in output


def test_existing_password_and_uri_redaction_behavior_is_preserved():
    redactor = CredentialRedactor(["old-password", "rtsp://old-user:old-password@host/path"])
    output = redactor.sanitize(
        "connection failed old-password at rtsp://old-user:old-password@host/path code=503"
    )
    assert "old-password" not in output
    assert "old-user" not in output
    assert "code=503" in output


def test_basic_auth_base64_representation_is_redacted():
    import base64

    redactor = CredentialRedactor()
    redactor.add_credentials(username="synthetic-user", password="synthetic-password")
    encoded = base64.b64encode(
        b"synthetic-user:synthetic-password"
    ).decode("ascii")
    output = redactor.sanitize("WWW-Authenticate Basic " + encoded)
    assert encoded not in output
    assert "WWW-Authenticate Basic" in output


def test_launcher_uses_redaction_for_python_child_and_native_username():
    launcher = (ROOT / "scripts/validate_cam_six.py").read_text()
    native = (ROOT / "scripts/cam_six_validation/main.cpp").read_text()
    assert "redactor.add_credentials(" in launcher
    assert "username=camera.username" in launcher
    assert "redactor.sanitize_child_output(line)" in launcher
    assert "redactor.sanitize_exception(exc)" in launcher
    assert "redact_text=redactor.sanitize" in launcher
    assert '"--log-driver=none"' in launcher
    assert "credentials.push_back(s->username)" in native
    assert "credential_redaction::sanitize(prefix + \" \" + detail, credentials)" in native
