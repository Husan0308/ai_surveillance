"""CPU-only architecture and backward-compatibility regression tests."""
from __future__ import annotations

import ast
from pathlib import Path

from services.gpu.runtime import live_config, live_launcher
from scripts.six_camera_live import config as old_config
from scripts.six_camera_live import launch as old_launcher


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_imports_are_exact_canonical_modules():
    assert old_config is live_config
    assert old_launcher is live_launcher
    assert old_launcher.main is live_launcher.main


def test_moved_launcher_locates_repository_root():
    assert live_launcher.ROOT == ROOT
    assert live_launcher.PLATFORM == ROOT / "config/deepstream-platform.json"


def test_gpu_config_remains_single_source_of_truth():
    assert old_config.CAMERAS is live_config.CAMERAS
    assert old_config.CHANNELS is live_config.CHANNELS
    assert old_config.build_app_config is live_config.build_app_config


def test_old_cli_delegates_to_canonical_runtime():
    text = (ROOT / "scripts/six_camera_live/launch.py").read_text()
    assert "from services.gpu.runtime import live_launcher as _canonical" in text
    assert "raise SystemExit(_canonical.main())" in text
    assert "sys.modules[__name__] = _canonical" in text


def test_only_gpu_runtime_owns_new_live_implementation():
    for file in (ROOT / "scripts/six_camera_live").glob("*.py"):
        if file.name == "__init__.py":
            continue
        source = file.read_text()
        assert "docker\", \"run\"" not in source
        assert "build_app_config(" not in source
    assert (ROOT / "services/gpu/runtime/live_launcher.py").is_file()


def test_gpu_runtime_does_not_import_api_ml_or_qt_frontend():
    prohibited = (
        "services.api_service", "services.ml_service",
        "services.frontend", "PySide6", "scripts",
    )
    for file in (ROOT / "services/gpu").rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            target = ""
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for target in names:
                assert not any(target == name or target.startswith(name + ".")
                               for name in prohibited), f"{file}: forbidden {target}"


def test_ml_api_frontend_have_distinct_existing_entrypoints():
    assert (ROOT / "services/ml_service/app/main.py").is_file()
    assert (ROOT / "services/api_service/app/main.py").is_file()
    assert (ROOT / "services/frontend/app/main.py").is_file()


def test_no_change_to_frozen_video_validator_layout():
    old_header = (ROOT / "scripts/cam_six_validation/camera_source_plan.hpp").read_text()
    assert "shape = {6, 3, 2, 1920, 1620};" in old_header


def test_shared_ownership_compatibility_import_and_lock_contention(tmp_path):
    from services.camera_v11.source_ownership import SourceOwnership as LegacyOwnership
    from services.shared.source_ownership import SourceOwnership

    assert LegacyOwnership is SourceOwnership
    with SourceOwnership(("CAM-01", "CAM-02"), tmp_path):
        from pytest import raises
        with raises(RuntimeError, match="another camera owner"):
            with SourceOwnership(("CAM-01",), tmp_path):
                pass
    # Existing lockfiles must remain on disk to avoid inode-based races.
    assert (tmp_path / "CAM-01.lock").exists()
    with SourceOwnership(("CAM-01",), tmp_path):
        pass


def test_gpu_launcher_acquires_existing_source_ownership_before_docker():
    file = ROOT / "services/gpu/runtime/live_launcher.py"
    source = file.read_text(encoding="utf-8")
    assert 'SourceOwnership(manifest["cameras"]' in source
    assert 'ROOT / ".runtime/camera-owner-locks"' in source
    assert "camera_ownership.__enter__()" in source
    assert "camera_ownership.__exit__(None, None, None)" in source
    assert source.index("camera_ownership.__enter__()") < source.index("proc = subprocess.Popen(")


def test_gpu_live_launcher_keeps_runtime_approval_checks():
    source = (ROOT / "services/gpu/runtime/live_launcher.py").read_text()
    for needle in (
        "args.authorize_physical_rtsp",
        "args.nvr_readonly_authorized",
        "args.network_egress_verified",
        "args.expected_engine_sha256",
        "checked_network(",
        "display_preflight()",
        "nvr_sessions_preflight(",
    ):
        assert needle in source


def test_shared_credential_redaction_is_one_implementation():
    from services.shared import credential_redaction as canonical
    from scripts.cam_six_validation import credential_redaction as legacy

    assert legacy is canonical
    redactor = canonical.CredentialRedactor()
    redactor.add_credentials(
        username="synthetic-user", password="synthetic-private-secret",
        uri="rtsp://synthetic-user:synthetic-private-secret@unit.invalid/stream",
    )
    out = redactor.sanitize("synthetic-private-secret")
    assert "synthetic-private-secret" not in out
