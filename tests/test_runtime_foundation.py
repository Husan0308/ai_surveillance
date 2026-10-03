"""F1 tests never launch services, connect Kafka, or open camera streams."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from services.shared.runtime_python import preflight_python, runtime_python

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_foundation", ROOT / "scripts/runtime_foundation.py")
foundation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foundation)
read_pins = foundation.pins


class InterpreterResolutionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = self.root / ".runtime/full-stack-venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text("#!/bin/sh\nexit 0\n")
        self.python.chmod(0o755)
        self.clean_env = patch.dict(os.environ, {}, clear=True)
        self.clean_env.start()
        self.addCleanup(self.clean_env.stop)

    def test_all_roles_default_to_one_project_python(self):
        for role in ("full", "identity", "kafka"):
            self.assertEqual(runtime_python(self.root, role), self.python)

    def test_missing_runtime_fails_without_fallback(self):
        self.python.unlink()
        with self.assertRaisesRegex(ValueError, "setup_runtime"):
            runtime_python(self.root)

    def test_full_override_is_used_by_all_roles(self):
        other = self.root / "override"
        other.symlink_to(self.python)
        with patch.dict(os.environ, FULL_STACK_PYTHON=str(other)):
            for role in ("full", "identity", "kafka"):
                self.assertEqual(runtime_python(self.root, role), other)

    def test_identity_override_takes_precedence(self):
        other = self.root / "identity-python"
        other.symlink_to(self.python)
        with patch.dict(os.environ, MV3DT_OSNET_PYTHON=str(other)):
            self.assertEqual(runtime_python(self.root, "identity"), other)
            self.assertEqual(runtime_python(self.root, "kafka"), self.python)

    def test_invalid_override_never_falls_back(self):
        with patch.dict(os.environ, FULL_STACK_PYTHON="python3"):
            with self.assertRaisesRegex(ValueError, "absolute"):
                runtime_python(self.root)
        with patch.dict(os.environ, MV3DT_OSNET_PYTHON=str(self.root / "missing")):
            with self.assertRaisesRegex(ValueError, "missing"):
                runtime_python(self.root, "identity")

    def test_trash_path_and_trash_symlink_are_rejected(self):
        trash = self.root / ".local/share" / "Trash" / "bin/python"
        trash.parent.mkdir(parents=True)
        trash.write_text("#!/bin/sh\nexit 0\n")
        trash.chmod(0o755)
        alias = self.root / "alias"
        alias.symlink_to(trash)
        for value in (trash, alias):
            with patch.dict(os.environ, MV3DT_OSNET_PYTHON=str(value)):
                with self.assertRaisesRegex(ValueError, "Trash"):
                    runtime_python(self.root, "identity")

    def test_preflight_uses_isolated_python_and_propagates_failure(self):
        with patch("services.shared.runtime_python.subprocess.run") as run:
            self.assertEqual(preflight_python(self.root, "identity"), self.python)
            args = run.call_args.args[0]
            self.assertEqual(args[1], "-I")
            self.assertEqual(args[-1], "identity")
            self.assertTrue(run.call_args.kwargs["check"])
            run.side_effect = subprocess.CalledProcessError(1, args)
            with self.assertRaises(subprocess.CalledProcessError):
                preflight_python(self.root, "identity")


class ShellResolutionTest(InterpreterResolutionTest):
    def shell(self, env=None):
        return subprocess.run(["/bin/bash", "-c",
            'ROOT="$1"; source "$2"; PY="$(resolve_python)" || exit 1; printf "%s\\n" "$PY"',
            "test", str(self.root), str(ROOT / "scripts/runtime_python.sh")],
            env={"PATH": "/usr/bin:/bin", **(env or {})}, capture_output=True, text=True)

    def test_helper_stdout_is_only_path(self):
        self.python.write_text("#!/bin/sh\necho 'checking imports' >&2\necho 'preflight JSON'\n")
        result = self.shell()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(self.python) + "\n")
        self.assertIn("checking imports", result.stderr)

    def test_missing_runtime_does_not_bootstrap(self):
        self.python.unlink()
        result = self.shell()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("setup_runtime", result.stderr)
        self.assertFalse(self.python.exists())

    def test_missing_required_modules_fail_closed(self):
        for module in ("gi(Gst)", "PySide6"):
            self.python.write_text(f"#!/bin/sh\necho 'missing {module}' >&2\nexit 1\n")
            result = self.shell()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn(module, result.stderr)
            self.assertIn("no fallback", result.stderr)

    def test_explicit_override_and_invalid_override(self):
        alternate = self.root / "alternate-python"
        alternate.symlink_to(self.python)
        result = self.shell({"FULL_STACK_PYTHON": str(alternate), "VIRTUAL_ENV": "/bad"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(alternate) + "\n")
        result = self.shell({"FULL_STACK_PYTHON": str(self.root / "missing")})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_preflight_exit_precedes_camera_processes(self):
        launcher = (ROOT / "scripts/start_full_live_stack.sh").read_text()
        self.assertLess(launcher.index('if [[ "${1:-}" == "--preflight" ]]'), launcher.index("mkdir -p"))
        self.assertNotIn("bootstrap_full_stack_python", launcher)
        self.assertIn("unset PYTHONHOME PYTHONPATH", launcher)


class ImportOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / "venv"
        self.prefix.mkdir()
        self.stack = [patch.object(foundation.sys, "prefix", str(self.prefix)),
                      patch.object(foundation.sys, "base_prefix", "/usr"),
                      patch.object(foundation.site, "ENABLE_USER_SITE", False),
                      patch.object(foundation, "pins", return_value={}),
                      patch.object(foundation, "GPU_VERSIONS", {})]
        for mock in self.stack:
            mock.start()
            self.addCleanup(mock.stop)

    def test_missing_gi_or_pyside_import_is_reported(self):
        for name in ("gi", "PySide6"):
            with patch.dict(foundation.ROLE_MODULES, identity=(name,)), \
                 patch.object(foundation.importlib, "import_module", side_effect=ImportError(f"No module named {name}")):
                with self.assertRaisesRegex(ValueError, name):
                    foundation.check(self.root, "identity")

    def test_undeclared_external_module_origin_is_rejected(self):
        with patch.dict(foundation.ROLE_MODULES, identity=("numpy",)), \
             patch.object(foundation.importlib, "import_module", return_value=SimpleNamespace(__file__="/other/venv/numpy/__init__.py")):
            with self.assertRaisesRegex(ValueError, "undeclared dependency origin"):
                foundation.check(self.root, "identity")

    def test_user_site_is_rejected(self):
        with patch.object(foundation.site, "ENABLE_USER_SITE", True):
            with self.assertRaisesRegex(ValueError, "user-site"):
                foundation.check(self.root)

    def test_platform_metadata_drift_is_rejected(self):
        target = self.root / "platform/METADATA"
        target.parent.mkdir()
        target.write_text("old")
        (self.prefix / "platform-libs.json").write_text(json.dumps({"links": {},
            "metadata": {str(target): foundation.sha(target)}}))
        target.write_text("new")
        with self.assertRaisesRegex(ValueError, "metadata drift"):
            foundation.platform_check(self.prefix)

    def test_platform_link_drift_is_rejected(self):
        link = self.prefix / "torch"
        (self.prefix / "platform-libs.json").write_text(json.dumps({
            "links": {str(link): str(self.root / "original")}, "metadata": {}}))
        with self.assertRaisesRegex(ValueError, "link drift"):
            foundation.platform_check(self.prefix)

    def test_setup_valid_runtime_is_idempotent_without_pip_or_venv(self):
        prefix = self.root / ".runtime/full-stack-venv"
        python = prefix / "bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("fixture")
        with patch.object(foundation.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(foundation.setup(self.root, None), python)
            self.assertEqual(run.call_count, 1)
            self.assertIn("check", run.call_args.args[0])

    def test_no_active_trash_runtime_paths(self):
        # Search source, not historical docs/evidence or test fixtures.
        for folder in ("services", "scripts"):
            for path in (ROOT / folder).rglob("*"):
                if path.suffix in (".py", ".sh"):
                    self.assertNotIn("/.local/share/" + "Trash/", path.read_text(), str(path))

    def test_lock_never_installs_nvidia_packages(self):
        for name in read_pins(ROOT):
            self.assertFalse(name.startswith(("nvidia", "cuda", "tensorrt", "pygobject", "torch==")))
        source = (ROOT / "scripts/runtime_foundation.py").read_text()
        self.assertIn('"--no-deps"', source)
        self.assertNotIn('"--upgrade"', source)

    def test_forbidden_platform_pin_is_rejected(self):
        folder = self.root / "requirements"
        folder.mkdir()
        for name in ("nvidia-cudnn-cu12", "cuda-bindings", "torch", "pygobject", "tensorrt"):
            (folder / "runtime-host.lock.txt").write_text(f"{name}==1.0\n")
            with self.assertRaisesRegex(ValueError, "protected platform"):
                read_pins(self.root)

    def test_incompatible_existing_runtime_is_not_overwritten(self):
        prefix = self.root / ".runtime/full-stack-venv"
        prefix.mkdir(parents=True)
        cfg = prefix / "pyvenv.cfg"
        before = "include-system-site-packages = false\nversion = 3.10.1\n"
        cfg.write_text(before)
        with patch.object(foundation.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "not replaced"):
                foundation.setup(self.root, None)
            run.assert_not_called()
        self.assertEqual(cfg.read_text(), before)

    def test_missing_runtime_setup_creates_only_one_venv(self):
        commands = []
        prefix = self.root / ".runtime/full-stack-venv"

        def fake_run(command, **kwargs):
            commands.append(command)
            if "venv" in command:
                (prefix / "bin").mkdir(parents=True)
                (prefix / "bin/python").write_text("fixture")
                (prefix / "pyvenv.cfg").write_text(
                    "include-system-site-packages = true\nversion = 3.12.3\n")
            return SimpleNamespace(returncode=0)

        with patch.object(foundation.subprocess, "run", side_effect=fake_run), \
             patch.object(foundation, "bridge_gpu") as bridge, \
             patch.object(foundation.metadata, "distributions", return_value=[]):
            self.assertEqual(foundation.setup(self.root, self.root / "gpu"), prefix / "bin/python")
            bridge.assert_called_once()
        self.assertEqual(sum("venv" in cmd for cmd in commands), 1)
        self.assertIn("--system-site-packages", commands[0])
        self.assertFalse(any("pip" in cmd for cmd in commands))

    def test_setup_repairs_only_missing_pin(self):
        prefix = self.root / ".runtime/full-stack-venv"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin/python").write_text("fixture")
        (prefix / "pyvenv.cfg").write_text("include-system-site-packages = true\nversion = 3.12.3\n")
        (prefix / "platform-libs.json").write_text("{}")
        installed = SimpleNamespace(metadata={"Name": "kept"}, version="2")
        with patch.object(foundation.subprocess, "run", side_effect=[
                SimpleNamespace(returncode=1), SimpleNamespace(returncode=0), SimpleNamespace(returncode=0)]) as run, \
             patch.object(foundation.metadata, "distributions", return_value=[installed]), \
             patch.object(foundation, "pins", return_value={"kept": "2", "added": "1"}):
            foundation.setup(self.root, None)
            pip_command = run.call_args_list[1].args[0]
            self.assertEqual(pip_command[-1], "added==1")
            self.assertIn("--no-deps", pip_command)
            self.assertNotIn("kept==2", pip_command)

    def test_role_worker_failure_precedes_room_pair_camera_setup(self):
        spec = importlib.util.spec_from_file_location("scripts.dev_room_mv3dt.run_room_pair",
            ROOT / "scripts/dev_room_mv3dt/run_room_pair.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with patch.object(runner, "preflight_python", side_effect=ValueError("missing OSNet")), \
             patch.object(runner, "camera_uris") as cameras, \
             patch.object(runner, "start_monitors") as monitors, \
             patch.object(runner, "copy_profile") as stage:
            with self.assertRaisesRegex(ValueError, "missing OSNet"):
                runner.run("live", 180, True)
            cameras.assert_not_called()
            monitors.assert_not_called()
            stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
