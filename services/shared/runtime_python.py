"""Interpreter ownership only; no model, camera, or identity policy lives here."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


def runtime_python(root: Path, role: str = "full") -> Path:
    """One default, with explicit fail-closed overrides (never PATH discovery)."""
    names = {"full": "FULL_STACK_PYTHON", "identity": "MV3DT_OSNET_PYTHON",
             "kafka": "MV3DT_KAFKA_PYTHON"}
    if role not in names:
        raise ValueError(f"unknown runtime role: {role}")
    value = os.environ.get(names[role]) or os.environ.get("FULL_STACK_PYTHON")
    python = Path(value) if value else root / ".runtime/full-stack-venv/bin/python"
    if not python.is_absolute():
        raise ValueError(f"{names[role]} must be an absolute interpreter path")
    # Do not resolve away the venv's bin/python symlink before execution.
    if any(part.lower() == "trash" for part in (*python.parts, *python.resolve().parts)):
        raise ValueError("runtime interpreters cannot reside in Trash")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"runtime Python missing: {python}; run bash scripts/setup_runtime.sh")
    return python


def preflight_python(root: Path, role: str = "full") -> Path:
    python = runtime_python(root, role)
    subprocess.run([str(python), "-I", "-B", str(root / "scripts/runtime_foundation.py"),
                    "check", "--root", str(root), "--role", role],
                   check=True, stdout=subprocess.DEVNULL)
    return python
