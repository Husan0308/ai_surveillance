"""Configuration helpers used by the current Core v1 services.

Machine-local secrets live in the gitignored project-root ``.env``.  The direct
camera desktop entry is launched with ``python -m ...`` rather than through a
shell wrapper, so it must load that file itself before expanding
``${SURVEILLANCE_RTSP_*}`` placeholders.
"""

from functools import lru_cache
from pathlib import Path
import os
import re

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config"

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV = re.compile(r"^\$\{([A-Z0-9_]+)(?::-(.*))?\}$")


def _load_project_env() -> bool:
    """Load ``PROJECT_ROOT/.env`` without adding a runtime dependency.

    Existing process environment always wins.  This matters when a service is
    launched by systemd/docker with injected secrets, while making the normal
    local ``python -m`` path work exactly like the old shell launchers that used
    ``source .env``.
    """

    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return False

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _ENV_KEY.fullmatch(key):
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        # Never overwrite credentials/options explicitly supplied by the parent
        # environment.
        os.environ.setdefault(key, value)

    return True


PROJECT_ENV_LOADED = _load_project_env()


@lru_cache(maxsize=None)
def load_yaml(name):
    path = CONFIG_ROOT / name
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        match = _ENV.match(value)
        if match:
            return os.getenv(match.group(1), match.group(2) or "")
    return value


def camera_config():
    base = _expand(load_yaml("cameras.yaml"))
    local_path = CONFIG_ROOT / "cameras.local.yaml"
    if not local_path.exists():
        return base
    local = _expand(yaml.safe_load(local_path.read_text(encoding="utf-8")) or {})
    defaults = local.get("defaults", {})
    overrides = {
        str(item["id"]): item
        for item in local.get("cameras", [])
        if item.get("id")
    }
    return {
        **base,
        "cameras": [
            {**item, **defaults, **overrides.get(str(item.get("id")), {})}
            for item in base.get("cameras", [])
        ],
    }
