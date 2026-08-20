"""Configuration helpers used by the current Core v1 services.

Machine-local secrets live in the gitignored project-root ``.env``. The project
has used two camera-config schemas over time (``source/display_source`` and
``uri/enabled``). Normalizing them here keeps the current branch compatible with
both without changing the user's machine-local camera file.
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

    Existing process environment always wins. This matches the old shell launchers
    that sourced ``.env`` before starting the Python process.
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


# These are only fallbacks for the known six-camera installation. Explicit codec
# fields in cameras.yaml/cameras.local.yaml always win.
_CODEC_FALLBACK = {
    "CAM-01": "h264",
    "CAM-02": "h264",
    "CAM-03": "h264",
    "CAM-04": "h265",
    "CAM-05": "h265",
    "CAM-06": "h265",
}


def _normalize_camera(item: dict) -> dict:
    """Return one canonical camera row accepted by the current capture stack.

    Supported historical shapes include:
      - id + source/display_source + online
      - id + uri + enabled
      - camera_id + uri

    No secret or RTSP address is logged here.
    """

    camera = dict(item or {})
    camera_id = str(camera.get("id") or camera.get("camera_id") or "").strip()
    if camera_id:
        camera["id"] = camera_id

    if "online" not in camera:
        camera["online"] = bool(camera.get("enabled", True))

    source = str(
        camera.get("display_source")
        or camera.get("source")
        or camera.get("uri")
        or camera.get("ai_source")
        or ""
    ).strip()
    if source:
        camera["source"] = source
        camera.setdefault("display_source", source)
        camera.setdefault("ai_source", str(camera.get("uri") or source))

    codec = str(
        camera.get("display_codec")
        or camera.get("codec")
        or camera.get("ai_codec")
        or _CODEC_FALLBACK.get(camera_id, "h264")
    ).strip().lower()
    if codec == "hevc":
        codec = "h265"
    camera["codec"] = codec
    camera.setdefault("display_codec", codec)
    camera.setdefault("ai_codec", codec)

    # Older camera files carried credentials on every row; newer minimal files
    # may only contain uri. Use the same project-level secrets in that case.
    if not camera.get("username"):
        camera["username"] = os.getenv("SURVEILLANCE_RTSP_USERNAME", "")
    if not camera.get("password"):
        camera["password"] = os.getenv("SURVEILLANCE_RTSP_PASSWORD", "")

    return camera


def camera_config():
    base = _expand(load_yaml("cameras.yaml"))
    local_path = CONFIG_ROOT / "cameras.local.yaml"

    base_rows = [dict(item) for item in base.get("cameras", []) if isinstance(item, dict)]
    if local_path.exists():
        local = _expand(yaml.safe_load(local_path.read_text(encoding="utf-8")) or {})
        defaults = local.get("defaults", {})
        overrides = {
            str(item.get("id") or item.get("camera_id")): item
            for item in local.get("cameras", [])
            if isinstance(item, dict) and (item.get("id") or item.get("camera_id"))
        }
        merged = []
        for item in base_rows:
            cid = str(item.get("id") or item.get("camera_id") or "")
            merged.append({**item, **defaults, **overrides.get(cid, {})})
        base_rows = merged

    return {
        **base,
        "cameras": [_normalize_camera(item) for item in base_rows],
    }
