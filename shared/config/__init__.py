"""Configuration helpers used by the current Core v1 services."""
from functools import lru_cache
from pathlib import Path
import os,re

import yaml

PROJECT_ROOT=Path(__file__).resolve().parents[2]
CONFIG_ROOT=PROJECT_ROOT/"config"

@lru_cache(maxsize=None)
def load_yaml(name):
    path=CONFIG_ROOT/name
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

_ENV=re.compile(r"^\$\{([A-Z0-9_]+)(?::-(.*))?\}$")
def _expand(value):
    if isinstance(value,dict):return {key:_expand(item) for key,item in value.items()}
    if isinstance(value,list):return [_expand(item) for item in value]
    if isinstance(value,str):
        match=_ENV.match(value)
        if match:return os.getenv(match.group(1),match.group(2) or "")
    return value

def camera_config():
    base=_expand(load_yaml("cameras.yaml"));local_path=CONFIG_ROOT/"cameras.local.yaml"
    if not local_path.exists():return base
    local=_expand(yaml.safe_load(local_path.read_text(encoding="utf-8")) or {})
    defaults=local.get("defaults",{});overrides={str(item["id"]):item for item in local.get("cameras",[]) if item.get("id")}
    return {**base,"cameras":[{**item,**defaults,**overrides.get(str(item.get("id")),{})} for item in base.get("cameras",[])]}
