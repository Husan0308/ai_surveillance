"""Single read-only configuration layer for all services."""
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT=Path(__file__).resolve().parents[2]
CONFIG_ROOT=PROJECT_ROOT/"config"


@lru_cache(maxsize=None)
def load_yaml(name):
    path=CONFIG_ROOT/name
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def project_config():
    return load_yaml("project.yaml")


def camera_config():
    return load_yaml("cameras.yaml")
