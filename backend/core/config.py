import os
import copy
import yaml


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
PROJECT_YAML = os.path.join(CONFIG_DIR, "project.yaml")


class ConfigService:
    def __init__(self, path: str = PROJECT_YAML):
        self.path = path
        self.data = {}
        self.load()

    def load(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            self.data = {}
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f) or {}
        except Exception as e:
            print("[Config] load error:", e)
            self.data = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print("[Config] save error:", e)

    def get(self, dotted_key: str, default=None):
        node = self.data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted_key: str, value):
        parts = dotted_key.split(".")
        node = self.data

        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]

        node[parts[-1]] = value

    def section(self, key: str) -> dict:
        value = self.get(key, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    # ---------------- cameras.yaml ----------------
    def cameras_path(self) -> str:
        rel = self.get("cameras_config", "config/cameras.yaml")
        if os.path.isabs(rel):
            return rel
        return os.path.join(BASE_DIR, rel)

    def load_cameras(self):
        path = self.cameras_path()

        if not os.path.exists(path):
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("cameras", [])
        except Exception as e:
            print("[Config] load_cameras error:", e)
            return []

    def save_cameras(self, cameras: list):
        path = self.cameras_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"cameras": cameras}, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print("[Config] save_cameras error:", e)