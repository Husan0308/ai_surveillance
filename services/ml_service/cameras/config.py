from shared.config import camera_config

def load_camera_configs(path=None) -> list[dict]:
    """Load enabled cameras from central YAML; URLs are never hardcoded."""
    if path is not None:
        raise ValueError("Camera configuration must use config/cameras.yaml")
    data = camera_config()
    return [c for c in data.get("cameras", []) if c.get("online", c.get("enabled", False))]
