"""Compatibility import for the production camera configuration.

New code must import from services.camera_v2.config directly. This module exists
only until the remaining baseline modules are migrated away from the historical
ml_service package.
"""

from services.camera_v2.config import CameraConfig, DeepStreamConfig, Settings, load_settings

__all__ = ["CameraConfig", "DeepStreamConfig", "Settings", "load_settings"]
