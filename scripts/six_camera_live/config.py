"""Compatibility alias: canonical implementation moved to services.gpu.runtime.

Keep module identity to preserve existing imports and test monkeypatch targets.
New code must import services.gpu.runtime.live_config directly.
"""
import sys
from services.gpu.runtime import live_config as _canonical

sys.modules[__name__] = _canonical
