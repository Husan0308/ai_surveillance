"""Compatibility CLI/import alias for the DeepStream 9.1 GPU live launcher.

New code should use "python -m services.gpu.runtime.live_launcher".
Old command and monkeypatch-based tests remain supported.
"""
import sys
from services.gpu.runtime import live_launcher as _canonical

if __name__ == "__main__":
    raise SystemExit(_canonical.main())
sys.modules[__name__] = _canonical
