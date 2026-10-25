"""Backward-compatible module alias for secure RTSP diagnostic redaction."""
import sys
from services.shared import credential_redaction as _canonical

sys.modules[__name__] = _canonical
