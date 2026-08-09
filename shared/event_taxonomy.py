"""Persistent business-event taxonomy. Realtime frame metadata is never persisted."""
CURRENT_CANONICAL=frozenset({"camera.online","camera.offline","person.identified","identity.conflict","enrollment.completed","enrollment.failed"})
LEGACY=frozenset({"person_detected","camera_online","camera_offline","unknown_detected","online","offline","system","snapshot","enrollment_completed"})
DEPRECATED=frozenset({"person.detected","unknown.detected"})
def event_type(payload):return str(payload.get("event_type") or payload.get("type") or "")
def classify(value):
    value=str(value or "")
    if value in CURRENT_CANONICAL:return "current"
    if value in LEGACY:return "legacy"
    if value in DEPRECATED:return "deprecated"
    return "unknown"
def is_persistent(payload):return event_type(payload) in CURRENT_CANONICAL
