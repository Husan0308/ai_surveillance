from __future__ import annotations


def should_hold_last_good(updated_at: float, now: float, hold_ms: float) -> bool:
    age_ms = max(0.0, (float(now) - float(updated_at)) * 1000.0)
    return age_ms <= max(0.0, float(hold_ms))
