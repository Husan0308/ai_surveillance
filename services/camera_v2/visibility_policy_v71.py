from __future__ import annotations


def should_hold_last_good(
    updated_at: float,
    now: float,
    hold_ms: float,
) -> bool:
    """Keep a last-good NvDCF box briefly when one tracker batch has no output.

    This is visibility hysteresis only: it never predicts or changes bbox coordinates.
    The original last-good timestamp is preserved so a stale box cannot live forever.
    """
    age_ms = max(0.0, (float(now) - float(updated_at)) * 1000.0)
    return age_ms <= max(0.0, float(hold_ms))
