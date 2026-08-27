from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DisplaySizeState:
    width: float
    height: float
    width_expanded_at: float
    height_expanded_at: float
    seen_at: float


def stable_size(
    previous: float,
    current: float,
    last_expand_at: float,
    now: float,
    *,
    hold_sec: float,
    shrink_alpha: float,
) -> tuple[float, float]:
    """Open immediately around a larger NvDCF box; close slowly after a short hold."""
    previous = max(1.0, float(previous))
    current = max(1.0, float(current))
    if current >= previous:
        return current, float(now)
    if float(now) - float(last_expand_at) <= float(hold_sec):
        return previous, float(last_expand_at)
    value = previous + float(shrink_alpha) * (current - previous)
    return max(current, value), float(last_expand_at)


def expand_box(
    box: tuple[float, float, float, float],
    frame_width: float,
    frame_height: float,
    *,
    side_margin: float,
    top_margin: float,
    bottom_margin: float,
) -> tuple[float, float, float, float]:
    """Display-only full-body envelope around the current NvDCF rectangle."""
    x1, y1, x2, y2 = (float(v) for v in box)
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    return (
        max(0.0, x1 - width * side_margin),
        max(0.0, y1 - height * top_margin),
        min(float(frame_width - 1), x2 + width * side_margin),
        min(float(frame_height - 1), y2 + height * bottom_margin),
    )
