from __future__ import annotations


def bounded_center_prediction(
    box: tuple[float, float, float, float],
    velocity: tuple[float, float, float, float],
    age_sec: float,
    *,
    max_predict_sec: float = 0.20,
    max_dx_width_frac: float = 0.20,
    max_dy_height_frac: float = 0.12,
) -> tuple[float, float, float, float]:
    """Predict only bbox position for a very short visual interval.

    Tracker V4 already smooths bbox size. Re-predicting width/height in the Qt viewer
    made arm motion and sparse 2 Hz detector updates look like a breathing/overshooting
    box. V5 keeps size fixed and applies only a tightly bounded center translation.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    vx, vy, _vw, _vh = (float(v) for v in velocity)
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    dt = max(0.0, min(float(max_predict_sec), float(age_sec)))

    dx = vx * dt
    dy = vy * dt
    max_dx = max_dx_width_frac * width
    max_dy = max_dy_height_frac * height
    dx = max(-max_dx, min(max_dx, dx))
    dy = max(-max_dy, min(max_dy, dy))

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = x2 + dx
    ny2 = y2 + dy

    # Preserve size while shifting the complete box back inside normalized frame space.
    if nx1 < 0.0:
        nx2 -= nx1
        nx1 = 0.0
    if nx2 > 1.0:
        shift = nx2 - 1.0
        nx1 -= shift
        nx2 = 1.0
    if ny1 < 0.0:
        ny2 -= ny1
        ny1 = 0.0
    if ny2 > 1.0:
        shift = ny2 - 1.0
        ny1 -= shift
        ny2 = 1.0

    return (
        max(0.0, min(1.0, nx1)),
        max(0.0, min(1.0, ny1)),
        max(0.0, min(1.0, nx2)),
        max(0.0, min(1.0, ny2)),
    )


def visual_track_is_fresh(age_sec: float, *, max_age_sec: float = 1.20) -> bool:
    return 0.0 <= float(age_sec) <= float(max_age_sec)
