#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _fit_display(frame: np.ndarray, max_height: int = 650) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_height / float(h))
    if scale >= 0.999:
        return frame.copy(), 1.0
    return (
        cv2.resize(
            frame,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def _draw_point(
    canvas: np.ndarray,
    point: tuple[float, float],
    label: str,
    *,
    scale: float,
    offset_x: int,
    validation: bool,
) -> None:
    x = int(round(point[0] * scale)) + offset_x
    y = int(round(point[1] * scale))
    radius = 7 if validation else 6
    cv2.circle(canvas, (x, y), radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), radius + 3, (0, 0, 0), 2, cv2.LINE_AA)
    if validation:
        cv2.drawMarker(canvas, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (x + 10, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (x + 10, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _interactive_pairs(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    camera_a: str,
    camera_b: str,
    control_count: int,
    validation_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    disp_a, scale_a = _fit_display(frame_a)
    disp_b, scale_b = _fit_display(frame_b)
    height = max(disp_a.shape[0], disp_b.shape[0])
    if disp_a.shape[0] < height:
        disp_a = np.vstack(
            (disp_a, np.zeros((height - disp_a.shape[0], disp_a.shape[1], 3), dtype=np.uint8))
        )
    if disp_b.shape[0] < height:
        disp_b = np.vstack(
            (disp_b, np.zeros((height - disp_b.shape[0], disp_b.shape[1], 3), dtype=np.uint8))
        )

    width_a = disp_a.shape[1]
    base = np.hstack((disp_a, disp_b))
    pairs_a: list[tuple[float, float]] = []
    pairs_b: list[tuple[float, float]] = []
    pending_a: tuple[float, float] | None = None
    total = control_count + validation_count
    window = f"V11 mesh calibration {camera_a} <-> {camera_b}"

    def mouse(event, x, y, _flags, _userdata):
        nonlocal pending_a
        if event != cv2.EVENT_LBUTTONDOWN or len(pairs_a) >= total:
            return
        if pending_a is None:
            if x >= width_a:
                return
            pending_a = (x / scale_a, y / scale_a)
        else:
            if x < width_a:
                return
            pairs_a.append(pending_a)
            pairs_b.append(((x - width_a) / scale_b, y / scale_b))
            pending_a = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            canvas = base.copy()
            for idx, point in enumerate(pairs_a):
                is_validation = idx >= control_count
                label = f"V{idx-control_count+1}" if is_validation else f"C{idx+1}"
                _draw_point(
                    canvas,
                    point,
                    label,
                    scale=scale_a,
                    offset_x=0,
                    validation=is_validation,
                )
                _draw_point(
                    canvas,
                    pairs_b[idx],
                    label,
                    scale=scale_b,
                    offset_x=width_a,
                    validation=is_validation,
                )

            if pending_a is not None:
                px = int(round(pending_a[0] * scale_a))
                py = int(round(pending_a[1] * scale_a))
                cv2.circle(canvas, (px, py), 10, (255, 255, 255), 2, cv2.LINE_AA)

            done = len(pairs_a)
            if done < control_count:
                phase = f"CONTROL {done+1}/{control_count}"
            elif done < total:
                phase = f"VALIDATION {done-control_count+1}/{validation_count}"
            else:
                phase = "READY TO SOLVE"
            expected = camera_a if pending_a is None else camera_b
            text = (
                f"{phase} | click SAME FLOOR point: now {expected} | "
                "ENTER=solve U=undo R=reset Q=quit"
            )
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.line(canvas, (width_a, 0), (width_a, height), (255, 255, 255), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt
            if key in (ord("r"), ord("R")):
                pairs_a.clear()
                pairs_b.clear()
                pending_a = None
            elif key in (ord("u"), ord("U")):
                if pending_a is not None:
                    pending_a = None
                elif pairs_a:
                    pairs_a.pop()
                    pairs_b.pop()
            elif key in (10, 13) and pending_a is None and len(pairs_a) == total:
                break
    finally:
        cv2.destroyWindow(window)

    a = np.asarray(pairs_a, dtype=np.float32)
    b = np.asarray(pairs_b, dtype=np.float32)
    return a[:control_count], b[:control_count], a[control_count:], b[control_count:]


def _nearest_index(points: np.ndarray, xy: np.ndarray, max_distance: float = 2.5) -> int | None:
    distances = np.linalg.norm(points - xy.reshape(1, 2), axis=1)
    idx = int(np.argmin(distances))
    return idx if float(distances[idx]) <= max_distance else None


def _build_mesh(
    src: np.ndarray,
    dst: np.ndarray,
    image_size: tuple[int, int],
) -> list[dict[str, object]]:
    width, height = image_size
    subdiv = cv2.Subdiv2D((0, 0, int(width), int(height)))
    for x, y in src:
        x = min(max(float(x), 0.001), width - 1.001)
        y = min(max(float(y), 0.001), height - 1.001)
        subdiv.insert((x, y))

    seen: set[tuple[int, int, int]] = set()
    triangles: list[dict[str, object]] = []
    for raw in subdiv.getTriangleList():
        coords = np.asarray(raw, dtype=np.float32).reshape(3, 2)
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= width):
            continue
        if np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= height):
            continue
        indices: list[int] = []
        for vertex in coords:
            idx = _nearest_index(src, vertex)
            if idx is None:
                indices = []
                break
            indices.append(idx)
        if len(indices) != 3 or len(set(indices)) != 3:
            continue
        key = tuple(sorted(indices))
        if key in seen:
            continue
        seen.add(key)
        src_tri = src[list(key)].astype(np.float32)
        dst_tri = dst[list(key)].astype(np.float32)
        area = abs(float(np.cross(src_tri[1] - src_tri[0], src_tri[2] - src_tri[0]))) / 2.0
        if area < 25.0:
            continue
        matrix = cv2.getAffineTransform(src_tri, dst_tri)
        triangles.append(
            {
                "indices": list(key),
                "src": src_tri,
                "dst": dst_tri,
                "matrix": matrix,
            }
        )
    if not triangles:
        raise RuntimeError("mesh_has_no_triangles")
    return triangles


def _barycentric(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    a, b, c = tri.astype(np.float64)
    v0 = b - a
    v1 = c - a
    v2 = point.astype(np.float64) - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-9:
        return None
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    w = 1.0 - u - v
    return np.asarray([w, u, v], dtype=np.float64)


def _project_mesh(point: np.ndarray, mesh: list[dict[str, object]]) -> np.ndarray | None:
    best: tuple[float, np.ndarray] | None = None
    for row in mesh:
        src_tri = np.asarray(row["src"], dtype=np.float32)
        bary = _barycentric(point, src_tri)
        if bary is None:
            continue
        minimum = float(np.min(bary))
        if minimum < -1e-5:
            continue
        matrix = np.asarray(row["matrix"], dtype=np.float64)
        projected = matrix[:, :2] @ point.astype(np.float64) + matrix[:, 2]
        if best is None or minimum > best[0]:
            best = (minimum, projected.astype(np.float32))
    return None if best is None else best[1]


def _validate(
    validation_src: np.ndarray,
    validation_dst: np.ndarray,
    mesh: list[dict[str, object]],
    dst_image_size: tuple[int, int],
) -> tuple[list[dict[str, object]], np.ndarray]:
    diag = math.hypot(float(dst_image_size[0]), float(dst_image_size[1]))
    rows: list[dict[str, object]] = []
    errors_pct: list[float] = []
    for idx, (src, expected) in enumerate(zip(validation_src, validation_dst), start=1):
        projected = _project_mesh(src, mesh)
        if projected is None:
            rows.append(
                {
                    "index": idx,
                    "inside_mesh": False,
                    "error_px": None,
                    "error_pct_diag": None,
                }
            )
            continue
        error_px = float(np.linalg.norm(projected - expected))
        error_pct = 100.0 * error_px / diag
        errors_pct.append(error_pct)
        rows.append(
            {
                "index": idx,
                "inside_mesh": True,
                "projected": projected.astype(float).tolist(),
                "expected": expected.astype(float).tolist(),
                "error_px": error_px,
                "error_pct_diag": error_pct,
            }
        )
    return rows, np.asarray(errors_pct, dtype=np.float64)


def _serializable_mesh(mesh: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "indices": list(row["indices"]),
            "src": np.asarray(row["src"]).astype(float).tolist(),
            "dst": np.asarray(row["dst"]).astype(float).tolist(),
            "affine_2x3": np.asarray(row["matrix"]).astype(float).tolist(),
        }
        for row in mesh
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Piecewise affine floor mesh calibration for one same-room camera pair."
    )
    parser.add_argument("--camera-a", default="CAM-01")
    parser.add_argument("--camera-b", default="CAM-04")
    parser.add_argument("--frame-a", required=True)
    parser.add_argument("--frame-b", required=True)
    parser.add_argument("--control-points", type=int, default=12)
    parser.add_argument("--validation-points", type=int, default=4)
    parser.add_argument(
        "--output",
        default="artifacts/calibration/devs_cam01_cam04_floor_mesh_v2.json",
    )
    parser.add_argument("--pass-median-pct", type=float, default=1.0)
    parser.add_argument("--pass-p95-pct", type=float, default=2.0)
    args = parser.parse_args()

    control_count = max(8, int(args.control_points))
    validation_count = max(3, int(args.validation_points))
    frame_a = cv2.imread(str(Path(args.frame_a).expanduser()), cv2.IMREAD_COLOR)
    frame_b = cv2.imread(str(Path(args.frame_b).expanduser()), cv2.IMREAD_COLOR)
    if frame_a is None or frame_b is None:
        print("V11_CAM_PAIR_MESH RESULT=FAIL reason=frame_read", flush=True)
        return 2

    print(
        "V11_CAM_PAIR_MESH READY "
        f"pair={args.camera_a},{args.camera_b} controls={control_count} "
        f"validation={validation_count} mode=piecewise-affine-floor-mesh canonical={args.camera_a}",
        flush=True,
    )
    print(
        "V11_CAM_PAIR_MESH_INSTRUCTION "
        "controls=spread_across_shared_floor validation=inside_control_hull "
        "click_order=left_then_right_same_physical_floor_point",
        flush=True,
    )

    control_a, control_b, validation_a, validation_b = _interactive_pairs(
        frame_a,
        frame_b,
        args.camera_a,
        args.camera_b,
        control_count,
        validation_count,
    )

    size_a = (int(frame_a.shape[1]), int(frame_a.shape[0]))
    size_b = (int(frame_b.shape[1]), int(frame_b.shape[0]))
    mesh_b_to_a = _build_mesh(control_b, control_a, size_b)
    mesh_a_to_b = _build_mesh(control_a, control_b, size_a)

    forward_rows, forward_pct = _validate(validation_b, validation_a, mesh_b_to_a, size_a)
    reverse_rows, reverse_pct = _validate(validation_a, validation_b, mesh_a_to_b, size_b)
    forward_inside = sum(int(bool(row["inside_mesh"])) for row in forward_rows)
    reverse_inside = sum(int(bool(row["inside_mesh"])) for row in reverse_rows)
    required_inside = validation_count

    combined = np.concatenate((forward_pct, reverse_pct)) if forward_pct.size or reverse_pct.size else np.asarray([], dtype=np.float64)
    median_pct = float(np.median(combined)) if combined.size else float("inf")
    p95_pct = float(np.percentile(combined, 95)) if combined.size else float("inf")
    max_pct = float(np.max(combined)) if combined.size else float("inf")
    passed = bool(
        forward_inside == required_inside
        and reverse_inside == required_inside
        and median_pct <= float(args.pass_median_pct)
        and p95_pct <= float(args.pass_p95_pct)
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "mode": "piecewise_affine_floor_mesh",
        "canonical_camera": args.camera_a,
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "image_size_a": list(size_a),
        "image_size_b": list(size_b),
        "control_points_a": control_a.astype(float).tolist(),
        "control_points_b": control_b.astype(float).tolist(),
        "validation_points_a": validation_a.astype(float).tolist(),
        "validation_points_b": validation_b.astype(float).tolist(),
        "mesh_b_to_a": _serializable_mesh(mesh_b_to_a),
        "mesh_a_to_b": _serializable_mesh(mesh_a_to_b),
        "validation_b_to_a": forward_rows,
        "validation_a_to_b": reverse_rows,
        "metrics": {
            "control_count": control_count,
            "validation_count": validation_count,
            "triangles_b_to_a": len(mesh_b_to_a),
            "triangles_a_to_b": len(mesh_a_to_b),
            "forward_inside": forward_inside,
            "reverse_inside": reverse_inside,
            "median_error_pct_diag": median_pct,
            "p95_error_pct_diag": p95_pct,
            "max_error_pct_diag": max_pct,
        },
        "pass_thresholds": {
            "all_validation_inside_mesh": True,
            "max_median_pct_diag": float(args.pass_median_pct),
            "max_p95_pct_diag": float(args.pass_p95_pct),
        },
        "passed": passed,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        "V11_CAM_PAIR_MESH_METRICS "
        f"controls={control_count} validation={validation_count} "
        f"triangles_fwd={len(mesh_b_to_a)} triangles_rev={len(mesh_a_to_b)} "
        f"inside_fwd={forward_inside}/{validation_count} inside_rev={reverse_inside}/{validation_count} "
        f"median={median_pct:.3f}%diag p95={p95_pct:.3f}%diag max={max_pct:.3f}%diag",
        flush=True,
    )
    for row in forward_rows:
        print(
            "V11_CAM_PAIR_MESH_VALIDATION "
            f"direction={args.camera_b}_to_{args.camera_a} index={row['index']} "
            f"inside={int(bool(row['inside_mesh']))} "
            f"error_px={row['error_px'] if row['error_px'] is not None else -1:.2f} "
            f"error_pct={row['error_pct_diag'] if row['error_pct_diag'] is not None else -1:.3f}",
            flush=True,
        )
    print(
        f"V11_CAM_PAIR_MESH RESULT={'PASS' if passed else 'FAIL'} output={output}",
        flush=True,
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
