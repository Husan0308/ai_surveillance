#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_cameras(path: Path) -> dict[str, dict[str, object]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(row["id"]): row for row in raw.get("cameras", []) if row.get("id")}


def _capture_latest(uri: str, *, timeout_sec: float = 5.0) -> np.ndarray:
    cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot_open_rtsp uri={uri}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    deadline = time.monotonic() + timeout_sec
    latest = None
    reads = 0
    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                latest = frame
                reads += 1
                if reads >= 12:
                    break
            else:
                time.sleep(0.03)
    finally:
        cap.release()
    if latest is None:
        raise RuntimeError(f"no_frame uri={uri}")
    return latest


def _fit_display(frame: np.ndarray, max_height: int = 650) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_height / float(h))
    if scale >= 0.999:
        return frame.copy(), 1.0
    out = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(src, matrix).reshape(-1, 2)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.percentile(values, q))


def _draw_numbered(frame: np.ndarray, points: list[tuple[float, float]], *, offset_x: int = 0) -> None:
    for index, (x, y) in enumerate(points, start=1):
        p = (int(round(x)) + offset_x, int(round(y)))
        cv2.circle(frame, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(index), (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(index), (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def _interactive_pairs(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    camera_a: str,
    camera_b: str,
    target_points: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    disp_a, scale_a = _fit_display(frame_a)
    disp_b, scale_b = _fit_display(frame_b)
    height = max(disp_a.shape[0], disp_b.shape[0])
    if disp_a.shape[0] != height:
        pad = np.zeros((height - disp_a.shape[0], disp_a.shape[1], 3), dtype=np.uint8)
        disp_a = np.vstack((disp_a, pad))
    if disp_b.shape[0] != height:
        pad = np.zeros((height - disp_b.shape[0], disp_b.shape[1], 3), dtype=np.uint8)
        disp_b = np.vstack((disp_b, pad))

    width_a = disp_a.shape[1]
    base = np.hstack((disp_a, disp_b))
    points_a: list[tuple[float, float]] = []
    points_b: list[tuple[float, float]] = []
    pending_a: tuple[float, float] | None = None
    window = f"V11 floor calibration {camera_a} <-> {camera_b}"

    def mouse(event, x, y, _flags, _userdata):
        nonlocal pending_a
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if pending_a is None:
            if x >= width_a:
                return
            pending_a = (x / scale_a, y / scale_a)
        else:
            if x < width_a:
                return
            bx = (x - width_a) / scale_b
            by = y / scale_b
            points_a.append(pending_a)
            points_b.append((bx, by))
            pending_a = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            canvas = base.copy()
            scaled_a = [(x * scale_a, y * scale_a) for x, y in points_a]
            scaled_b = [(x * scale_b, y * scale_b) for x, y in points_b]
            _draw_numbered(canvas, scaled_a)
            _draw_numbered(canvas, scaled_b, offset_x=width_a)
            if pending_a is not None:
                px = int(round(pending_a[0] * scale_a))
                py = int(round(pending_a[1] * scale_a))
                cv2.circle(canvas, (px, py), 9, (255, 255, 255), 2, cv2.LINE_AA)

            expected = camera_a if pending_a is None else camera_b
            text = (
                f"Pairs {len(points_a)}/{target_points} | click same FLOOR point: now {expected} | "
                "ENTER=solve  U=undo  R=reset  Q=quit"
            )
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(canvas, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (width_a, 0), (width_a, height), (255, 255, 255), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt
            if key in (ord("r"), ord("R")):
                points_a.clear()
                points_b.clear()
                pending_a = None
            elif key in (ord("u"), ord("U")):
                if pending_a is not None:
                    pending_a = None
                elif points_a:
                    points_a.pop()
                    points_b.pop()
            elif key in (10, 13):
                if pending_a is None and len(points_a) >= 4:
                    return points_a, points_b
            if len(points_a) >= target_points and pending_a is None:
                # Do not auto-close: ENTER makes the user's acceptance explicit.
                pass
    finally:
        cv2.destroyWindow(window)


def _verification_image(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    projected_b_to_a: np.ndarray,
    inliers: np.ndarray,
) -> np.ndarray:
    a = frame_a.copy()
    b = frame_b.copy()
    for idx, (pa, pb, proj, inlier) in enumerate(zip(points_a, points_b, projected_b_to_a, inliers), start=1):
        ia = (int(round(pa[0])), int(round(pa[1])))
        ib = (int(round(pb[0])), int(round(pb[1])))
        ip = (int(round(proj[0])), int(round(proj[1])))
        cv2.circle(a, ia, 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(a, ip, (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.line(a, ia, ip, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(a, f"{idx}{'*' if inlier else 'x'}", (ia[0] + 8, ia[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(b, ib, 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(b, str(idx), (ib[0] + 8, ib[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    target_h = min(a.shape[0], b.shape[0], 720)
    def resize_to_h(img: np.ndarray) -> np.ndarray:
        s = target_h / img.shape[0]
        return cv2.resize(img, (int(round(img.shape[1] * s)), target_h), interpolation=cv2.INTER_AREA)
    return np.hstack((resize_to_h(a), resize_to_h(b)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate one same-room camera pair on the floor plane.")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--camera-a", default="CAM-01")
    parser.add_argument("--camera-b", default="CAM-04")
    parser.add_argument("--frame-a", default="", help="Optional still image instead of RTSP capture")
    parser.add_argument("--frame-b", default="", help="Optional still image instead of RTSP capture")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--ransac-px", type=float, default=8.0)
    parser.add_argument("--output", default="artifacts/calibration/devs_cam01_cam04_floor_v1.json")
    parser.add_argument("--pass-median-px", type=float, default=12.0)
    parser.add_argument("--pass-p95-px", type=float, default=25.0)
    parser.add_argument("--pass-inlier-ratio", type=float, default=0.75)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cameras = _load_cameras(config_path)
    if args.camera_a not in cameras or args.camera_b not in cameras:
        raise SystemExit("V11_CAM_PAIR_CALIB RESULT=FAIL reason=camera_not_in_config")
    row_a, row_b = cameras[args.camera_a], cameras[args.camera_b]
    if str(row_a.get("room", "")) != str(row_b.get("room", "")):
        raise SystemExit("V11_CAM_PAIR_CALIB RESULT=FAIL reason=not_same_room")

    def load_or_capture(path_text: str, row: dict[str, object]) -> np.ndarray:
        if path_text:
            path = Path(path_text).expanduser()
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot_read_image path={path}")
            return image
        return _capture_latest(str(row["uri"]))

    print(
        f"V11_CAM_PAIR_CALIB READY room={row_a.get('room')} pair={args.camera_a},{args.camera_b} "
        f"target_points={max(4, args.points)} mode=floor-plane-homography",
        flush=True,
    )
    frame_a = load_or_capture(args.frame_a, row_a)
    frame_b = load_or_capture(args.frame_b, row_b)
    points_a_list, points_b_list = _interactive_pairs(
        frame_a, frame_b, args.camera_a, args.camera_b, max(4, args.points)
    )
    points_a = np.asarray(points_a_list, dtype=np.float32)
    points_b = np.asarray(points_b_list, dtype=np.float32)

    h_b_to_a, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC, float(args.ransac_px))
    if h_b_to_a is None or mask is None:
        print("V11_CAM_PAIR_CALIB RESULT=FAIL reason=findHomography", flush=True)
        return 2
    try:
        h_a_to_b = np.linalg.inv(h_b_to_a)
    except np.linalg.LinAlgError:
        print("V11_CAM_PAIR_CALIB RESULT=FAIL reason=singular_homography", flush=True)
        return 2

    projected_b_to_a = _project(points_b, h_b_to_a)
    errors = np.linalg.norm(projected_b_to_a - points_a, axis=1)
    inliers = mask.reshape(-1).astype(bool)
    inlier_errors = errors[inliers]
    inlier_ratio = float(np.mean(inliers))
    median_px = float(np.median(inlier_errors)) if inlier_errors.size else float("inf")
    p95_px = _percentile(inlier_errors, 95)
    max_px = float(np.max(inlier_errors)) if inlier_errors.size else float("inf")
    passed = bool(
        inlier_errors.size >= 4
        and inlier_ratio >= float(args.pass_inlier_ratio)
        and median_px <= float(args.pass_median_px)
        and p95_px <= float(args.pass_p95_px)
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    frame_a_path = stem.parent / f"{stem.name}__{args.camera_a}.jpg"
    frame_b_path = stem.parent / f"{stem.name}__{args.camera_b}.jpg"
    verify_path = stem.parent / f"{stem.name}__verify.jpg"
    cv2.imwrite(str(frame_a_path), frame_a, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(frame_b_path), frame_b, [cv2.IMWRITE_JPEG_QUALITY, 95])
    verify = _verification_image(frame_a, frame_b, points_a, points_b, projected_b_to_a, inliers)
    cv2.imwrite(str(verify_path), verify, [cv2.IMWRITE_JPEG_QUALITY, 95])

    payload = {
        "version": 1,
        "mode": "floor_plane_pair_homography",
        "room": str(row_a.get("room", "")),
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "image_size_a": [int(frame_a.shape[1]), int(frame_a.shape[0])],
        "image_size_b": [int(frame_b.shape[1]), int(frame_b.shape[0])],
        "points_a": points_a.astype(float).tolist(),
        "points_b": points_b.astype(float).tolist(),
        "homography_b_to_a": h_b_to_a.astype(float).tolist(),
        "homography_a_to_b": h_a_to_b.astype(float).tolist(),
        "inliers": inliers.astype(int).tolist(),
        "metrics": {
            "point_count": int(points_a.shape[0]),
            "inlier_count": int(np.sum(inliers)),
            "inlier_ratio": inlier_ratio,
            "median_reprojection_px": median_px,
            "p95_reprojection_px": p95_px,
            "max_reprojection_px": max_px,
        },
        "pass_thresholds": {
            "min_inlier_ratio": float(args.pass_inlier_ratio),
            "max_median_px": float(args.pass_median_px),
            "max_p95_px": float(args.pass_p95_px),
        },
        "passed": passed,
        "verification_image": str(verify_path),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        "V11_CAM_PAIR_CALIB_METRICS "
        f"points={len(points_a)} inliers={int(np.sum(inliers))} ratio={inlier_ratio:.3f} "
        f"median={median_px:.2f}px p95={p95_px:.2f}px max={max_px:.2f}px "
        f"verify={verify_path}",
        flush=True,
    )
    print(
        f"V11_CAM_PAIR_CALIB RESULT={'PASS' if passed else 'FAIL'} output={output}",
        flush=True,
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
