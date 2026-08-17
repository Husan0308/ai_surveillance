from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "reid_room_calibration.json"


def parse_room_points(value: str) -> list[list[float]]:
    points: list[list[float]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        xy = [part.strip() for part in item.split(",")]
        if len(xy) != 2:
            raise argparse.ArgumentTypeError(
                "room points format: x,y;x,y;x,y;x,y (metres)"
            )
        points.append([float(xy[0]), float(xy[1])])
    if len(points) < 4:
        raise argparse.ArgumentTypeError("at least 4 room points are required")
    return points


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manually click floor correspondences for one camera. Click image points "
            "in exactly the same order as --room-points. Coordinates are saved "
            "normalized, so runtime resolution changes do not invalidate calibration."
        )
    )
    parser.add_argument("--camera", required=True, help="CAM-01 .. CAM-06")
    parser.add_argument("--image", required=True, help="saved frame/screenshot from that camera")
    parser.add_argument(
        "--room-points",
        required=True,
        type=parse_room_points,
        help='physical floor points in metres, e.g. "0,0;4,0;4,6;0,6"',
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    camera = data.get("cameras", {}).get(args.camera)
    if camera is None:
        raise SystemExit(f"camera not found in config: {args.camera}")

    image = cv2.imread(str(Path(args.image).expanduser()))
    if image is None or image.size == 0:
        raise SystemExit(f"could not read image: {args.image}")
    h, w = image.shape[:2]
    room_points = args.room_points
    clicks: list[tuple[int, int]] = []
    display = image.copy()
    window = f"Calibrate {args.camera}: click {len(room_points)} points in order"

    def redraw() -> None:
        nonlocal display
        display = image.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(display, (x, y), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                display,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            display,
            f"click {len(clicks)+1}/{len(room_points)} | u=undo r=reset q=cancel",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)

    def mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= len(room_points):
            return
        clicks.append((int(x), int(y)))
        redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1280, w), min(800, h))
    cv2.setMouseCallback(window, mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("CALIBRATION_CANCELLED")
            return 2
        if key == ord("u") and clicks:
            clicks.pop()
            redraw()
        if key == ord("r"):
            clicks.clear()
            redraw()
        if len(clicks) == len(room_points):
            break

    cv2.destroyAllWindows()
    src = np.asarray([[x / float(w), y / float(h)] for x, y in clicks], dtype=np.float64)
    dst = np.asarray(room_points, dtype=np.float64)
    method = cv2.RANSAC if len(src) > 4 else 0
    H, mask = cv2.findHomography(src, dst, method, 0.12)
    if H is None:
        raise SystemExit("findHomography failed; choose non-collinear floor points")
    projected = cv2.perspectiveTransform(src.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    if mask is not None and int(mask.sum()) >= 4:
        keep = mask.reshape(-1).astype(bool)
        projected = projected[keep]
        dst_eval = dst[keep]
    else:
        dst_eval = dst
    rmse = float(np.sqrt(np.mean(np.sum((projected - dst_eval) ** 2, axis=1))))

    camera["image_points_norm"] = [[round(float(x), 8), round(float(y), 8)] for x, y in src]
    camera["room_points_m"] = [[float(x), float(y)] for x, y in dst]
    camera["enabled"] = True
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(f"CALIBRATION_SAVED camera={args.camera} points={len(src)} rmse={rmse:.4f}m")
    print(f"config={config_path}")
    print("Use the SAME physical room coordinate system for the peer camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
