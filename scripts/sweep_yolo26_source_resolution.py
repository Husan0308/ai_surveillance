#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def _person_stats(result) -> tuple[int, float]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0, 0.0
    cls = boxes.cls.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    mask = cls.astype(np.int64) == 0
    if not np.any(mask):
        return 0, 0.0
    values = conf[mask]
    return int(values.size), float(values.max(initial=0.0))


def _parse_sizes(raw: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        try:
            width_s, height_s = token.split("x", 1)
            width = int(width_s)
            height = int(height_s)
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"invalid size {item!r}; expected WIDTHxHEIGHT") from exc
        if width <= 0 or height <= 0:
            raise argparse.ArgumentTypeError(f"invalid size {item!r}")
        output.append((width, height))
    if not output:
        raise argparse.ArgumentTypeError("at least one size is required")
    return output


def _predict(model, frame: np.ndarray, width: int, height: int, conf: float, device: str, end2end: bool):
    results = model.predict(
        source=frame,
        imgsz=(height, width),
        rect=False,
        conf=conf,
        classes=[0],
        device=device,
        verbose=False,
        end2end=end2end,
    )
    if not results:
        return 0, 0.0
    return _person_stats(results[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep YOLO26 model scale and detector input resolution on one-shot high-res source frames."
    )
    parser.add_argument("--dir", default=".runtime/yolo26_resolution")
    parser.add_argument("--models", default="yolo26s.pt,yolo26m.pt")
    parser.add_argument("--sizes", default="672x384,896x512,1024x576")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.08)
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"RES_SWEEP_ERROR missing directory: {root}")

    models = [v.strip() for v in args.models.split(",") if v.strip()]
    sizes = _parse_sizes(args.sizes)
    frames = sorted(root.glob("CAM-*_source_*.npy"))
    if not frames:
        raise SystemExit(f"RES_SWEEP_ERROR no source NPY frames found in {root}")

    from ultralytics import YOLO

    aggregate: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)

    for model_name in models:
        e2e_model = YOLO(model_name)
        otm_model = YOLO(model_name)
        for npy_path in frames:
            frame = np.load(npy_path, allow_pickle=False)
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise RuntimeError(f"{npy_path}: expected uint8 BGR HxWx3, got {frame.shape}/{frame.dtype}")
            cid = npy_path.name.split("_source_", 1)[0]
            src_h, src_w = frame.shape[:2]

            for width, height in sizes:
                e2e_count, e2e_max = _predict(
                    e2e_model, frame, width, height, args.conf, args.device, True
                )
                otm_count, otm_max = _predict(
                    otm_model, frame, width, height, args.conf, args.device, False
                )
                aggregate[(cid, model_name, f"{width}x{height}", "e2e")].append(e2e_count)
                aggregate[(cid, model_name, f"{width}x{height}", "otm")].append(otm_count)
                print(
                    "YOLO26_RES_SWEEP "
                    f"cid={cid} source={src_w}x{src_h} model={model_name} "
                    f"imgsz={width}x{height} "
                    f"e2e={e2e_count}/{e2e_max:.3f} "
                    f"otm={otm_count}/{otm_max:.3f}",
                    flush=True,
                )

    for key in sorted(aggregate):
        cid, model_name, size, head = key
        counts = aggregate[key]
        print(
            "YOLO26_RES_SWEEP_CAMERA "
            f"cid={cid} model={model_name} imgsz={size} head={head} "
            f"nonzero={sum(v > 0 for v in counts)}/{len(counts)} "
            f"max_count={max(counts, default=0)}",
            flush=True,
        )

    print(
        "YOLO26_RES_SWEEP_GUIDE "
        "if CAM-02/CAM-05 recover at 896x512 or 1024x576 with yolo26s, prefer higher detector resolution before a larger model; "
        "if only yolo26m recovers at the same resolution, prefer the larger model; "
        "if neither recovers from 1920x1080 source frames, pretrained COCO recall is insufficient for that view and camera-specific fine-tuning/cropping is the next step.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
