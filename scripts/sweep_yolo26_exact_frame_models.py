#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _person_stats(result) -> tuple[int, float]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0, 0.0
    cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
    conf = boxes.conf.detach().cpu().numpy()
    mask = cls == 0
    if not np.any(mask):
        return 0, 0.0
    values = conf[mask]
    return int(values.size), float(values.max(initial=0.0))


def _predict(model, frame: np.ndarray, conf: float, device: str, end2end: bool):
    results = model.predict(
        source=frame,
        imgsz=(384, 672),
        conf=conf,
        classes=[0],
        device=device,
        verbose=False,
        end2end=end2end,
    )
    return _person_stats(results[0]) if results else (0, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple YOLO26 PyTorch model sizes on the exact 672x384 "
            "production detector frames saved by the parity capture."
        )
    )
    parser.add_argument("--dir", default=".runtime/yolo26_parity")
    parser.add_argument(
        "--models",
        default="yolo26s.pt,yolo26m.pt",
        help="Comma-separated checkpoints, e.g. yolo26s.pt,yolo26m.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument(
        "--cameras",
        default="CAM-01,CAM-02,CAM-04,CAM-05",
        help="Control + suspect cameras to evaluate",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"SWEEP_ERROR missing directory: {root}")

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    cameras = {x.strip() for x in args.cameras.split(",") if x.strip()}
    if not models:
        raise SystemExit("SWEEP_ERROR no models requested")

    from ultralytics import YOLO

    frames: list[tuple[str, int, Path, np.ndarray]] = []
    for npy_path in sorted(root.glob("CAM-*_sample*.npy")):
        cid = npy_path.name.split("_sample", 1)[0]
        if cameras and cid not in cameras:
            continue
        json_path = npy_path.with_suffix(".json")
        sample = 0
        if json_path.is_file():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            sample = int(meta.get("sample", 0))
        frame = np.load(npy_path, allow_pickle=False)
        if frame.shape != (384, 672, 3) or frame.dtype != np.uint8:
            raise RuntimeError(
                f"{npy_path}: expected uint8 BGR 384x672x3, got {frame.shape}/{frame.dtype}"
            )
        frames.append((cid, sample, npy_path, frame))

    if not frames:
        raise SystemExit("SWEEP_ERROR no matching parity frames")

    summary: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    rows = []

    for model_name in models:
        # Separate predictor instances prevent one head's state from affecting the other.
        e2e_model = YOLO(model_name)
        otm_model = YOLO(model_name)
        for cid, sample, npy_path, frame in frames:
            e2e_count, e2e_max = _predict(
                e2e_model, frame, args.conf, args.device, True
            )
            otm_count, otm_max = _predict(
                otm_model, frame, args.conf, args.device, False
            )
            rows.append(
                {
                    "camera": cid,
                    "sample": sample,
                    "model": model_name,
                    "e2e_count": e2e_count,
                    "e2e_max": e2e_max,
                    "otm_count": otm_count,
                    "otm_max": otm_max,
                    "file": npy_path.name,
                }
            )
            summary[(cid, model_name, "e2e")].append(e2e_count)
            summary[(cid, model_name, "otm")].append(otm_count)
            print(
                "YOLO26_MODEL_SWEEP "
                f"cid={cid} sample={sample} model={model_name} "
                f"e2e={e2e_count}/{e2e_max:.3f} "
                f"otm={otm_count}/{otm_max:.3f}",
                flush=True,
            )

    (root / "model_sweep_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )

    for (cid, model_name, head), counts in sorted(summary.items()):
        nonzero = sum(1 for value in counts if value > 0)
        print(
            "YOLO26_MODEL_SWEEP_CAMERA "
            f"cid={cid} model={model_name} head={head} "
            f"nonzero={nonzero}/{len(counts)} max_count={max(counts, default=0)}",
            flush=True,
        )

    print(
        "YOLO26_MODEL_SWEEP_GUIDE "
        "if yolo26m recovers CAM-02/CAM-05 at the same 672x384 input, prefer a larger detector; "
        "if yolo26m still misses, the next test must recapture higher-resolution source frames "
        "because upscaling these saved 672x384 frames cannot restore lost detail.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
