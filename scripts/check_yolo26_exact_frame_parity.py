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
    cls = boxes.cls.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    mask = cls.astype(np.int64) == 0
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
    if not results:
        return 0, 0.0
    return _person_stats(results[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the exact production TRT input frame with YOLO26 PyTorch."
    )
    parser.add_argument(
        "--dir",
        default=".runtime/yolo26_parity",
        help="Directory written by CAMERA_V2_PARITY_CAPTURE_*",
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.08,
        help="Person confidence threshold used for all three comparisons",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"PARITY_ERROR missing directory: {root}")

    from ultralytics import YOLO

    # Keep independent predictor state for the two heads.
    e2e_model = YOLO(args.model)
    otm_model = YOLO(args.model)

    pairs = sorted(root.glob("CAM-*_sample*.npy"))
    if not pairs:
        raise SystemExit(f"PARITY_ERROR no NPY samples found in {root}")

    aggregate: dict[str, list[str]] = defaultdict(list)
    rows = []

    for npy_path in pairs:
        json_path = npy_path.with_suffix(".json")
        if not json_path.is_file():
            print(f"PARITY_SKIP frame={npy_path} reason=missing-json", flush=True)
            continue

        frame = np.load(npy_path, allow_pickle=False)
        if frame.shape != (384, 672, 3) or frame.dtype != np.uint8:
            raise RuntimeError(
                f"{npy_path}: expected uint8 BGR 384x672x3, got {frame.shape}/{frame.dtype}"
            )
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        cid = str(meta.get("camera") or npy_path.name.split("_sample", 1)[0])

        trt_boxes = meta.get("trt_boxes") or []
        trt_count = len(trt_boxes)
        trt_max = max((float(row[4]) for row in trt_boxes if len(row) >= 5), default=0.0)

        pt_e2e_count, pt_e2e_max = _predict(
            e2e_model, frame, args.conf, args.device, True
        )
        pt_otm_count, pt_otm_max = _predict(
            otm_model, frame, args.conf, args.device, False
        )

        if trt_count > 0 and pt_e2e_count > 0:
            verdict = "PARITY_OK"
        elif trt_count == 0 and pt_e2e_count > 0:
            verdict = "TRT_MISS"
        elif trt_count == 0 and pt_e2e_count == 0 and pt_otm_count > 0:
            verdict = "ONE_TO_MANY_RECOVERS"
        elif trt_count == 0 and pt_e2e_count == 0 and pt_otm_count == 0:
            verdict = "MODEL_SCENE_MISS"
        elif trt_count > 0 and pt_e2e_count == 0:
            verdict = "REFERENCE_MISMATCH"
        else:
            verdict = "REVIEW"

        aggregate[cid].append(verdict)
        row = {
            "camera": cid,
            "sample": int(meta.get("sample", 0)),
            "trt_count": trt_count,
            "trt_max": trt_max,
            "pt_e2e_count": pt_e2e_count,
            "pt_e2e_max": pt_e2e_max,
            "pt_otm_count": pt_otm_count,
            "pt_otm_max": pt_otm_max,
            "verdict": verdict,
            "file": npy_path.name,
        }
        rows.append(row)
        print(
            "YOLO26_PARITY "
            f"cid={cid} sample={row['sample']} "
            f"trt={trt_count}/{trt_max:.3f} "
            f"pt_e2e={pt_e2e_count}/{pt_e2e_max:.3f} "
            f"pt_otm={pt_otm_count}/{pt_otm_max:.3f} "
            f"verdict={verdict}",
            flush=True,
        )

    (root / "parity_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )

    for cid in sorted(aggregate):
        verdicts = aggregate[cid]
        counts = {name: verdicts.count(name) for name in sorted(set(verdicts))}
        print(f"YOLO26_PARITY_CAMERA cid={cid} verdicts={counts}", flush=True)

    print(
        "YOLO26_PARITY_GUIDE "
        "TRT_MISS=engine/preprocess parity issue; "
        "ONE_TO_MANY_RECOVERS=YOLO26 one-to-one recall gap; "
        "MODEL_SCENE_MISS=model/view/resolution recall issue; "
        "PARITY_OK=TRT and PyTorch agree",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
