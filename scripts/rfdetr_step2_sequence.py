#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rfdetr_step1_image import _camera_rtsp_url, _class_name, _coco_classes, _redacted_rtsp_url
from rfdetr_step1_dedupe import _dedupe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: RF-DETR-S robustness test over a short single-camera frame sequence."
    )
    parser.add_argument("--camera", default="CAM-03")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--capture-fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--person-threshold", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.62)
    parser.add_argument("--containment", type=float, default=0.90)
    parser.add_argument("--center", type=float, default=0.35)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rfdetr_step2_sequence")
    )
    return parser.parse_args()


def _capture_sequence(args: argparse.Namespace) -> list[Path]:
    from services.ml_service.app.config import load_settings

    settings = load_settings()
    camera = next((row for row in settings.cameras if row.camera_id == args.camera), None)
    if camera is None:
        available = ",".join(row.camera_id for row in settings.cameras)
        raise SystemExit(
            f"STEP2_FAIL camera_not_found={args.camera} available=[{available}]"
        )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("STEP2_FAIL ffmpeg_not_found")

    capture_dir = args.output_dir / "frames"
    capture_dir.mkdir(parents=True, exist_ok=True)
    for old in capture_dir.glob("frame_*.jpg"):
        old.unlink()

    url = _camera_rtsp_url(camera)
    redacted = _redacted_rtsp_url(url)
    pattern = capture_dir / "frame_%03d.jpg"
    timeout_sec = max(20, int(args.frames / max(0.1, args.capture_fps) + 15))

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-an",
        "-vf",
        f"fps={float(args.capture_fps):.6f}",
        "-frames:v",
        str(int(args.frames)),
        "-q:v",
        "2",
        "-y",
        str(pattern),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(
            f"STEP2_FAIL capture_timeout camera={camera.camera_id} timeout={timeout_sec}s"
        )

    frames = sorted(capture_dir.glob("frame_*.jpg"))
    if result.returncode != 0 or len(frames) != int(args.frames):
        error = (result.stderr or "ffmpeg capture failed").strip().replace("\n", " | ")
        error = error.replace(url, redacted)
        if len(error) > 500:
            error = error[-500:]
        raise SystemExit(
            f"STEP2_FAIL capture camera={camera.camera_id} got={len(frames)}/{args.frames} error={error}"
        )

    print(
        f"STEP2_CAPTURE camera={camera.camera_id} frames={len(frames)} fps={args.capture_fps:.2f} dir={capture_dir}",
        flush=True,
    )
    return frames


def _person_rows(detections, classes, threshold: float, source_w: int, source_h: int):
    rows = []
    xyxy = getattr(detections, "xyxy", None)
    confidences = getattr(detections, "confidence", None)
    class_ids = getattr(detections, "class_id", None)
    if xyxy is None or confidences is None or class_ids is None:
        raise RuntimeError("unexpected RF-DETR prediction format")

    for box, confidence, class_id in zip(xyxy, confidences, class_ids):
        class_id = int(class_id)
        confidence = float(confidence)
        if _class_name(classes, class_id) != "person" or confidence < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(float(source_w - 1), x1))
        y1 = max(0.0, min(float(source_h - 1), y1))
        x2 = max(x1 + 1.0, min(float(source_w), x2))
        y2 = max(y1 + 1.0, min(float(source_h), y2))
        rows.append(
            {
                "class_id": class_id,
                "class_name": "person",
                "confidence": confidence,
                "xyxy": [x1, y1, x2, y2],
            }
        )
    rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return rows


def main() -> int:
    args = _parse_args()
    if args.frames < 3 or args.frames > 120:
        raise SystemExit("STEP2_FAIL frames_must_be_3_to_120")
    if args.capture_fps <= 0.0 or args.capture_fps > 10.0:
        raise SystemExit("STEP2_FAIL capture_fps_must_be_0_to_10")
    if args.width % 32 or args.height % 32:
        raise SystemExit(
            f"STEP2_FAIL shape_must_be_divisible_by_32 got={args.width}x{args.height}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = _capture_sequence(args)

    import torch
    import rfdetr
    from rfdetr import RFDETRSmall

    if not torch.cuda.is_available():
        raise SystemExit("STEP2_FAIL torch_cuda_unavailable")

    shape = (int(args.height), int(args.width))
    classes = _coco_classes()
    model = RFDETRSmall(device="cuda:0")

    first_image = Image.open(frames[0]).convert("RGB")
    _ = model.predict(
        first_image,
        threshold=0.05,
        shape=shape,
        include_source_image=False,
    )
    torch.cuda.synchronize()

    frame_reports = []
    infer_times = []
    unique_counts = []
    min_confidences = []

    for index, frame_path in enumerate(frames, start=1):
        image = Image.open(frame_path).convert("RGB")
        source_w, source_h = image.size
        started = time.perf_counter()
        detections = model.predict(
            image,
            threshold=0.05,
            shape=shape,
            include_source_image=False,
        )
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - started) * 1000.0

        raw_persons = _person_rows(
            detections,
            classes,
            float(args.person_threshold),
            source_w,
            source_h,
        )
        kept, rejected = _dedupe(
            raw_persons,
            float(args.iou),
            float(args.containment),
            float(args.center),
        )
        confidences = [float(row["confidence"]) for row in kept]
        min_conf = min(confidences) if confidences else 0.0
        mean_conf = statistics.mean(confidences) if confidences else 0.0

        infer_times.append(infer_ms)
        unique_counts.append(len(kept))
        min_confidences.append(min_conf)
        frame_reports.append(
            {
                "index": index,
                "image": str(frame_path),
                "raw_persons": len(raw_persons),
                "unique_persons": len(kept),
                "duplicates": len(rejected),
                "min_confidence": round(min_conf, 6),
                "mean_confidence": round(mean_conf, 6),
                "infer_ms": round(infer_ms, 3),
                "persons": kept,
            }
        )
        print(
            "STEP2_FRAME "
            f"idx={index:02d} raw={len(raw_persons)} unique={len(kept)} "
            f"dup={len(rejected)} min_conf={min_conf:.2f} mean_conf={mean_conf:.2f} "
            f"infer_ms={infer_ms:.1f}",
            flush=True,
        )

    mode_count = max(set(unique_counts), key=unique_counts.count)
    stable_frames = sum(1 for count in unique_counts if count == mode_count)
    stable_ratio = stable_frames / len(unique_counts)
    count_min = min(unique_counts)
    count_max = max(unique_counts)

    report = {
        "stage": 2,
        "camera": args.camera,
        "backend": "RF-DETR-S PyTorch CUDA sequence truth",
        "model_shape_hw": [args.height, args.width],
        "person_threshold": float(args.person_threshold),
        "dedupe": {
            "iou": float(args.iou),
            "containment": float(args.containment),
            "center_distance": float(args.center),
        },
        "capture": {
            "frames": len(frames),
            "fps": float(args.capture_fps),
        },
        "summary": {
            "mode_unique_persons": mode_count,
            "stable_frames": stable_frames,
            "stable_ratio": round(stable_ratio, 6),
            "unique_count_min": count_min,
            "unique_count_max": count_max,
            "infer_ms_mean": round(statistics.mean(infer_times), 3),
            "infer_ms_min": round(min(infer_times), 3),
            "infer_ms_max": round(max(infer_times), 3),
            "min_kept_confidence": round(min(min_confidences), 6),
        },
        "frames": frame_reports,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "cuda": str(torch.version.cuda),
            "rfdetr": getattr(rfdetr, "__version__", "unknown"),
        },
    }
    report_path = args.output_dir / "sequence_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "STEP2_RESULT "
        f"camera={args.camera} frames={len(frames)} mode_unique={mode_count} "
        f"stable={stable_frames}/{len(frames)}({stable_ratio*100.0:.1f}%) "
        f"count_range={count_min}-{count_max} "
        f"mean_ms={statistics.mean(infer_times):.1f} "
        f"min_kept_conf={min(min_confidences):.2f}",
        flush=True,
    )
    print(f"STEP2_JSON={report_path}", flush=True)
    print("STEP2_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
