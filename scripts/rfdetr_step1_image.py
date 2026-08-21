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
from urllib.parse import quote, urlsplit, urlunsplit

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _coco_classes():
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except ImportError:
        from rfdetr.util.coco_classes import COCO_CLASSES
    return COCO_CLASSES


def _class_name(classes, class_id: int) -> str:
    try:
        value = classes[int(class_id)]
    except Exception:
        try:
            value = classes.get(int(class_id), str(class_id))
        except Exception:
            value = str(class_id)
    return str(value).strip().lower()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: isolated RF-DETR-S CUDA person detection on one image or one configured camera frame."
    )
    parser.add_argument("image", nargs="?", type=Path, help="Camera JPG/PNG frame")
    parser.add_argument(
        "--camera",
        type=str,
        default="",
        help="Configured camera id, for example CAM-03. Captures one RTSP frame automatically.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rfdetr_step1")
    )
    parser.add_argument(
        "--person-threshold",
        type=float,
        default=0.12,
        help="Final person confidence gate; raw model is queried from 0.05.",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def _camera_rtsp_url(camera) -> str:
    """Build an authenticated RTSP URL without printing credentials."""
    raw = str(camera.uri)
    parts = urlsplit(raw)
    if parts.username:
        return raw
    if not camera.username:
        return raw

    host = parts.hostname or ""
    if not host:
        raise RuntimeError(f"invalid RTSP URI for {camera.camera_id}")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"

    user = quote(str(camera.username), safe="")
    password = quote(str(camera.password or ""), safe="")
    auth = user if not password else f"{user}:{password}"
    netloc = f"{auth}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _redacted_rtsp_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or "camera"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))


def _capture_camera_frame(camera_id: str, output_dir: Path) -> Path:
    from services.ml_service.app.config import load_settings

    settings = load_settings()
    camera = next((row for row in settings.cameras if row.camera_id == camera_id), None)
    if camera is None:
        available = ",".join(row.camera_id for row in settings.cameras)
        raise SystemExit(
            f"STEP1_FAIL camera_not_found={camera_id} available=[{available}]"
        )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("STEP1_FAIL ffmpeg_not_found install=ffmpeg")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = camera.camera_id.lower().replace("-", "_")
    frame_path = output_dir / f"{safe_id}_input.jpg"
    url = _camera_rtsp_url(camera)
    redacted_url = _redacted_rtsp_url(url)

    # Do not depend on ffmpeg-build-specific socket timeout options such as
    # rw_timeout/stimeout.  The Python subprocess timeout is portable and also
    # guarantees a wedged RTSP request cannot block this isolated detector test.
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
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(frame_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(f"STEP1_FAIL camera_capture_timeout camera={camera.camera_id}")

    if result.returncode != 0 or not frame_path.is_file() or frame_path.stat().st_size == 0:
        error = (result.stderr or "ffmpeg capture failed").strip().replace("\n", " | ")
        error = error.replace(url, redacted_url)
        if len(error) > 500:
            error = error[-500:]
        raise SystemExit(
            f"STEP1_FAIL camera_capture camera={camera.camera_id} error={error}"
        )

    print(
        f"STEP1_CAPTURE camera={camera.camera_id} image={frame_path}",
        flush=True,
    )
    return frame_path


def main() -> int:
    args = _parse_args()
    if bool(args.image) == bool(args.camera):
        raise SystemExit("STEP1_FAIL choose_exactly_one_source image_or_--camera")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = (
        _capture_camera_frame(str(args.camera).strip(), args.output_dir)
        if args.camera
        else args.image
    )
    assert image_path is not None
    if not image_path.is_file():
        raise SystemExit(f"STEP1_FAIL image_not_found={image_path}")
    if args.height <= 0 or args.width <= 0:
        raise SystemExit("STEP1_FAIL invalid_shape")
    if args.height % 32 or args.width % 32:
        raise SystemExit(
            f"STEP1_FAIL shape_must_be_divisible_by_32 got={args.width}x{args.height}"
        )
    if not 0.01 <= args.person_threshold <= 0.95:
        raise SystemExit("STEP1_FAIL invalid_person_threshold")

    import torch
    import rfdetr
    from rfdetr import RFDETRSmall

    if not torch.cuda.is_available():
        raise SystemExit("STEP1_FAIL torch_cuda_unavailable")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    image = Image.open(image_path).convert("RGB")
    source_w, source_h = image.size
    shape = (int(args.height), int(args.width))

    print(
        "STEP1_ENV "
        f"rfdetr={getattr(rfdetr, '__version__', 'unknown')} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={device_name!r} sm={capability[0]}.{capability[1]} "
        f"source={source_w}x{source_h} model_shape={args.width}x{args.height}",
        flush=True,
    )

    model = RFDETRSmall(device="cuda:0")

    raw_threshold = 0.05
    _ = model.predict(
        image,
        threshold=raw_threshold,
        shape=shape,
        include_source_image=False,
    )
    torch.cuda.synchronize()

    timings_ms: list[float] = []
    detections = None
    for _ in range(max(1, int(args.runs))):
        started = time.perf_counter()
        detections = model.predict(
            image,
            threshold=raw_threshold,
            shape=shape,
            include_source_image=False,
        )
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    if detections is None:
        raise SystemExit("STEP1_FAIL no_prediction_object")

    classes = _coco_classes()
    raw_summary: dict[str, int] = {}
    persons = []

    xyxy = getattr(detections, "xyxy", None)
    confidences = getattr(detections, "confidence", None)
    class_ids = getattr(detections, "class_id", None)
    if xyxy is None or confidences is None or class_ids is None:
        raise SystemExit("STEP1_FAIL unexpected_prediction_format")

    for box, confidence, class_id in zip(xyxy, confidences, class_ids):
        class_id = int(class_id)
        confidence = float(confidence)
        name = _class_name(classes, class_id)
        raw_summary[name] = raw_summary.get(name, 0) + 1
        if name != "person" or confidence < float(args.person_threshold):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(float(source_w - 1), x1))
        y1 = max(0.0, min(float(source_h - 1), y1))
        x2 = max(x1 + 1.0, min(float(source_w), x2))
        y2 = max(y1 + 1.0, min(float(source_h), y2))
        persons.append(
            {
                "class_id": class_id,
                "class_name": "person",
                "confidence": round(confidence, 6),
                "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    persons.sort(key=lambda row: row["confidence"], reverse=True)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for index, row in enumerate(persons, start=1):
        x1, y1, x2, y2 = row["xyxy"]
        conf = row["confidence"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 64), width=4)
        label = f"person {index}  {conf:.2f}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_h = max(16, text_box[3] - text_box[1] + 6)
        label_y = max(0.0, y1 - text_h)
        text_w = max(40, text_box[2] - text_box[0] + 8)
        draw.rectangle((x1, label_y, x1 + text_w, label_y + text_h), fill=(255, 196, 64))
        draw.text((x1 + 4, label_y + 3), label, fill=(15, 18, 22), font=font)

    image_out = args.output_dir / "person_detection.jpg"
    json_out = args.output_dir / "person_detection.json"
    annotated.save(image_out, quality=95, subsampling=0)

    report = {
        "stage": 1,
        "backend": "RF-DETR-S PyTorch CUDA truth",
        "source_image": str(image_path),
        "source_camera": str(args.camera).strip() if args.camera else None,
        "source_size": [source_w, source_h],
        "model_shape_hw": [args.height, args.width],
        "raw_threshold": raw_threshold,
        "person_threshold": float(args.person_threshold),
        "person_count": len(persons),
        "persons": persons,
        "raw_class_counts": dict(sorted(raw_summary.items())),
        "timing_ms": {
            "runs": len(timings_ms),
            "mean": round(statistics.mean(timings_ms), 3),
            "min": round(min(timings_ms), 3),
            "max": round(max(timings_ms), 3),
        },
        "gpu": {
            "name": device_name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "cuda": str(torch.version.cuda),
        },
    }
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    person_text = " ".join(
        f"#{i}:{row['confidence']:.2f}@{row['xyxy']}"
        for i, row in enumerate(persons, start=1)
    ) or "none"
    print(
        "STEP1_RESULT "
        f"persons={len(persons)} detections=[{person_text}] "
        f"mean_ms={report['timing_ms']['mean']:.1f} "
        f"raw_classes={report['raw_class_counts']}",
        flush=True,
    )
    print(f"STEP1_IMAGE={image_out}", flush=True)
    print(f"STEP1_JSON={json_out}", flush=True)
    print("STEP1_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
