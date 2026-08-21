#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rfdetr_step1_dedupe import _dedupe
from rfdetr_step1_image import _camera_rtsp_url, _coco_classes, _redacted_rtsp_url
from rfdetr_step2_sequence import _person_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: tracker-free live RF-DETR-S test using a latest-frame-only "
            "mailbox so detector latency never creates a stale-frame backlog."
        )
    )
    parser.add_argument("--camera", default="CAM-03")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--pipe-width", type=int, default=1600)
    parser.add_argument("--pipe-height", type=int, default=900)
    parser.add_argument("--width", type=int, default=800, help="RF-DETR model width")
    parser.add_argument("--height", type=int, default=448, help="RF-DETR model height")
    parser.add_argument("--person-threshold", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.62)
    parser.add_argument("--containment", type=float, default=0.90)
    parser.add_argument("--center", type=float, default=0.35)
    parser.add_argument(
        "--expected-persons",
        type=int,
        default=-1,
        help="Optional ground-truth count for this live test; -1 disables matching metric.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save one annotated processed frame every N detector results; 0 disables.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rfdetr_step3_live")
    )
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[pos])


def _annotate(image: Image.Image, persons: list[dict], footer: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for index, row in enumerate(persons, start=1):
        x1, y1, x2, y2 = [float(v) for v in row["xyxy"]]
        conf = float(row["confidence"])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 64), width=4)
        label = f"person {index} {conf:.2f}"
        box = draw.textbbox((x1, y1), label, font=font)
        text_h = max(18, box[3] - box[1] + 6)
        text_w = max(50, box[2] - box[0] + 8)
        label_y = max(0.0, y1 - text_h)
        draw.rectangle((x1, label_y, x1 + text_w, label_y + text_h), fill=(255, 196, 64))
        draw.text((x1 + 4, label_y + 3), label, fill=(15, 18, 22), font=font)

    footer_box = draw.textbbox((8, 8), footer, font=font)
    footer_w = footer_box[2] - footer_box[0] + 16
    footer_h = footer_box[3] - footer_box[1] + 12
    draw.rectangle((4, 4, 4 + footer_w, 4 + footer_h), fill=(15, 18, 22))
    draw.text((12, 10), footer, fill=(255, 255, 255), font=font)
    return output


class LatestFramePipe:
    """Continuously reads RGB frames from ffmpeg and keeps only the newest frame."""

    def __init__(self, process: subprocess.Popen, width: int, height: int) -> None:
        self.process = process
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * 3
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.stop_event = threading.Event()
        self.latest_seq = 0
        self.latest_t = 0.0
        self.latest_frame: bytes | None = None
        self.read_error = ""
        self.thread = threading.Thread(target=self._run, name="rfdetr-step3-capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read_exact(self, count: int) -> bytes | None:
        stream = self.process.stdout
        if stream is None:
            return None
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0 and not self.stop_event.is_set():
            chunk = stream.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining:
            return None
        return b"".join(chunks)

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                frame = self._read_exact(self.frame_bytes)
                if frame is None:
                    if not self.stop_event.is_set():
                        self.read_error = "ffmpeg rawvideo stream ended"
                    break
                captured_t = time.monotonic()
                with self.lock:
                    self.latest_seq += 1
                    self.latest_t = captured_t
                    self.latest_frame = frame
                self.event.set()
        except BaseException as exc:
            self.read_error = f"{type(exc).__name__}: {exc}"
            self.event.set()

    def snapshot(self, after_seq: int) -> tuple[int, float, bytes] | None:
        with self.lock:
            if self.latest_frame is None or self.latest_seq <= after_seq:
                return None
            return self.latest_seq, self.latest_t, self.latest_frame

    def stop(self) -> None:
        self.stop_event.set()
        self.event.set()


def _start_ffmpeg(args: argparse.Namespace):
    from services.ml_service.app.config import load_settings

    settings = load_settings()
    camera = next((row for row in settings.cameras if row.camera_id == args.camera), None)
    if camera is None:
        available = ",".join(row.camera_id for row in settings.cameras)
        raise SystemExit(f"STEP3_FAIL camera_not_found={args.camera} available=[{available}]")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("STEP3_FAIL ffmpeg_not_found")

    url = _camera_rtsp_url(camera)
    redacted = _redacted_rtsp_url(url)
    vf = (
        f"fps={float(args.capture_fps):.6f},"
        f"scale={int(args.pipe_width)}:{int(args.pipe_height)}:flags=lanczos"
    )
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
        vf,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=max(65536, int(args.pipe_width) * int(args.pipe_height) * 3),
    )
    return camera, process, url, redacted


def _terminate_process(process: subprocess.Popen) -> str:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    stderr = b""
    if process.stderr is not None:
        try:
            stderr = process.stderr.read() or b""
        except Exception:
            stderr = b""
    return stderr.decode("utf-8", errors="replace").strip()


def main() -> int:
    args = _parse_args()
    if args.seconds < 5.0 or args.seconds > 300.0:
        raise SystemExit("STEP3_FAIL seconds_must_be_5_to_300")
    if args.capture_fps <= 0.0 or args.capture_fps > 20.0:
        raise SystemExit("STEP3_FAIL capture_fps_must_be_0_to_20")
    if args.pipe_width <= 0 or args.pipe_height <= 0:
        raise SystemExit("STEP3_FAIL invalid_pipe_shape")
    if args.width % 32 or args.height % 32:
        raise SystemExit(
            f"STEP3_FAIL model_shape_must_be_divisible_by_32 got={args.width}x{args.height}"
        )
    if args.expected_persons < -1:
        raise SystemExit("STEP3_FAIL invalid_expected_persons")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in sample_dir.glob("sample_*.jpg"):
        old.unlink()

    import torch
    import rfdetr
    from rfdetr import RFDETRSmall

    if not torch.cuda.is_available():
        raise SystemExit("STEP3_FAIL torch_cuda_unavailable")

    print(
        "STEP3_ENV "
        f"gpu={torch.cuda.get_device_name(0)!r} sm={'.'.join(map(str, torch.cuda.get_device_capability(0)))} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"rfdetr={getattr(rfdetr, '__version__', 'unknown')} "
        f"model={args.width}x{args.height} pipe={args.pipe_width}x{args.pipe_height} "
        f"capture_fps={args.capture_fps:.1f} latest_only=ON tracker=OFF",
        flush=True,
    )

    model = RFDETRSmall(device="cuda:0")
    classes = _coco_classes()
    shape = (int(args.height), int(args.width))

    camera, process, url, redacted_url = _start_ffmpeg(args)
    mailbox = LatestFramePipe(process, args.pipe_width, args.pipe_height)
    mailbox.start()

    last_seq = 0
    first = None
    startup_deadline = time.monotonic() + 15.0
    while time.monotonic() < startup_deadline:
        first = mailbox.snapshot(last_seq)
        if first is not None:
            break
        if process.poll() is not None or mailbox.read_error:
            break
        mailbox.event.wait(0.1)
        mailbox.event.clear()

    if first is None:
        mailbox.stop()
        stderr = _terminate_process(process).replace(url, redacted_url)
        error = mailbox.read_error or stderr or "no first frame"
        raise SystemExit(f"STEP3_FAIL live_start camera={camera.camera_id} error={error[-500:]}")

    first_seq, _first_t, first_bytes = first
    first_image = Image.frombytes("RGB", (args.pipe_width, args.pipe_height), first_bytes)
    _ = model.predict(first_image, threshold=0.05, shape=shape, include_source_image=False)
    torch.cuda.synchronize()
    last_seq = first_seq

    print(
        f"STEP3_LIVE camera={camera.camera_id} duration={args.seconds:.1f}s "
        f"threshold={args.person_threshold:.2f} dedupe=0.62/0.90/0.35",
        flush=True,
    )

    started_t = time.monotonic()
    deadline = started_t + float(args.seconds)
    frame_reports: list[dict] = []
    infer_times: list[float] = []
    input_ages: list[float] = []
    unique_counts: list[int] = []
    skipped_total = 0
    processed = 0

    try:
        while time.monotonic() < deadline:
            snapshot = mailbox.snapshot(last_seq)
            if snapshot is None:
                if process.poll() is not None or mailbox.read_error:
                    break
                mailbox.event.wait(0.01)
                mailbox.event.clear()
                continue

            seq, captured_t, raw = snapshot
            skipped = max(0, seq - last_seq - 1)
            skipped_total += skipped
            last_seq = seq
            input_age_ms = max(0.0, (time.monotonic() - captured_t) * 1000.0)
            image = Image.frombytes("RGB", (args.pipe_width, args.pipe_height), raw)

            infer_start = time.perf_counter()
            detections = model.predict(
                image,
                threshold=0.05,
                shape=shape,
                include_source_image=False,
            )
            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            raw_persons = _person_rows(
                detections,
                classes,
                float(args.person_threshold),
                int(args.pipe_width),
                int(args.pipe_height),
            )
            kept, rejected = _dedupe(
                raw_persons,
                float(args.iou),
                float(args.containment),
                float(args.center),
            )
            processed += 1
            infer_times.append(infer_ms)
            input_ages.append(input_age_ms)
            unique_counts.append(len(kept))

            confidences = [float(row["confidence"]) for row in kept]
            min_conf = min(confidences) if confidences else 0.0
            mean_conf = statistics.mean(confidences) if confidences else 0.0
            print(
                "STEP3_FRAME "
                f"n={processed:03d} seq={seq} raw={len(raw_persons)} unique={len(kept)} "
                f"dup={len(rejected)} skipped={skipped} age_ms={input_age_ms:.1f} "
                f"infer_ms={infer_ms:.1f} min_conf={min_conf:.2f} mean_conf={mean_conf:.2f}",
                flush=True,
            )

            frame_reports.append(
                {
                    "n": processed,
                    "capture_seq": seq,
                    "skipped_since_previous": skipped,
                    "input_age_ms": round(input_age_ms, 3),
                    "infer_ms": round(infer_ms, 3),
                    "raw_persons": len(raw_persons),
                    "unique_persons": len(kept),
                    "duplicates": len(rejected),
                    "persons": kept,
                }
            )

            if args.save_every > 0 and (processed == 1 or processed % args.save_every == 0):
                footer = (
                    f"{camera.camera_id} n={processed} unique={len(kept)} "
                    f"infer={infer_ms:.0f}ms age={input_age_ms:.0f}ms"
                )
                annotated = _annotate(image, kept, footer)
                annotated.save(sample_dir / f"sample_{processed:04d}.jpg", quality=92, subsampling=0)
    finally:
        mailbox.stop()
        stderr = _terminate_process(process)
        mailbox.thread.join(timeout=1.0)

    elapsed = max(1e-6, time.monotonic() - started_t)
    if processed == 0:
        error = (mailbox.read_error or stderr or "no processed frames").replace(url, redacted_url)
        raise SystemExit(f"STEP3_FAIL no_results error={error[-500:]}")

    mode_count = max(set(unique_counts), key=unique_counts.count)
    stable_frames = sum(1 for count in unique_counts if count == mode_count)
    stable_ratio = stable_frames / processed
    expected_matches = None
    expected_ratio = None
    if args.expected_persons >= 0:
        expected_matches = sum(1 for count in unique_counts if count == args.expected_persons)
        expected_ratio = expected_matches / processed

    captured_total = mailbox.latest_seq
    report = {
        "stage": 3,
        "camera": camera.camera_id,
        "backend": "RF-DETR-S PyTorch CUDA latest-frame-only live truth",
        "tracker": False,
        "model_shape_hw": [args.height, args.width],
        "pipe_shape_wh": [args.pipe_width, args.pipe_height],
        "person_threshold": float(args.person_threshold),
        "dedupe": {
            "iou": float(args.iou),
            "containment": float(args.containment),
            "center_distance": float(args.center),
        },
        "summary": {
            "elapsed_sec": round(elapsed, 3),
            "captured_frames": captured_total,
            "processed_frames": processed,
            "skipped_latest_frames": skipped_total,
            "processed_fps": round(processed / elapsed, 3),
            "mode_unique_persons": mode_count,
            "stable_frames": stable_frames,
            "stable_ratio": round(stable_ratio, 6),
            "unique_count_min": min(unique_counts),
            "unique_count_max": max(unique_counts),
            "infer_ms_mean": round(statistics.mean(infer_times), 3),
            "infer_ms_p95": round(_percentile(infer_times, 0.95), 3),
            "input_age_ms_mean": round(statistics.mean(input_ages), 3),
            "input_age_ms_p95": round(_percentile(input_ages, 0.95), 3),
            "expected_persons": None if args.expected_persons < 0 else args.expected_persons,
            "expected_matches": expected_matches,
            "expected_ratio": None if expected_ratio is None else round(expected_ratio, 6),
        },
        "frames": frame_reports,
    }
    report_path = args.output_dir / "live_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    expected_text = (
        "disabled"
        if expected_ratio is None
        else f"{args.expected_persons}:{expected_matches}/{processed}({expected_ratio*100.0:.1f}%)"
    )
    print(
        "STEP3_RESULT "
        f"camera={camera.camera_id} elapsed={elapsed:.1f}s captured={captured_total} "
        f"processed={processed} skipped={skipped_total} processed_fps={processed/elapsed:.2f} "
        f"mode_unique={mode_count} stable={stable_frames}/{processed}({stable_ratio*100.0:.1f}%) "
        f"count_range={min(unique_counts)}-{max(unique_counts)} "
        f"mean_ms={statistics.mean(infer_times):.1f} p95_ms={_percentile(infer_times, 0.95):.1f} "
        f"age_mean_ms={statistics.mean(input_ages):.1f} age_p95_ms={_percentile(input_ages, 0.95):.1f} "
        f"expected={expected_text}",
        flush=True,
    )
    print(f"STEP3_JSON={report_path}", flush=True)
    print(f"STEP3_SAMPLES={sample_dir}", flush=True)
    print("STEP3_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
