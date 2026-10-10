"""Record synchronized, annotated human-validation evidence from existing preview IPC."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QGuiApplication, QColor, QFont, QImage, QPainter, QPen

from services.camera_v11.ui_preview_ipc_v1 import PreviewFrame, PreviewFrameReader
from scripts.run_six_camera_v13_wall_candidate import CAMERAS

ROOT = Path(__file__).resolve().parents[1]
PHASES = {"empty-baseline-start", "empty-baseline-end", "person-entry",
          "person-positive-start", "person-positive-end", "movement",
          "person-exit", "post-exit"}
WIDTH, HEIGHT = 960, 540


def annotate(frame: PreviewFrame) -> tuple[QImage, list[dict]]:
    image = QImage(frame.payload, frame.width, frame.height, frame.stride, QImage.Format.Format_RGB32).copy()
    snapshot = frame.tracker_snapshot or {}
    geometry_width = int(snapshot.get("bbox_coordinate_width") or snapshot.get("pipeline_width") or frame.width)
    geometry_height = int(snapshot.get("bbox_coordinate_height") or snapshot.get("pipeline_height") or frame.height)
    rows = snapshot.get("objects", [])
    painter = QPainter(image)
    painter.setFont(QFont("Sans Serif", 10, QFont.Weight.DemiBold))
    painted = []
    for row in rows:
        try:
            x1, y1, x2, y2 = (float(value) for value in row["bbox"])
            native_id = int(row["native_track_id"])
            source = str(row.get("bbox_source", "TRACKER"))
        except (KeyError, TypeError, ValueError):
            continue
        scale_x, scale_y = frame.width / geometry_width, frame.height / geometry_height
        left, top, right, bottom = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
        if not (0 <= left < right <= frame.width and 0 <= top < bottom <= frame.height):
            continue
        fallback = source in {"PGIE_SHADOW_FALLBACK", "PGIE_VERIFIED_LOWCONF_FALLBACK"}
        color = QColor(255, 170, 35) if fallback else QColor(38, 230, 93)
        painter.setPen(QPen(color, 3 if frame.width >= 900 else 2))
        painter.drawRect(QRectF(left, top, right - left, bottom - top))
        painter.drawText(QRectF(left, max(0.0, top - 22), 330, 22),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"{'FALLBACK' if fallback else 'TRACK'} native:{native_id}")
        painted.append({"native_track_id": native_id, "bbox": [x1, y1, x2, y2],
                        "bbox_source": source,
                        "detector_confidence": row.get("detector_confidence"),
                        "tracker_confidence": row.get("tracker_confidence")})
    painter.end()
    return image, painted


def image_bytes(image: QImage) -> bytes:
    bits = image.constBits()
    return bits.tobytes()


def snapshot_jpeg(image: QImage, path: Path) -> None:
    if not image.save(str(path), "JPG", 94):
        raise RuntimeError(f"failed to save review screenshot: {path.name}")


def ffmpeg_command(path: Path) -> list[str]:
    common = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
              "-f", "rawvideo", "-pixel_format", "bgr0", "-video_size", f"{WIDTH}x{HEIGHT}",
              "-framerate", "20", "-i", "pipe:0", "-an"]
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True, check=True).stdout
    if "h264_nvenc" in encoders:
        return common + ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
                         "-rc", "constqp", "-qp", "26", "-bf", "0", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", str(path)]
    return common + ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                     "-crf", "25", "-threads", "1", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", str(path)]


class CameraRecorder:
    def __init__(self, camera_id: str, root: Path):
        self.camera_id = camera_id
        self.root = root / camera_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.video_path = self.root / "entry-exit-annotated.mp4"
        self.log_path = self.root / "encoder.log"
        self.log = self.log_path.open("x")
        self.encoder = subprocess.Popen(ffmpeg_command(self.video_path), stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL, stderr=self.log, bufsize=0)
        self.rows = (self.root / "machine-rows.jsonl").open("x")
        self.markers = (self.root / "markers.jsonl").open("x")
        self.last_sequence: int | None = None
        self.frame_count = 0
        self.sequence_gaps = 0
        self.last_frame: PreviewFrame | None = None
        self.last_image: QImage | None = None
        self.last_objects: list[dict] = []
        self.first_monotonic_ns: int | None = None
        self.last_monotonic_ns: int | None = None
        self.positive_native_ids: set[int] = set()
        self.empty_baseline_count: int | None = None
        self.post_exit_count: int | None = None
        self.validation_path = self.root / "validation-result.json"
        self._write_row({"event": "recorder_started", "camera_id": camera_id,
                         "monotonic_ns": time.monotonic_ns(), "codec": "h264",
                         "resolution": [WIDTH, HEIGHT], "transport": "shared-memory preview IPC"})

    def _write_row(self, row: dict) -> None:
        self.rows.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        self.rows.flush()

    def add_frame(self, frame: PreviewFrame) -> None:
        if frame.width != WIDTH or frame.height != HEIGHT or not frame.tracker_snapshot:
            raise RuntimeError(f"{self.camera_id} preview is not the expected 960x540 synchronized v5 frame")
        snapshot = frame.tracker_snapshot
        exact = (int(snapshot.get("source_frame_num", -1)) == frame.source_frame_num
                 and int(snapshot.get("frame_num", -1)) == frame.tracker_frame_num
                 and int(snapshot.get("pts_ns", -1)) == frame.pts_ns)
        if not exact:
            raise RuntimeError(f"{self.camera_id} image/metadata sequence mismatch")
        if self.last_sequence is not None and frame.sequence > self.last_sequence + 1:
            self.sequence_gaps += frame.sequence - self.last_sequence - 1
        image, painted = annotate(frame)
        if self.encoder.stdin is None:
            raise RuntimeError("ffmpeg encoder stdin closed")
        self.encoder.stdin.write(image_bytes(image))
        self.frame_count += 1
        self.last_sequence = frame.sequence
        self.last_frame = frame
        self.last_image = image
        self.last_objects = painted
        self.first_monotonic_ns = self.first_monotonic_ns or time.monotonic_ns()
        self.last_monotonic_ns = time.monotonic_ns()
        self._write_row({"event": "frame", "camera_id": self.camera_id,
            "capture_monotonic_ns": self.last_monotonic_ns, "shared_sequence": frame.sequence,
            "source_frame_num": frame.source_frame_num, "tracker_frame_num": frame.tracker_frame_num,
            "pts_ns": frame.pts_ns, "width": frame.width, "height": frame.height,
            "stride": frame.stride, "bbox_coordinate_width": snapshot.get("bbox_coordinate_width"),
            "bbox_coordinate_height": snapshot.get("bbox_coordinate_height"),
            "source_frame_width": snapshot.get("source_frame_width"),
            "source_frame_height": snapshot.get("source_frame_height"),
            "generation": snapshot.get("generation"),
            "object_count": len(snapshot.get("objects", [])), "painted_boxes": painted,
            "detector_proposals": snapshot.get("detector_proposals", []),
            "image_bbox_sync": exact})

    def mark(self, phase: str) -> dict:
        if self.last_frame is None or self.last_image is None:
            raise RuntimeError(f"{self.camera_id} has no current preview frame to mark")
        event = {"event": phase, "camera_id": self.camera_id,
                 "monotonic_ns": time.monotonic_ns(), "shared_sequence": self.last_frame.sequence,
                 "source_frame_num": self.last_frame.source_frame_num, "pts_ns": self.last_frame.pts_ns,
                 "object_count": len(self.last_objects), "objects": self.last_objects}
        if phase == "empty-baseline-start":
            snapshot_jpeg(self.last_image, self.root / "empty-baseline.jpg")
            self.empty_baseline_count = len(self.last_objects)
        elif phase == "person-positive-start":
            snapshot_jpeg(self.last_image, self.root / "person-positive.jpg")
            self.positive_native_ids = {int(row["native_track_id"]) for row in self.last_objects}
        elif phase == "post-exit":
            snapshot_jpeg(self.last_image, self.root / "post-exit.jpg")
            self.post_exit_count = len(self.last_objects)
            positive_identities_gone = not bool(
                self.positive_native_ids & {int(row["native_track_id"]) for row in self.last_objects})
            result = {
                "status": "REVIEW_REQUIRED",
                "camera_id": self.camera_id,
                "empty_baseline_object_count": self.empty_baseline_count,
                "positive_native_track_ids": sorted(self.positive_native_ids),
                "post_exit_object_count": self.post_exit_count,
                "positive_native_ids_absent_after_exit": positive_identities_gone,
                "review_required": ["visible person confirmation", "bbox placement", "duplicates/background",
                                    "stable movement tracking", "post-exit visual stale-box check"],
            }
            self.validation_path.write_text(json.dumps(result, indent=2) + "\n")
            event["positive_native_ids_absent_after_exit"] = positive_identities_gone
        self.markers.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
        self.markers.flush()
        return event

    def close(self) -> dict:
        if self.encoder.stdin:
            self.encoder.stdin.close()
        code = self.encoder.wait(timeout=30)
        self.rows.close()
        self.markers.close()
        self.log.close()
        result = {"camera_id": self.camera_id, "video": str(self.video_path),
                  "frame_count": self.frame_count, "sequence_gaps": self.sequence_gaps,
                  "encoder_exit": code}
        if code == 0:
            rate = self.frame_count / max(0.001, (self.last_monotonic_ns - self.first_monotonic_ns) / 1e9) if self.frame_count > 1 else 0
            result["capture_fps"] = rate
        return result


def command_reader(commands: queue.Queue, stop_event: threading.Event) -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        commands.put(line)
        if line.lower() == "stop":
            stop_event.set()
            return


def run(output: Path, selected_cameras: tuple[str, ...] = CAMERAS) -> int:
    # QPainter can draw rectangles without an application object, but text/font
    # access is not safe until Qt's GUI application/font database is initialized.
    # This process is evidence-only and never creates a visible window.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication([sys.argv[0]])
    output = output.resolve()
    if not output.is_relative_to(ROOT / ".runtime") or output.exists():
        raise ValueError("evidence directory must be a new path under .runtime")
    output.mkdir(parents=True)
    recorders: dict[str, CameraRecorder] = {}
    last_sequences: dict[str, int | None] = {camera: None for camera in selected_cameras}
    commands: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    ready = threading.Condition()
    ready_cameras: set[str] = set()
    camera_locks = {camera: threading.Lock() for camera in selected_cameras}
    worker_errors: list[str] = []
    worker_error_lock = threading.Lock()
    workers: list[threading.Thread] = []
    thread = threading.Thread(target=command_reader, args=(commands, stop_event), daemon=True)
    started_ns = time.monotonic_ns()
    event_log = (output / "operator-events.jsonl").open("x")

    def capture_camera(camera: str) -> None:
        reader = PreviewFrameReader(
            f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin")
        try:
            while not stop_event.is_set():
                frame = reader.read_latest(max_age_sec=1.0, after_sequence=last_sequences[camera],
                                           metadata_only=False)
                if frame is None:
                    time.sleep(0.001)
                    continue
                last_sequences[camera] = frame.sequence
                recorder = recorders.get(camera)
                if recorder is None:
                    recorder = CameraRecorder(camera, output)
                    recorders[camera] = recorder
                    with ready:
                        ready_cameras.add(camera)
                        ready.notify_all()
                with camera_locks[camera]:
                    recorder.add_frame(frame)
        except Exception as exc:
            with worker_error_lock:
                worker_errors.append(f"{camera}: {type(exc).__name__}: {exc}")
            stop_event.set()
            with ready:
                ready.notify_all()
        finally:
            reader.close()

    for camera in selected_cameras:
        worker = threading.Thread(target=capture_camera, args=(camera,),
                                  name=f"capture-{camera}", daemon=True)
        workers.append(worker)
        worker.start()
    with ready:
        deadline = time.monotonic() + 15.0
        while len(ready_cameras) < len(selected_cameras) and not worker_errors:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready.wait(timeout=remaining)
    if len(ready_cameras) != len(selected_cameras):
        stop_event.set()
        for worker in workers:
            worker.join(timeout=3)
        event_log.close()
        (output / "capture-summary.json").write_text(json.dumps({
            "started_monotonic_ns": started_ns, "stopped_monotonic_ns": time.monotonic_ns(),
            "ready_cameras": sorted(ready_cameras), "errors": worker_errors,
        }, indent=2) + "\n")
        raise RuntimeError("did not receive fresh synchronized preview frames for all six cameras")

    thread.start()
    print(f"HUMAN_VALIDATION_RECORDER_READY cameras={','.join(selected_cameras)}; commands: mark CAM-XX <phase>; stop", flush=True)
    try:
        while not stop_event.is_set():
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                parts = command.split()
                if len(parts) == 1 and parts[0].lower() == "stop":
                    stop_event.set()
                    break
                if (len(parts) != 3 or parts[0].lower() != "mark"
                        or parts[1] not in selected_cameras or parts[2] not in PHASES):
                    print(f"Invalid command. Use: mark {'|'.join(selected_cameras)} <phase> or stop", flush=True)
                    continue
                camera, phase = parts[1], parts[2]
                if camera not in recorders:
                    print(f"{camera} has not produced a current preview yet", flush=True)
                    continue
                with camera_locks[camera]:
                    marker = recorders[camera].mark(phase)
                event_log.write(json.dumps(marker, sort_keys=True) + "\n")
                event_log.flush()
                print(f"MARKED {camera} {phase} seq={marker['shared_sequence']} objects={marker['object_count']}", flush=True)
            if worker_errors:
                raise RuntimeError("preview evidence worker failed: " + "; ".join(worker_errors))
            time.sleep(0.005)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=5)
        summaries = []
        for recorder in recorders.values():
            summaries.append(recorder.close())
        event_log.close()
        (output / "capture-summary.json").write_text(json.dumps({
            "started_monotonic_ns": started_ns, "stopped_monotonic_ns": time.monotonic_ns(),
            "cameras": summaries, "errors": worker_errors,
            "all_encoders_ok": all(row["encoder_exit"] == 0 for row in summaries),
        }, indent=2) + "\n")
    del app
    return 0 if summaries and all(row["encoder_exit"] == 0 for row in summaries) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", default=",".join(CAMERAS),
                        help="comma-separated cameras to capture from shared-memory previews; no RTSP is opened")
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.cameras.split(",") if value.strip())
    if not selected or len(set(selected)) != len(selected) or any(camera not in CAMERAS for camera in selected):
        parser.error("--cameras must be a nonempty unique subset of the six configured camera IDs")
    try:
        return run(args.output, selected)
    except Exception as exc:
        print(f"HUMAN_VALIDATION_CAPTURE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
