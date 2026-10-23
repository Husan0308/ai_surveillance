"""Private, exact-frame visual evidence for the Entrance live review.

The recorder accepts only TRACK metadata whose source ID, input-frame index,
source PTS/DTS, and monotonic arrival timestamp match the corresponding raw
frame held by the live bridge. It writes a bounded, lossless native BGR frame
sequence, JPEG annotated review images, and a JSONL crosswalk; it does not
rewrite tracker IDs or camera metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any

import cv2
import numpy as np

from .native_frame_bridge import SOURCE_SIZE, detector_box_to_source


CAMERA_BY_SOURCE_ID = {0: "CAM-02", 1: "CAM-05"}
SOURCE_ID_BY_CAMERA = {camera: source_id for source_id, camera in CAMERA_BY_SOURCE_ID.items()}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def annotate_native_frame(frame: np.ndarray, camera_id: str,
                         objects: list[dict[str, Any]], *,
                         person_class_id: int = 0,
                         source_size: tuple[int, int] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Copy a native BGR frame and draw camera-local tracker boxes on the copy."""
    dimensions = source_size or SOURCE_SIZE.get(camera_id)
    if dimensions is None:
        raise ValueError(f"source geometry is required for camera {camera_id}")
    width, height = dimensions
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.shape != (height, width, 3) or not frame.flags.c_contiguous):
        raise ValueError(f"{camera_id}: evidence frame must be contiguous native uint8 BGR {width}x{height}")
    if objects and camera_id not in SOURCE_SIZE:
        raise ValueError(f"detector-to-native bbox mapping is not validated for {camera_id}")

    annotated = frame.copy()
    mapped_objects: list[dict[str, Any]] = []
    for raw in objects:
        if int(raw.get("class_id", -1)) != person_class_id:
            continue
        native = detector_box_to_source(raw, camera_id)
        coords = [float(native.get(key, 0.0)) for key in ("left", "top", "width", "height")]
        if not all(np.isfinite(value) for value in coords):
            raise ValueError("non-finite tracker box cannot be written to visual evidence")
        x, y, box_w, box_h = coords
        x1 = max(0, min(width - 1, int(round(x))))
        y1 = max(0, min(height - 1, int(round(y))))
        x2 = max(0, min(width - 1, int(round(x + box_w))))
        y2 = max(0, min(height - 1, int(round(y + box_h))))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 240, 128), 4)
        track_id = raw.get("tracker_id")
        label = f"LOCAL NvDCF {track_id if track_id is not None else 'pending'} · {float(raw.get('confidence', 0.0)):.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_x = max(0, min(x1, width - text_w - 8))
        label_y = max(text_h + baseline + 6, y1)
        cv2.rectangle(annotated, (label_x, label_y - text_h - baseline - 6),
                      (label_x + text_w + 8, label_y + 4), (0, 0, 0), -1)
        cv2.putText(annotated, label, (label_x + 4, label_y - baseline), font,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        mapped_objects.append({
            "class_id": int(raw.get("class_id", -1)),
            "confidence": float(raw.get("confidence", 0.0)),
            "tracker_id": None if track_id is None else int(track_id),
            "detector_box_1920x1080": {key: float(raw.get(key, 0.0))
                                       for key in ("left", "top", "width", "height")},
            "native_box": {key: float(native.get(key, 0.0))
                           for key in ("left", "top", "width", "height")},
        })
    return annotated, mapped_objects


class FrameEvidenceCapture:
    """Write a bounded, lossless, frame-matched raw/annotated image sequence."""

    def __init__(self, output_dir: Path, *, sample_every_frames: int = 4,
                 max_samples_per_camera: int = 220, queue_size: int = 8,
                 person_class_id: int = 0,
                 camera_by_source_id: dict[int, str] | None = None,
                 source_size_by_camera: dict[str, tuple[int, int]] | None = None):
        if sample_every_frames < 1 or max_samples_per_camera < 1 or queue_size < 1:
            raise ValueError("evidence sampling and queue bounds must be positive")
        camera_map = dict(camera_by_source_id or CAMERA_BY_SOURCE_ID)
        sizes = dict(source_size_by_camera or SOURCE_SIZE)
        if (not camera_map or len(set(camera_map.values())) != len(camera_map)
                or set(camera_map.values()) != set(sizes)):
            raise ValueError("every evidence source must map to exactly one camera with declared dimensions")
        for source_id, camera in camera_map.items():
            if int(source_id) < 0 or not re.fullmatch(r"CAM-\d{2}", camera):
                raise ValueError("evidence camera mapping contains an invalid source or camera ID")
            width, height = sizes[camera]
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError(f"{camera}: source dimensions must be positive")
        self.output_dir = Path(output_dir)
        self.sample_every_frames = int(sample_every_frames)
        self.max_samples_per_camera = int(max_samples_per_camera)
        self.person_class_id = int(person_class_id)
        self.camera_by_source_id = {int(source_id): camera for source_id, camera in camera_map.items()}
        self.source_id_by_camera = {camera: source_id for source_id, camera in self.camera_by_source_id.items()}
        self.source_size_by_camera = {camera: (int(size[0]), int(size[1]))
                                      for camera, size in sizes.items()}
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=int(queue_size))
        self._lock = threading.Lock()
        self._counts = {
            camera: {
                "offered": 0, "submitted": 0, "written": 0,
                "sample_stride_skipped": 0, "cap_skipped": 0,
                "queue_dropped": 0, "writer_error_skipped": 0, "write_failed": 0,
                "queue_high_water": 0, "bytes_written": 0,
                "submit_latency_ns": [], "queue_wait_ns": [], "write_latency_ns": [],
            }
            for camera in self.source_id_by_camera
        }
        self._submitted_per_camera = {camera: 0 for camera in self.source_id_by_camera}
        self.error: str | None = None
        self._closed = False
        self._writer_start_ns: int | None = None
        self._writer_end_ns: int | None = None
        self.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        os.chmod(self.output_dir, 0o700)
        for camera in self.source_id_by_camera:
            for kind in ("raw", "annotated"):
                directory = self.output_dir / camera / kind
                directory.mkdir(parents=True, exist_ok=False, mode=0o700)
                os.chmod(directory, 0o700)
            os.chmod(self.output_dir / camera, 0o700)
        self._thread = threading.Thread(target=self._write_loop, name="entrance-visual-evidence", daemon=True)
        self._thread.start()

    def submit(self, source_id: int, frame_index: int, frame: np.ndarray | None,
               metadata: dict[str, Any], *, cached_source_pts_ns: int,
               cached_source_dts_ns: int, cached_arrival_mono_ns: int,
               force: bool = False) -> bool:
        """Non-blockingly queue an exact frame/metadata pair for sparse evidence.

        The source array is held by reference, not copied on the tracker callback.
        Callers must pass an immutable cached frame (the live bridge owns its
        decoded BGR allocation and never mutates it). Slow disk/JPEG work remains
        on the dedicated writer thread. False means skipped or explicitly
        dropped evidence; tracking continues and the per-camera counters explain
        why.
        """
        submit_start_ns = time.monotonic_ns()
        if self._closed:
            raise RuntimeError("visual evidence writer is already closed")
        if source_id not in self.camera_by_source_id:
            raise ValueError(f"unexpected evidence source_id {source_id}")
        camera_id = self.camera_by_source_id[source_id]
        if int(metadata.get("source_id", -1)) != source_id:
            raise ValueError("tracker metadata camera source_id does not match cached source frame")
        if metadata.get("input_frame_index") is None or int(metadata["input_frame_index"]) != frame_index:
            raise ValueError("tracker metadata frame index does not match cached source frame")
        pts = metadata.get("source_pts_ns")
        dts = metadata.get("source_dts_ns")
        arrival = metadata.get("arrival_mono_ns")
        if pts is None or int(pts) < 0 or int(pts) != int(cached_source_pts_ns):
            raise ValueError("tracker metadata source PTS does not match cached source frame")
        if dts is None or int(dts) != int(cached_source_dts_ns):
            raise ValueError("tracker metadata source DTS does not match cached source frame")
        if arrival is None or int(arrival) != int(cached_arrival_mono_ns):
            raise ValueError("tracker metadata arrival time does not match cached source frame")
        with self._lock:
            self._counts[camera_id]["offered"] += 1
        if not force and frame_index % self.sample_every_frames:
            with self._lock:
                self._counts[camera_id]["sample_stride_skipped"] += 1
                self._counts[camera_id]["submit_latency_ns"].append(time.monotonic_ns() - submit_start_ns)
            return False
        with self._lock:
            if self._submitted_per_camera[camera_id] >= self.max_samples_per_camera:
                self._counts[camera_id]["cap_skipped"] += 1
                self._counts[camera_id]["submit_latency_ns"].append(time.monotonic_ns() - submit_start_ns)
                return False
            if self.error:
                self._counts[camera_id]["writer_error_skipped"] += 1
                self._counts[camera_id]["submit_latency_ns"].append(time.monotonic_ns() - submit_start_ns)
                return False
        width, height = self.source_size_by_camera[camera_id]
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.shape != (height, width, 3) or not frame.flags.c_contiguous):
            raise ValueError(f"{camera_id}: unexpected source-frame geometry/layout")
        objects = [dict(obj) for obj in metadata.get("objects", [])]
        item = {
            "source_id": int(source_id), "camera_id": camera_id,
            "frame_index": int(frame_index), "source_pts_ns": int(pts),
            "source_dts_ns": int(dts), "arrival_mono_ns": int(arrival),
            "frame_num": int(metadata.get("frame_num", -1)),
            "buf_pts": int(metadata.get("buf_pts", -1)),
            "batch_id": int(metadata.get("batch_id", -1)),
            "stage_mono_ns": int(metadata.get("stage_mono_ns", -1)),
            "queued_mono_ns": time.monotonic_ns(),
            "sample_reason": "identity_set_change" if force else "periodic_stride",
            "objects": objects,
            "raw_frame": frame,
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._counts[camera_id]["queue_dropped"] += 1
                self._counts[camera_id]["submit_latency_ns"].append(time.monotonic_ns() - submit_start_ns)
            return False
        with self._lock:
            self._submitted_per_camera[camera_id] += 1
            self._counts[camera_id]["submitted"] += 1
            self._counts[camera_id]["queue_high_water"] = max(
                self._counts[camera_id]["queue_high_water"], self._queue.qsize()
            )
            self._counts[camera_id]["submit_latency_ns"].append(time.monotonic_ns() - submit_start_ns)
        return True

    def _write_loop(self) -> None:
        manifests: dict[str, Any] = {}
        active_camera: str | None = None
        try:
            self._writer_start_ns = time.monotonic_ns()
            for camera in self.source_id_by_camera:
                manifest = self.output_dir / camera / "frames.jsonl"
                manifests[camera] = manifest.open("x", encoding="utf-8")
                os.chmod(manifest, 0o600)
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    active_camera = item["camera_id"]
                    self._write_item(item, manifests[item["camera_id"]])
                    active_camera = None
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                if active_camera is not None:
                    self._counts[active_camera]["write_failed"] += 1
            # Account for queued samples abandoned after an I/O failure. They
            # are evidence losses only; the tracker callback never waits here.
            while True:
                try:
                    abandoned = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if abandoned is not None:
                        with self._lock:
                            self._counts[abandoned["camera_id"]]["writer_error_skipped"] += 1
                finally:
                    self._queue.task_done()
        finally:
            self._writer_end_ns = time.monotonic_ns()
            for stream in manifests.values():
                try:
                    stream.flush()
                    stream.close()
                except Exception:
                    pass

    def _write_item(self, item: dict[str, Any], manifest) -> None:
        write_start_ns = time.monotonic_ns()
        camera = item["camera_id"]
        raw = item.pop("raw_frame")
        annotated, mapped_objects = annotate_native_frame(
            raw, camera, item["objects"], person_class_id=self.person_class_id,
            source_size=self.source_size_by_camera[camera],
        )
        # Store the source BGR bytes directly: exact pixels, no source-image
        # compression cost, and a deterministic byte hash. JPEG is limited to
        # the human-review overlay; the source record remains lossless.
        raw_bytes = raw.tobytes(order="C")
        ok_annotated, annotated_jpeg = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok_annotated:
            raise RuntimeError("OpenCV failed to encode annotated evidence JPEG")
        annotated_bytes = annotated_jpeg.tobytes()
        stem = f"frame_{item['frame_index']:06d}"
        raw_path = self.output_dir / camera / "raw" / f"{stem}.bgr"
        annotated_path = self.output_dir / camera / "annotated" / f"{stem}.jpg"
        for path, data in ((raw_path, raw_bytes), (annotated_path, annotated_bytes)):
            with path.open("xb") as stream:
                stream.write(data)
            os.chmod(path, 0o600)
        raw_digest = _sha256(raw_bytes)
        annotated_digest = _sha256(annotated.tobytes(order="C"))
        record = {
            "camera_id": camera, "source_id": item["source_id"],
            "input_frame_index": item["frame_index"],
            "source_pts_ns": item["source_pts_ns"], "source_dts_ns": item["source_dts_ns"],
            "arrival_mono_ns": item["arrival_mono_ns"], "deepstream_frame_num": item["frame_num"],
            "deepstream_buf_pts": item["buf_pts"], "batch_id": item["batch_id"],
            "tracker_stage_mono_ns": item["stage_mono_ns"],
            "sample_reason": item["sample_reason"],
            "image_geometry": {"width": raw.shape[1], "height": raw.shape[0], "pixel_format": "BGR8"},
            "raw_bgr": str(raw_path.relative_to(self.output_dir)),
            "annotated_jpeg": str(annotated_path.relative_to(self.output_dir)),
            "raw_frame_sha256": raw_digest,
            "annotated_frame_sha256": annotated_digest,
            "raw_bgr_sha256": raw_digest,
            "annotated_jpeg_sha256": _sha256(annotated_bytes),
            "tracker_objects_raw_1920x1080": item["objects"],
            "tracker_objects_native": mapped_objects,
        }
        manifest_line = json.dumps(record, separators=(",", ":")) + "\n"
        manifest.write(manifest_line)
        manifest.flush()
        os.chmod(self.output_dir / camera / "frames.jsonl", 0o600)
        write_latency_ns = time.monotonic_ns() - write_start_ns
        queue_wait_ns = max(0, write_start_ns - int(item["queued_mono_ns"]))
        with self._lock:
            self._counts[camera]["written"] += 1
            self._counts[camera]["bytes_written"] += len(raw_bytes) + len(annotated_bytes) + len(manifest_line.encode("utf-8"))
            self._counts[camera]["queue_wait_ns"].append(queue_wait_ns)
            self._counts[camera]["write_latency_ns"].append(write_latency_ns)

    def close(self, timeout_sec: float = 10.0) -> dict[str, Any]:
        if self._closed:
            return self.summary()
        self._closed = True
        if self._thread.is_alive():
            try:
                self._queue.put(None, timeout=max(0.1, timeout_sec / 2))
            except queue.Full as exc:
                raise RuntimeError("visual evidence queue did not drain before shutdown") from exc
        self._thread.join(timeout=max(0.1, timeout_sec))
        if self._thread.is_alive():
            raise RuntimeError("visual evidence writer did not stop before shutdown timeout")
        if self.error:
            raise RuntimeError(f"visual evidence writer failed: {self.error}")
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = {camera: dict(values) for camera, values in self._counts.items()}
        latency_stats = {}
        for camera, values in counts.items():
            latency_stats[camera] = {
                field: _latency_percentiles_ms(values.pop(field))
                for field in ("submit_latency_ns", "queue_wait_ns", "write_latency_ns")
            }
        total_bytes = sum(values["bytes_written"] for values in counts.values())
        writer_elapsed = (
            max(0, self._writer_end_ns - self._writer_start_ns)
            if self._writer_start_ns is not None and self._writer_end_ns is not None else 0
        )
        return {
            "output_dir": str(self.output_dir),
            "format": "lossless native BGR source frames + annotated JPEG review images + per-camera JSONL crosswalk",
            "sample_every_frames": self.sample_every_frames,
            "max_samples_per_camera": self.max_samples_per_camera,
            "queue_size": self._queue.maxsize,
            "queue_depth_at_close": self._queue.qsize(),
            "camera_counts": counts,
            "latency_ms": latency_stats,
            "total_bytes_written": total_bytes,
            "writer_elapsed_seconds": writer_elapsed / 1e9,
            "writer_throughput_mib_per_second": (
                total_bytes / (1024 * 1024) / (writer_elapsed / 1e9) if writer_elapsed else 0.0
            ),
            "error": self.error,
            "private_permissions_expected": "directories 0700; evidence files 0600",
        }

    @property
    def queue_depth(self) -> int:
        """Current pending evidence count, intended for offline/live telemetry."""
        return self._queue.qsize()


def _latency_percentiles_ms(values_ns: list[int]) -> dict[str, float | int | None]:
    if not values_ns:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    data = np.asarray(values_ns, dtype=np.float64) / 1e6
    return {
        "count": int(data.size),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }
