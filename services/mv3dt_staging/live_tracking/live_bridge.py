"""Isolated Entrance Room live RTSP → appsrc → DeepStream 9.1 UI.

The host side uses protected camera settings/ownership and the existing RTSP
decoder. Rectified mode requires its validated empirical bridge; native-frame
mode performs only full-FOV scale normalization and shows boxes on the matching
native-resolution frame. A disposable DeepStream container receives BGR pixels
over stdin (no camera URL or credentials), converts to NVMM, batches through
legacy nvstreammux, and runs the selected staging PGIE plus local NvDCF.
MV3DT/global IDs and production map output are excluded. An explicit staging
option may display the saved LIVE_NATIVE_ALIGNMENT candidate after exact
source/frame/PTS validation; map locations remain EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent
DS_IMAGE = "nvcr.io/nvidia/deepstream:9.1-triton-multiarch"
DS_IMAGE_ID = "sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b"
YOLO_DS_IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
YOLO_DS_IMAGE_ID = "sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
MODEL_DIR = Path("/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/PeopleNetTransformer")
BRIDGE_DEFAULT = Path("/tmp/entrance-live-rectification-bridge-20261019/rectification_bridge.py")
CAMERAS = ("CAM-02", "CAM-05")
CODECS = {"CAM-02": "h264", "CAM-05": "h265"}
NATIVE_SIZE = {"CAM-02": (3200, 1800), "CAM-05": (2560, 1440)}
FRAME_HEADER = struct.Struct("<IHHIIIQqqqQ")
FRAME_MAGIC = 0x454E5446
FRAME_VERSION = 1
SENTINEL_SOURCE = 0xFFFF
SOURCE_BY_ID = {0: "CAM-02", 1: "CAM-05"}
AUTHORIZED_RTSP_MAX_SECONDS = 60.0
# Stop RTSP frame acquisition three seconds before the requested maximum. The
# reserve covers the bounded appsink read timeout plus normal GStreamer NULL
# transition/teardown; every actual close timestamp is still checked against
# the full authorization cap and written to the private run report.
RTSP_CAPTURE_STOP_SECONDS = 57.0
SHADOW_INFERENCE_CACHE_FRAMES = 20


def redact_sensitive(text: str) -> str:
    text = re.sub(r"(?i)(rtsps?://)[^@/\s]+@", r"\1***@", text)
    text = re.sub(r'(?i)user-(?:id|pw)="[^"]*"', 'credential="***"', text)
    text = re.sub(r"(?i)(password|passwd|token)=([^\s&]+)", r"\1=***", text)
    return text


def redact(text: str) -> str:
    return redact_sensitive(text)[-1000:]


class StartupTrace:
    """Append-only, private startup/lifecycle diagnostics for a single run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        self._lock = threading.Lock()
        self.error: str | None = None

    def record(self, stage: str, **fields: Any) -> None:
        row = {
            "stage": stage,
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "host_pid": os.getpid(),
            "thread": threading.current_thread().name,
        }
        row.update(fields)
        try:
            encoded = json.dumps(row, sort_keys=True, default=str)
            encoded = redact_sensitive(encoded)
            with self._lock:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, (encoded + "\n").encode("utf-8"))
                finally:
                    os.close(fd)
                os.chmod(self.path, 0o600)
        except Exception as exc:  # Diagnostics must not alter pipeline behavior.
            self.error = type(exc).__name__


def source_panel_mode_label(*, native_frame: bool, recorded_mode: bool) -> str:
    """Keep recorded replay visibly distinct from a physical live source."""
    if recorded_mode:
        return "RECORDED RAW SOURCE · scale-only inference" if native_frame else "RECORDED rectified frame"
    return "LIVE RAW SOURCE · scale-only inference" if native_frame else "LIVE rectified frame"


def camera_track_label(local_id: Any, confidence: float,
                       provisional_global_id: str | None = None) -> str:
    """Put the unverified candidate beside—not in place of—the local ID."""
    local = "pending" if local_id is None else str(local_id)
    provisional = f" · {provisional_global_id}?" if provisional_global_id else ""
    # Keep the per-box label short enough to distinguish nearby tracks. The
    # adjacent purple camera badge spells out PROVISIONAL; this label retains
    # the local NvDCF namespace plus the candidate ID.
    return f"local {local}{provisional} · det {float(confidence):.2f}"


def import_bridge(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"validated rectification bridge missing: {path}")
    spec = importlib.util.spec_from_file_location("entrance_live_rectification_bridge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the validated rectification bridge")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def smoke_gate_passed(stats: dict[int, dict[str, Any]], baseline: dict[int, dict[str, int]]) -> bool:
    """Require advancing, correctly attributed pipeline output, not zero drops.

    The live path intentionally uses leaky, bounded queues. A late tracker
    result whose source frame has already left the small preview cache is
    counted, but it must not be drawn against another frame. Such an expected
    drop is not itself a pipeline failure.
    """
    return all(
        stats[sid]["mux_frames"] > baseline[sid]["mux_frames"]
        and stats[sid]["pgie_frames"] > baseline[sid]["pgie_frames"]
        and stats[sid]["tracker_frames"] > baseline[sid]["tracker_frames"]
        and stats[sid]["display_frame_matched"] > baseline[sid]["display_frame_matched"]
        and stats[sid]["source_id_mismatch"] == 0
        and stats[sid]["mux_frame_num_nonmonotonic"] == 0
        and stats[sid]["input_frame_nonmonotonic"] == 0
        for sid in (0, 1)
    )


def safe_for_extended_review(gpu_samples: list[dict[str, Any]], limit_percent: int = 95) -> bool:
    """Fail closed if device-wide load is unavailable or repeatedly saturated."""
    recent = [sample for sample in gpu_samples[-3:] if sample.get("status") == "AVAILABLE"]
    if len(recent) < 3:
        return False
    return sum(int(sample["gpu_utilization_percent"]) >= limit_percent for sample in recent) < 2


def safe_for_smoke(gpu_samples: list[dict[str, Any]], limit_percent: int = 95) -> bool:
    """Require three fresh, available baseline samples below the existing overload limit."""
    recent = [sample for sample in gpu_samples[-3:] if sample.get("status") == "AVAILABLE"]
    return len(recent) == 3 and all(
        int(sample["gpu_utilization_percent"]) < limit_percent for sample in recent
    )


def provisional_identity_summary(
    mapping: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    """Format accepted provisional links without collapsing camera-local IDs."""
    grouped: dict[str, dict[str, list[str]]] = {}
    for camera in CAMERAS:
        for local_id, entry in mapping.get(camera, {}).items():
            gid = str(entry.get("provisional_global_id", ""))
            if gid:
                grouped.setdefault(gid, {name: [] for name in CAMERAS})[camera].append(str(local_id))
    return [
        f"{gid}? · CAM-02 local {','.join(sorted(cameras['CAM-02'])) or '—'} / "
        f"CAM-05 local {','.join(sorted(cameras['CAM-05'])) or '—'} · map points remain separate"
        for gid, cameras in sorted(grouped.items())
    ]


def provisional_camera_overlay(
    mapping: dict[str, dict[str, dict[str, Any]]],
    camera_id: str,
    current_local_ids: set[str] | list[str],
) -> str | None:
    """Compact, explicit candidate badge for one camera's current objects."""
    if camera_id not in CAMERAS:
        raise ValueError(f"unsupported camera namespace: {camera_id}")
    current = {str(value) for value in current_local_ids}
    items = []
    for local_id, entry in mapping.get(camera_id, {}).items():
        if str(local_id) not in current:
            continue
        gid = str(entry.get("provisional_global_id", ""))
        if gid:
            items.append((gid, str(local_id)))
    if not items:
        return None
    return "PROVISIONAL " + "; ".join(
        f"{gid}? · local {local_id}" for gid, local_id in sorted(set(items))
    )


def _rect_overlap_area(first: tuple[int, int, int, int],
                      second: tuple[int, int, int, int]) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return max(0, right - left) * max(0, bottom - top)


def clamp_box_to_frame(
    frame_rect: tuple[int, int, int, int],
    box: tuple[float, float, float, float],
) -> tuple[int, int, int, int, bool] | None:
    """Clamp only the rendered box copy; tracker and map metadata stay unchanged."""
    frame_x, frame_y, frame_width, frame_height = (int(value) for value in frame_rect)
    left, top, width, height = (float(value) for value in box)
    if frame_width <= 0 or frame_height <= 0 or width <= 0 or height <= 0:
        return None
    right = left + width
    bottom = top + height
    clipped_left = max(float(frame_x), left)
    clipped_top = max(float(frame_y), top)
    clipped_right = min(float(frame_x + frame_width), right)
    clipped_bottom = min(float(frame_y + frame_height), bottom)
    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        return None
    result = (
        int(round(clipped_left)), int(round(clipped_top)),
        max(1, int(round(clipped_right - clipped_left))),
        max(1, int(round(clipped_bottom - clipped_top))),
        clipped_left != left or clipped_top != top
        or clipped_right != right or clipped_bottom != bottom,
    )
    return result


def layout_track_label_rectangles(
    frame_rect: tuple[int, int, int, int],
    boxes: list[tuple[float, float, float, float]],
    requested_widths: list[int],
    *,
    label_height: int = 20,
    gap: int = 2,
) -> list[dict[str, Any]]:
    """Place labels inside the displayed image, avoiding boxes/other labels where possible.

    Inputs and outputs are display-pixel coordinates. Crowded scenes may not
    admit a conflict-free placement; the least-overlapping bounded location is
    selected and reported instead of moving or clamping a detection box.
    """
    frame_x, frame_y, frame_width, frame_height = (int(value) for value in frame_rect)
    if frame_width <= 0 or frame_height <= 0 or label_height <= 0:
        return []
    if len(boxes) != len(requested_widths):
        raise ValueError("one requested label width is required per detection box")

    frame_bounds = (frame_x, frame_y, frame_width, frame_height)
    box_rectangles = [
        (
            int(round(x)), int(round(y)), max(0, int(round(width))),
            max(0, int(round(height))),
        )
        for x, y, width, height in boxes
    ]
    placed: list[tuple[int, int, int, int] | None] = [None] * len(boxes)
    occupied: list[tuple[int, int, int, int]] = []
    label_max_y = frame_y + max(0, frame_height - label_height)

    order = sorted(
        range(len(boxes)),
        key=lambda index: (boxes[index][1], boxes[index][0], index),
    )
    for index in order:
        x, y, box_width, _box_height = boxes[index]
        width = min(frame_width, max(1, int(requested_widths[index])))
        left_bound = frame_x
        right_bound = frame_x + frame_width - width
        preferred_x = min(max(int(round(x)), left_bound), right_bound)
        preferred_y = min(
            max(int(round(y)) - label_height - gap, frame_y), label_max_y,
        )
        x_candidates = list(dict.fromkeys((
            preferred_x,
            min(max(int(round(x + box_width - width)), left_bound), right_bound),
            min(max(int(round(x + (box_width - width) / 2)), left_bound), right_bound),
            left_bound,
            right_bound,
        )))
        step = label_height + gap
        y_offsets = [0]
        for step_index in range(1, max(5, len(boxes) + 2)):
            y_offsets.extend((step * step_index, -step * step_index))
        y_candidates = list(dict.fromkeys(
            min(max(preferred_y + offset, frame_y), label_max_y)
            for offset in y_offsets
        ))

        def score(candidate: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            label_overlap = sum(_rect_overlap_area(candidate, old) for old in occupied)
            detection_overlap = sum(
                _rect_overlap_area(candidate, box)
                for box_index, box in enumerate(box_rectangles)
                if box_index != index
            )
            displacement = abs(candidate[0] - preferred_x) + abs(candidate[1] - preferred_y)
            return label_overlap, detection_overlap, displacement, candidate[1]

        candidates = [
            (candidate_x, candidate_y, width, label_height)
            for candidate_y in y_candidates
            for candidate_x in x_candidates
        ]
        best = min(candidates, key=score)
        conflict = score(best)[0] > 0 or score(best)[1] > 0
        placed[index] = best
        occupied.append(best)

    result = []
    for index, rect in enumerate(placed):
        assert rect is not None
        result.append({
            "rect": rect,
            "conflict": bool(
                any(
                    other_index != index and other is not None
                    and _rect_overlap_area(rect, other) > 0
                    for other_index, other in enumerate(placed)
                )
                or any(
                    _rect_overlap_area(rect, box) > 0
                    for box_index, box in enumerate(box_rectangles)
                    if box_index != index
                )
            ),
        })
    return result


def paint_camera_track_labels(
    painter: Any,
    frame_rect: tuple[int, int, int, int],
    objects: list[dict[str, Any]],
    identity_mapping: dict[str, dict[str, Any]],
    font_metrics: Any,
    qt_elide_right: Any,
    background_color: Any,
    text_color: Any,
) -> list[dict[str, Any]]:
    """Shared renderer used by the live pane and deterministic UI-only tests."""
    labels = []
    boxes = []
    widths = []
    normalized_objects = []
    for obj in objects:
        clipped = clamp_box_to_frame(
            frame_rect,
            (float(obj.get("left", 0)), float(obj.get("top", 0)),
             float(obj.get("width", 0)), float(obj.get("height", 0))),
        )
        if clipped is None:
            continue
        left, top, box_width, box_height, clipped_flag = clipped
        tracker_id = obj.get("tracker_id")
        shadow = identity_mapping.get(str(tracker_id)) if tracker_id is not None else None
        gid = str(shadow["provisional_global_id"]) if shadow else None
        text = camera_track_label(obj.get("tracker_id"), float(obj.get("confidence", 0.0)), gid)
        width = min(
            int(frame_rect[2]),
            max(108, int(font_metrics.horizontalAdvance(text)) + 8),
        )
        labels.append({"tracker_id": tracker_id, "text": text})
        boxes.append((float(left), float(top), float(box_width), float(box_height)))
        widths.append(width)
        normalized_objects.append({
            **obj, "left": left, "top": top, "width": box_width,
            "height": box_height, "box_was_clamped": clipped_flag,
        })

    placements = layout_track_label_rectangles(frame_rect, boxes, widths)
    rendered = []
    for label, obj, placement in zip(labels, normalized_objects, placements):
        x, y, width, height = placement["rect"]
        text = font_metrics.elidedText(label["text"], qt_elide_right, max(1, width - 8))
        painter.fillRect(x, y, width, height, background_color)
        painter.setPen(text_color)
        painter.drawText(x + 4, y + 15, text)
        rendered.append({**label, "text": text, "box_was_clamped": obj["box_was_clamped"], **placement})
    return rendered


def paint_camera_provisional_badge(
    painter: Any,
    frame_rect: tuple[int, int, int, int],
    identity_mapping: dict[str, dict[str, Any]],
    camera_id: str,
    current_local_ids: set[str] | list[str],
    font_metrics: Any,
    qt_elide_right: Any,
    background_color: Any,
    text_color: Any,
) -> dict[str, Any] | None:
    """Draw the explicit provisional-ID badge for one camera namespace."""
    badge = provisional_camera_overlay(identity_mapping, camera_id, current_local_ids)
    if not badge:
        return None
    frame_x, frame_y, frame_width, frame_height = (int(value) for value in frame_rect)
    if frame_width <= 0 or frame_height <= 0:
        return None
    text = font_metrics.elidedText(badge, qt_elide_right, max(1, frame_width - 16))
    width = min(frame_width, int(font_metrics.horizontalAdvance(text)) + 16)
    height = 22
    x = frame_x + frame_width - width
    y = min(frame_y + 6, frame_y + max(0, frame_height - height))
    painter.fillRect(x, y, width, height, background_color)
    painter.setPen(text_color)
    painter.drawText(x + 8, y + 16, text)
    return {"text": text, "rect": (x, y, width, height)}


def visible_provisional_identity_groups(
    mapping: dict[str, dict[str, dict[str, Any]]],
    displayed_local_ids: dict[str, set[str]],
    display_frame_keys: dict[str, tuple[int, int, int] | None],
    map_states: dict[str, dict[str, Any] | None],
    video_identity_mappings: dict[str, dict[str, dict[str, Any]]] | None = None,
    map_identity_mappings: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Return only provisional links visible on both exact video/map frames.

    A visibility result requires each camera's current video source/frame/PTS
    key to equal its own map-projection snapshot, and the associated local ID
    to occur in both the displayed tracker objects and projected markers.
    Camera clocks are deliberately not compared with one another.
    """
    grouped: dict[str, dict[str, set[str]]] = {}
    for camera in CAMERAS:
        for local_id, entry in mapping.get(camera, {}).items():
            gid = str(entry.get("provisional_global_id", ""))
            if gid:
                grouped.setdefault(gid, {name: set() for name in CAMERAS})[camera].add(str(local_id))

    visible: list[dict[str, Any]] = []
    for gid, per_camera in sorted(grouped.items()):
        selected: dict[str, str] = {}
        frame_keys: dict[str, dict[str, int]] = {}
        for camera in CAMERAS:
            state = map_states.get(camera) or {}
            key = display_frame_keys.get(camera)
            if key is None or state.get("status") != "LIVE":
                break
            try:
                map_key = (int(state["source_id"]), int(state["frame_index"]),
                           int(state["source_pts_ns"]))
            except (KeyError, TypeError, ValueError):
                break
            if map_key != tuple(int(value) for value in key):
                break
            map_ids = {str(marker.get("local_tracker_id"))
                       for marker in state.get("markers", [])}
            candidates = (per_camera[camera]
                          & {str(value) for value in displayed_local_ids.get(camera, set())}
                          & map_ids)
            if video_identity_mappings is not None:
                candidates = {
                    local_id for local_id in candidates
                    if str(video_identity_mappings.get(camera, {}).get(local_id, {}).get(
                        "provisional_global_id", ""
                    )) == gid
                }
            if map_identity_mappings is not None:
                candidates = {
                    local_id for local_id in candidates
                    if str(map_identity_mappings.get(camera, {}).get(local_id, {}).get(
                        "provisional_global_id", ""
                    )) == gid
                }
            if not candidates:
                break
            local_id = sorted(candidates)[0]
            selected[camera] = local_id
            frame_keys[camera] = {
                "source_id": map_key[0], "frame_index": map_key[1],
                "source_pts_ns": map_key[2],
            }
        if len(selected) == len(CAMERAS):
            visible.append({
                "provisional_global_id": gid,
                "camera_local_ids": selected,
                "per_camera_exact_frame_keys": frame_keys,
                "map_points_fused": False,
                "identity_accuracy": "UNVERIFIED",
            })
    return visible


def live_gpu_guard_reason(gpu_samples: list[dict[str, Any]], limit_percent: int = 95) -> str | None:
    """Stop a bounded live session on sustained overload or lost telemetry."""
    recent = gpu_samples[-3:]
    if len(recent) < 3:
        return None
    if any(sample.get("status") != "AVAILABLE" for sample in recent):
        return "live GPU telemetry became unavailable for three consecutive samples"
    if sum(int(sample["gpu_utilization_percent"]) >= limit_percent for sample in recent) >= 2:
        return f"live run stopped after sustained device-total GPU utilization at or above {limit_percent}%"
    return None


def post_smoke_action(*, smoke_only: bool, gpu_samples: list[dict[str, Any]]) -> str:
    """Keep an explicitly smoke-only run from entering the longer review."""
    if smoke_only:
        return "STOP_SMOKE_ONLY"
    if not safe_for_extended_review(gpu_samples):
        return "STOP_RESOURCE_GUARD"
    return "CONTINUE_REVIEW"


def rtsp_deadline_ns(first_attempt_ns: int, stop_after_sec: float = RTSP_CAPTURE_STOP_SECONDS) -> int:
    """Return the absolute monotonic deadline for the bounded RTSP session."""
    if first_attempt_ns < 0 or not 0 < stop_after_sec < AUTHORIZED_RTSP_MAX_SECONDS:
        raise ValueError("invalid RTSP attempt time or stop-before-cap interval")
    return int(first_attempt_ns + round(stop_after_sec * 1_000_000_000))


def rtsp_close_within_authorized_cap(first_attempt_ns: int, close_return_ns: int,
                                     max_seconds: float = AUTHORIZED_RTSP_MAX_SECONDS) -> bool:
    """Conservatively include construction and teardown in the authorized duration."""
    if first_attempt_ns < 0 or close_return_ns < first_attempt_ns or max_seconds <= 0:
        return False
    return close_return_ns <= first_attempt_ns + round(max_seconds * 1_000_000_000)


def validate_live_duration(*, capture_stop_seconds: float, authorized_max_seconds: float,
                           smoke_seconds: float, review_seconds: float) -> None:
    """Validate a bounded live session while reserving teardown inside its cap."""
    if not (0 < capture_stop_seconds < authorized_max_seconds <= AUTHORIZED_RTSP_MAX_SECONDS):
        raise ValueError("capture stop must be positive and below the authorized maximum (at most 60s)")
    if authorized_max_seconds - capture_stop_seconds < 3.0:
        raise ValueError("at least three seconds must remain for RTSP teardown")
    if smoke_seconds <= 0 or review_seconds < 0 or smoke_seconds + review_seconds < 30.0:
        raise ValueError("the planned bounded live review must total at least 30 seconds")


def _system_cpu_ticks() -> tuple[int, int] | None:
    """Return aggregate (total, idle) CPU ticks from Linux procfs."""
    try:
        fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
        if len(values) < 5:
            return None
        return sum(values), values[3] + values[4]
    except (OSError, ValueError, IndexError):
        return None


def _process_rss_mib() -> float | None:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None


def detector_person_class(detector: str) -> int:
    if detector == "yolo26m":
        return 0
    if detector == "peoplenet":
        return 1
    raise ValueError(f"unsupported detector: {detector}")


def percentile_summary_ns(values_ns: list[int] | deque[int]) -> dict[str, Any] | str:
    values_ms = [int(value) / 1_000_000 for value in values_ns if value is not None and int(value) >= 0]
    if not values_ms:
        return "NOT_MEASURED"
    return {
        "samples": len(values_ms),
        "p50_ms": round(float(np.percentile(values_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(values_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(values_ms, 99)), 3),
        "max_ms": round(max(values_ms), 3),
    }


def high_overlap_pairs(records: list[dict[str, Any]], *, class_id: int,
                       threshold: float = 0.90) -> int:
    """Count same-frame box-overlap candidates; this is not a false-positive label."""
    count = 0
    for record in records:
        objects = [obj for obj in record.get("objects", [])
                   if int(obj.get("class_id", -1)) == class_id]
        for index, left in enumerate(objects):
            lx1, ly1 = float(left.get("left", 0)), float(left.get("top", 0))
            lx2 = lx1 + float(left.get("width", 0))
            ly2 = ly1 + float(left.get("height", 0))
            left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
            for right in objects[index + 1:]:
                rx1, ry1 = float(right.get("left", 0)), float(right.get("top", 0))
                rx2 = rx1 + float(right.get("width", 0))
                ry2 = ry1 + float(right.get("height", 0))
                intersection = (max(0.0, min(lx2, rx2) - max(lx1, rx1))
                               * max(0.0, min(ly2, ry2) - max(ly1, ry1)))
                right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
                union = left_area + right_area - intersection
                if union > 0 and intersection / union >= threshold:
                    count += 1
    return count


def track_frame_summary(records: list[dict[str, Any]], *, class_id: int) -> dict[str, dict[str, Any]]:
    """Summarize raw camera-local IDs and frame spans without identity claims."""
    tracks: dict[str, dict[int, set[int]]] = {"0": {}, "1": {}}
    for record in records:
        sid = str(record.get("source_id"))
        if sid not in tracks or record.get("input_frame_index") is None:
            continue
        frame_index = int(record["input_frame_index"])
        for obj in record.get("objects", []):
            track_id = obj.get("tracker_id")
            if int(obj.get("class_id", -1)) == class_id and track_id is not None:
                tracks[sid].setdefault(int(track_id), set()).add(frame_index)
    result: dict[str, dict[str, Any]] = {}
    for sid, by_id in tracks.items():
        result[sid] = {}
        for track_id, frames in sorted(by_id.items()):
            ordered = sorted(frames)
            if not ordered:
                continue
            result[sid][str(track_id)] = {
                "frames_observed": len(frames),
                "first_input_frame": ordered[0],
                "last_input_frame": ordered[-1],
                "observed_frame_gaps": sum(max(0, b - a - 1) for a, b in zip(ordered, ordered[1:])),
            }
    return result


def _gpu_sample() -> dict[str, Any]:
    sample: dict[str, Any] = {"monotonic_ns": time.monotonic_ns(), "scope": "device-total"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.5, check=True,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"},
        )
        fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        if len(fields) == 3:
            sample.update({"gpu_utilization_percent": int(fields[0]),
                           "memory_used_mib": int(fields[1]),
                           "memory_total_mib": int(fields[2]), "status": "AVAILABLE"})
        else:
            sample.update({"status": "UNAVAILABLE", "reason": "unexpected nvidia-smi output"})
    except Exception as exc:
        sample.update({"status": "UNAVAILABLE", "reason": type(exc).__name__})
    return sample


def read_host_memory_summary() -> dict[str, Any]:
    """Read host-visible memory and this process cgroup counters, if exposed."""
    result: dict[str, Any] = {}
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            if key in values:
                result[key.lower() + "_mib"] = round(values[key] / 1024, 2)
    except OSError:
        result["host_meminfo"] = "UNAVAILABLE"
    cgroup = Path("/sys/fs/cgroup")
    for file_name in ("memory.current", "memory.max", "memory.events"):
        try:
            content = (cgroup / file_name).read_text().strip()
        except OSError:
            continue
        if file_name == "memory.events":
            events = {}
            for line in content.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    events[parts[0]] = int(parts[1])
            result["current_process_cgroup_memory_events"] = events
        else:
            result["current_process_cgroup_" + file_name.replace(".", "_")] = content
    result["scope_note"] = "host /proc/meminfo plus current runner cgroup; not historical R6 container accounting"
    return result


def collect_gpu_baseline(*, samples: int = 3, interval_sec: float = 1.0) -> list[dict[str, Any]]:
    rows = []
    for index in range(samples):
        rows.append(_gpu_sample())
        if index + 1 < samples:
            time.sleep(interval_sec)
    return rows


def docker_command(*, mux_only: bool, output_root: Path, detector: str = "peoplenet",
                   yolo_artifacts: Path | None = None) -> list[str]:
    if detector not in {"peoplenet", "yolo26m"}:
        raise ValueError("unsupported detector")
    source = str(APP_DIR.resolve())
    image = DS_IMAGE
    pgie_config = "/workspace/app/config_pgie_pnt_batch2.txt"
    mounts: list[str] = []
    pull_policy: list[str] = []
    if not mux_only and detector == "yolo26m":
        if yolo_artifacts is None:
            raise ValueError("YOLO26m requires explicit --yolo-artifacts under .runtime")
        from services.mv3dt_staging.live_tracking.yolo_candidate import (
            CONTAINER_CONFIG, stage_yolo_batch_two,
        )
        profile = stage_yolo_batch_two(yolo_artifacts, output_root, engine_only=True)
        image = YOLO_DS_IMAGE
        pgie_config = CONTAINER_CONFIG
        pull_policy = ["--pull=never"]
        mounts = [
            "--mount", f"type=bind,source={profile['model_dir']},target=/models,readonly",
            "--mount", f"type=bind,source={profile['config_file']},target={CONTAINER_CONFIG},readonly",
        ]
    elif not mux_only:
        models = str(MODEL_DIR.resolve())
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(f"existing PeopleNet Transformer model directory missing: {MODEL_DIR}")
        mounts = ["--mount", f"type=bind,source={models},target=/workspace/models/PeopleNetTransformer,readonly"]
    compile_and_run = (
        "set -euo pipefail; "
        "diag() { local stage=\"$1\"; local wall; wall=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
        "local mono; mono=$(cut -d' ' -f1 /proc/uptime); "
        "printf 'DS_BOOT {\\\"stage\\\":\\\"%s\\\",\\\"utc\\\":\\\"%s\\\",\\\"monotonic_seconds\\\":%s,\\\"pid\\\":%s,\\\"ppid\\\":%s}\\n' "
        "\"$stage\" \"$wall\" \"$mono\" \"$$\" \"$PPID\" >&2; }; "
        "diag CONTAINER_ENTRY; diag COMPILE_BEGIN; "
        "g++ -std=c++17 -O2 -Wall -Wextra "
        "-I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-app-1.0 gstreamer-video-1.0) "
        "/workspace/app/deepstream_appsrc.cpp "
        "-L/opt/nvidia/deepstream/deepstream/lib "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib "
        "-lnvds_meta -lnvdsgst_meta $(pkg-config --libs gstreamer-app-1.0 gstreamer-video-1.0) "
        "-o /tmp/entrance_ds_appsrc; "
        "diag COMPILE_COMPLETE; "
        "exec /tmp/entrance_ds_appsrc "
        + ("--mux-only" if mux_only else f"--pgie-config {pgie_config}")
    )
    return [
        "docker", "run", "--rm", *pull_policy, "--init", "--interactive", "--gpus", "all",
        "--network", "none", "--env", "USE_NEW_NVSTREAMMUX=no",
        "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "--mount", f"type=bind,source={source},target=/workspace/app,readonly",
        *mounts,
        "--workdir", "/workspace/app", "--entrypoint", "/bin/bash",
        image, "-lc", compile_and_run,
    ]


class DeepStreamProcess:
    def __init__(self, *, mux_only: bool, output_root: Path, detector: str = "peoplenet",
                 yolo_artifacts: Path | None = None, startup_trace: StartupTrace | None = None):
        self.output_root = output_root
        self.startup_trace = startup_trace
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.output_root, 0o700)
        self.stdout_path = self.output_root / "deepstream.stdout.log"
        self.stderr_path = self.output_root / "deepstream.stderr.log"
        self.image = YOLO_DS_IMAGE if detector == "yolo26m" and not mux_only else DS_IMAGE
        self.image_id = YOLO_DS_IMAGE_ID if detector == "yolo26m" and not mux_only else DS_IMAGE_ID
        self._stdout_file = self.stdout_path.open("w", encoding="utf-8")
        self._stderr_file = self.stderr_path.open("w", encoding="utf-8")
        os.chmod(self.stdout_path, 0o600)
        os.chmod(self.stderr_path, 0o600)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8192)
        self.ready = threading.Event()
        self.done = threading.Event()
        self.done_message_seen = threading.Event()
        self.failed = threading.Event()
        self._input_lock = threading.Lock()
        self._first_stage_records: set[tuple[str, Any]] = set()
        self._input_end_recorded = False
        command = docker_command(mux_only=mux_only, output_root=output_root,
                                 detector=detector, yolo_artifacts=yolo_artifacts)
        safe_env = {
            "DISPLAY": os.environ.get("DISPLAY"),
            "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM"),
            "USE_NEW_NVSTREAMMUX": "no",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        }
        if self.startup_trace:
            self.startup_trace.record(
                "DEEPSTREAM_PROCESS_CREATE_REQUESTED",
                parent_pid=os.getpid(), command=redact(shlex.join(command)),
                image=self.image, image_id=self.image_id, network="none",
                host_network=False, host_ipc=False, selected_environment=safe_env,
                stdout_log=str(self.stdout_path), stderr_log=str(self.stderr_path),
            )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env={**os.environ, "USE_NEW_NVSTREAMMUX": "no"},
        )
        if self.startup_trace:
            self.startup_trace.record(
                "DEEPSTREAM_PROCESS_CREATED", parent_pid=os.getpid(),
                docker_cli_pid=self.process.pid, process_poll=self.process.poll(),
            )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _event(self, record: dict[str, Any]) -> None:
        try:
            self.events.put_nowait(record)
        except queue.Full:
            self.failed.set()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for raw in iter(self.process.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                self._stdout_file.write(line + "\n")
                self._stdout_file.flush()
                if line.startswith("READY "):
                    try: self._event({"type": "READY", **json.loads(line[6:])})
                    except json.JSONDecodeError: self._event({"type": "READY", "raw": line})
                    if self.startup_trace:
                        self.startup_trace.record(
                            "READY_RECEIVED", docker_cli_pid=self.process.pid,
                            ready_payload=redact(line[6:]),
                        )
                    self.ready.set()
                elif line.startswith("DONE "):
                    try: self._event({"type": "DONE", **json.loads(line[5:])})
                    except json.JSONDecodeError: self._event({"type": "DONE", "raw": line})
                    self.done_message_seen.set()
                    self.done.set()
                    if self.startup_trace:
                        self.startup_trace.record(
                            "DONE_RECEIVED", docker_cli_pid=self.process.pid,
                            done_payload=redact(line[5:]),
                        )
                else:
                    prefix, separator, body = line.partition(" ")
                    if separator and prefix in {"MUX", "PGIE", "TRACK", "ERROR"}:
                        try: record = {"type": prefix, **json.loads(body)}
                        except json.JSONDecodeError: self._event({"type": prefix, "raw": redact(line)})
                        else:
                            self._event(record)
                            if prefix in {"MUX", "PGIE", "TRACK"}:
                                key = (prefix, record.get("source_id"))
                                if key not in self._first_stage_records:
                                    self._first_stage_records.add(key)
                                    if self.startup_trace:
                                        self.startup_trace.record(
                                            f"FIRST_{prefix}_OUTPUT",
                                            docker_cli_pid=self.process.pid,
                                            source_id=record.get("source_id"),
                                            frame_num=record.get("frame_num"),
                                            input_frame_index=record.get("input_frame_index"),
                                            source_pts_ns=record.get("source_pts_ns"),
                                            buf_pts=record.get("buf_pts"),
                                        )
                            elif self.startup_trace:
                                self.startup_trace.record(
                                    "DEEPSTREAM_ERROR_EVENT", docker_cli_pid=self.process.pid,
                                    error=redact(json.dumps(record, default=str)),
                                )
        finally:
            self._stdout_file.flush()
            self.done.set()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            for raw in iter(self.process.stderr.readline, b""):
                line = redact(raw.decode("utf-8", errors="replace").rstrip())
                self._stderr_file.write(line + "\n")
                self._stderr_file.flush()
                if line.startswith("DS_BOOT ") and self.startup_trace:
                    try:
                        boot = json.loads(line[len("DS_BOOT "):])
                    except json.JSONDecodeError:
                        self.startup_trace.record("CONTAINER_BOOT_DIAGNOSTIC_PARSE_ERROR", line=line)
                    else:
                        stage = str(boot.pop("stage", "UNKNOWN"))
                        self.startup_trace.record(
                            f"CONTAINER_{stage}", docker_cli_pid=self.process.pid,
                            container_shell_pid=boot.get("pid"), container_boot=boot,
                        )
                if line.startswith("DS_DIAG ") and self.startup_trace:
                    try:
                        child = json.loads(line[len("DS_DIAG "):])
                    except json.JSONDecodeError:
                        self.startup_trace.record("CHILD_DIAGNOSTIC_PARSE_ERROR", line=line)
                    else:
                        stage = str(child.pop("stage", "UNKNOWN"))
                        self.startup_trace.record(
                            f"CHILD_{stage}", docker_cli_pid=self.process.pid,
                            child_pid=child.get("pid"), child_diagnostic=child,
                        )
        finally:
            self._stderr_file.flush()

    def wait_ready(self, timeout_sec: float = 30.0) -> None:
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + int(timeout_sec * 1e9)
        deadline_utc = (datetime.now(timezone.utc) + timedelta(seconds=timeout_sec)).isoformat()
        if self.startup_trace:
            self.startup_trace.record(
                "READY_WAIT_STARTED", timeout_sec=timeout_sec,
                deadline_monotonic_ns=deadline_ns, deadline_utc=deadline_utc,
                docker_cli_pid=self.process.pid, child_alive=self.process.poll() is None,
            )
        while not self.ready.is_set():
            remaining = (deadline_ns - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                break
            self.ready.wait(min(0.1, remaining))
            returncode = self.process.poll()
            if returncode is not None and not self.ready.is_set():
                if self.startup_trace:
                    self.startup_trace.record(
                        "PROCESS_EXITED_BEFORE_READY", docker_cli_pid=self.process.pid,
                        exit_code=returncode,
                    )
                self.failed.set()
                self.stop(timeout_sec=5.0)
                raise RuntimeError(
                    f"DeepStream 9.1 process exited with code {returncode} before READY"
                )
        if self.ready.is_set():
            if self.startup_trace:
                self.startup_trace.record(
                    "READY_WAIT_COMPLETED", elapsed_ms=round(
                        (time.monotonic_ns() - started_ns) / 1e6, 3
                    ), docker_cli_pid=self.process.pid,
                )
            return
        if self.startup_trace:
            self.startup_trace.record(
                "READY_TIMEOUT", timeout_sec=timeout_sec,
                deadline_monotonic_ns=deadline_ns, docker_cli_pid=self.process.pid,
                process_alive=self.process.poll() is None,
            )
        self.failed.set()
        self.stop(timeout_sec=5.0)
        tail = self.stderr_path.read_text(errors="replace")[-2000:] if self.stderr_path.exists() else ""
        raise RuntimeError(f"DeepStream 9.1 did not signal READY within {timeout_sec}s: {redact(tail)}")

    @staticmethod
    def _write_all(fd, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = fd.write(view)
            if written is None or written <= 0:
                raise BrokenPipeError("DeepStream appsrc stdin closed")
            view = view[written:]

    def send_frame(self, source_id: int, frame_index: int, frame: np.ndarray,
                   source_pts_ns: int, source_dts_ns: int, arrival_mono_ns: int) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("DeepStream container exited before appsrc accepted the frame")
        image = np.ascontiguousarray(frame)
        if image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
            raise ValueError(f"appsrc requires 1920x1080 uint8 BGR, got {image.shape}/{image.dtype}")
        data = memoryview(image).cast("B")
        header = FRAME_HEADER.pack(
            FRAME_MAGIC, FRAME_VERSION, source_id, 1920, 1080, 1920 * 3,
            frame_index, source_pts_ns, source_dts_ns, arrival_mono_ns, data.nbytes,
        )
        with self._input_lock:
            self._write_all(self.process.stdin, header)
            self._write_all(self.process.stdin, data)

    def end_input(self) -> None:
        if self.process.stdin is None or self.process.stdin.closed or self.process.poll() is not None:
            return
        sentinel = FRAME_HEADER.pack(FRAME_MAGIC, FRAME_VERSION, SENTINEL_SOURCE,
                                     0, 0, 0, 0, -1, -1, time.monotonic_ns(), 0)
        with self._input_lock:
            try:
                if not self._input_end_recorded:
                    self._input_end_recorded = True
                    if self.startup_trace:
                        self.startup_trace.record(
                            "HOST_EOS_SENTINEL_WRITE_REQUESTED",
                            docker_cli_pid=self.process.pid,
                        )
                self._write_all(self.process.stdin, sentinel)
                self.process.stdin.close()
                if self.startup_trace:
                    self.startup_trace.record(
                        "HOST_EOS_SENTINEL_WRITTEN", docker_cli_pid=self.process.pid,
                    )
            except (BrokenPipeError, OSError, ValueError):
                if self.startup_trace:
                    self.startup_trace.record(
                        "HOST_EOS_SENTINEL_WRITE_FAILED", docker_cli_pid=self.process.pid,
                    )

    def stop(self, timeout_sec: float = 10.0) -> None:
        if self.startup_trace:
            self.startup_trace.record(
                "DEEPSTREAM_STOP_REQUESTED", docker_cli_pid=self.process.pid,
                timeout_sec=timeout_sec, process_poll=self.process.poll(),
                done_message_seen=self.done_message_seen.is_set(),
            )
        self.end_input()
        try:
            self.process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            if self.startup_trace:
                self.startup_trace.record(
                    "HOST_SIGTERM_SENT", target="docker-cli-process",
                    target_pid=self.process.pid, reason="bounded wait expired",
                )
            self.process.terminate()
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                if self.startup_trace:
                    self.startup_trace.record(
                        "HOST_SIGKILL_SENT", target="docker-cli-process",
                        target_pid=self.process.pid, reason="SIGTERM grace expired",
                    )
                self.process.kill()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if self.startup_trace:
                        self.startup_trace.record(
                            "HOST_PROCESS_STILL_ALIVE_AFTER_SIGKILL",
                            target_pid=self.process.pid,
                        )
        self.done.set()
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)
        if self.startup_trace:
            self.startup_trace.record(
                "DEEPSTREAM_PROCESS_STOPPED", docker_cli_pid=self.process.pid,
                exit_code=self.process.poll(),
                done_message_seen=self.done_message_seen.is_set(),
                stdout_reader_alive=self._stdout_thread.is_alive(),
                stderr_reader_alive=self._stderr_thread.is_alive(),
            )
        self._stdout_file.close()
        self._stderr_file.close()


def run_mux_smoke(args) -> int:
    import cv2

    output = Path(args.output).resolve() if args.output else Path(tempfile.mkdtemp(prefix="entrance-appsrc-mux-"))
    os.chmod(output, 0o700)
    bridge_mod = import_bridge(Path(args.bridge).expanduser())
    cameras = [
        (0, "CAM-02", Path(args.cam02_video).expanduser(), "CAM-02"),
        (1, "CAM-05", Path(args.cam05_video).expanduser(), "CAM-05"),
    ]
    captures = []
    for _, camera_id, path, _ in cameras:
        if not path.is_file(): raise FileNotFoundError(f"recorded input missing: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): raise RuntimeError(f"OpenCV could not read recorded {camera_id} frames")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (1920, 1080):
            cap.release(); raise RuntimeError(f"{camera_id} recorded test expected 1920x1080, got {w}x{h}")
        captures.append(cap)
    mux_only = not bool(args.inference)
    pipe = DeepStreamProcess(mux_only=mux_only, output_root=output)
    sent = {0: 0, 1: 0}
    try:
        pipe.wait_ready()
        transforms = {"CAM-02": bridge_mod.RectificationBridge("CAM-02"),
                      "CAM-05": bridge_mod.RectificationBridge("CAM-05")}
        for index in range(args.frames):
            for (source_id, camera_id, _path, _), cap in zip(cameras, captures):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"{camera_id}: recorded decode ended before frame {index}")
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                pts_ns = int(pts_ms * 1_000_000) if np.isfinite(pts_ms) and pts_ms >= 0 else -1
                rectified = transforms[camera_id](frame)
                pipe.send_frame(source_id, index, rectified, pts_ns, -1, time.monotonic_ns())
                sent[source_id] += 1
            time.sleep(0.05)
        pipe.end_input()
        if not pipe.done.wait(20): raise RuntimeError("recorded appsrc mux smoke did not reach EOS")
        try: pipe.process.wait(timeout=10)
        except subprocess.TimeoutExpired: raise RuntimeError("DeepStream reported EOS but did not exit")
        if pipe.process.returncode != 0: raise RuntimeError(f"DeepStream mux smoke exited {pipe.process.returncode}")
    finally:
        for cap in captures: cap.release()
        pipe.stop()
    mux_rows = []
    pgie_rows = []
    track_rows = []
    for record in list(pipe.events.queue):
        if record.get("type") == "MUX": mux_rows.append(record)
        elif record.get("type") == "PGIE": pgie_rows.append(record)
        elif record.get("type") == "TRACK": track_rows.append(record)
    counts = {0: 0, 1: 0}
    frame_sequences = {0: [], 1: []}
    sidecar_ok = True
    for row in mux_rows:
        sid = row.get("source_id")
        if sid in counts:
            counts[sid] += 1
            if row.get("input_frame_index") is None: sidecar_ok = False
            frame_sequences[sid].append((row.get("frame_num"), row.get("input_frame_index"), row.get("buf_pts")))
    if (not args.inference and counts != sent) or (args.inference and any(value == 0 for value in counts.values())) or not sidecar_ok:
        raise RuntimeError(f"mux metadata mismatch: sent={sent}, muxed={counts}, sidecar_ok={sidecar_ok}")
    if args.inference:
        pgie_counts = {sid: sum(1 for row in pgie_rows if row.get("source_id") == sid) for sid in (0, 1)}
        track_counts = {sid: sum(1 for row in track_rows if row.get("source_id") == sid) for sid in (0, 1)}
        if any(value == 0 for value in pgie_counts.values()) or any(value == 0 for value in track_counts.values()):
            raise RuntimeError(f"recorded inference did not reach PGIE/NvDCF for both sources: PGIE={pgie_counts}, NvDCF={track_counts}")
        if any(row.get("input_frame_index") is None for row in pgie_rows + track_rows):
            raise RuntimeError("PGIE/NvDCF output lost its input-frame sidecar crosswalk")
    result = {
        "status": "PASS", "image": DS_IMAGE, "image_id": DS_IMAGE_ID,
        "mux": "legacy via process-scoped USE_NEW_NVSTREAMMUX=no", "batch_size": 2,
        "geometry": "1920x1080 BGR→NV12→NVMM", "sent_by_source": sent,
        "muxed_by_source": counts, "bounded_queue_drops_before_mux": {str(k): sent[k]-counts[k] for k in sent},
        "source_mapping": {"0": "CAM-02", "1": "CAM-05"},
        "mode": "PeopleNet Transformer + local NvDCF" if args.inference else "mux-only",
        "pgie_frames_by_source": ({str(k): v for k, v in pgie_counts.items()} if args.inference else None),
        "tracker_frames_by_source": ({str(k): v for k, v in track_counts.items()} if args.inference else None),
        "recorded_person_detection_observations": (sum(int(row.get("num_obj_meta", 0)) for row in pgie_rows) if args.inference else None),
        "recorded_local_track_observations": (sum(len(row.get("objects", [])) for row in track_rows) if args.inference else None),
        "frame_meta_to_input_crosswalk": {str(sid): frame_sequences[sid] for sid in frame_sequences},
        "source_pts_scope": "recorded-file presentation timestamps; no cross-camera synchronization claim",
        "deepstream_stderr": str(output / "deepstream.stderr.log"),
        "deepstream_stdout": str(output / "deepstream.stdout.log"),
    }
    (output / "mux-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


class CaptureWorker(threading.Thread):
    def __init__(self, controller, source_id: int, camera, settings, transform):
        super().__init__(name=f"entrance-capture-{camera.camera_id}", daemon=True)
        self.controller = controller
        self.source_id = source_id
        self.camera = camera
        self.settings = settings
        self.transform = transform
        self.stop_event = controller.stop_event
        self.frames = 0
        self.started_ns = 0
        self.started_utc = None
        self.connection_attempt_ns = 0
        self.connection_attempt_utc = None
        self.first_frame_ns = 0
        self.first_frame_utc = None
        self.first_frame_pts_ns = None
        self.last_pts_ns = -1
        self.last_frame_ns = 0
        self.last_frame_utc = None
        self.close_started_ns = 0
        self.close_started_utc = None
        self.close_return_ns = 0
        self.close_return_utc = None
        self.capture_initialized = False
        self.capture_close_attempted = False
        self.capture_close_succeeded: bool | None = None
        self.capture_close_error_type: str | None = None
        self.error = ""
        self.state = "CONNECTING"

    def _create_capture(self, config):
        """Create the protected RTSP reader; overridable only for offline lifecycle tests."""
        from services.ml_service.app.deepstream.capture import DeepStreamCapture

        return DeepStreamCapture(
            self.camera.camera_id, self.camera.uri, CODECS[self.camera.camera_id], config,
            transport=self.settings.deepstream.rtsp_transport,
            username=self.camera.username, password=self.camera.password,
            output_bgrx=True, latency_ms=config.latency_ms,
            low_latency_mode=self.camera.decoder_low_latency_mode,
            decoder_extra_surfaces=self.camera.decoder_extra_surfaces,
        )

    def _close_capture(self, capture) -> None:
        """Close the RTSP reader and record failures without exposing source credentials."""
        self.capture_close_attempted = True
        self.close_started_ns = time.monotonic_ns()
        self.close_started_utc = datetime.now(timezone.utc).isoformat()
        try:
            capture.close()
        except Exception as exc:
            self.capture_close_succeeded = False
            self.capture_close_error_type = type(exc).__name__
            message = f"{self.camera.camera_id}: RTSP capture close failed ({self.capture_close_error_type})"
            self.error = self.error or message
            self.state = "ERROR"
            self.stop_event.set()
            self.controller.error_queue.put(message)
        else:
            self.capture_close_succeeded = True
        finally:
            self.close_return_ns = time.monotonic_ns()
            self.close_return_utc = datetime.now(timezone.utc).isoformat()

    def run(self) -> None:
        width, height = NATIVE_SIZE[self.camera.camera_id]
        config = replace(self.settings.deepstream, display_width=width, display_height=height,
                         postdecode_queue_buffers=1, capture_timeout_ms=600)
        capture = None
        try:
            # This timestamp is immediately before the protected loader creates
            # and starts the RTSP GStreamer pipeline. Credentials and URI are
            # deliberately excluded from all timing records.
            self.connection_attempt_ns = time.monotonic_ns()
            self.connection_attempt_utc = datetime.now(timezone.utc).isoformat()
            self.controller.register_rtsp_attempt(self)
            capture = self._create_capture(config)
            self.capture_initialized = True
            while not self.stop_event.is_set():
                if self.controller.rtsp_deadline_reached():
                    break
                ok, frame = capture.read()
                # read() blocks for at most capture_timeout_ms; discard a frame
                # returned after the stop boundary instead of extending capture.
                if self.controller.rtsp_deadline_reached():
                    break
                if not ok or frame is None:
                    if capture.last_error(): raise RuntimeError(capture.last_error())
                    continue
                if frame.ndim != 3 or frame.shape != (height, width, 4):
                    raise RuntimeError(f"{self.camera.camera_id}: decoder returned unexpected BGRx geometry")
                arrival_ns = capture.last_timing.appsink_receive_ns or time.monotonic_ns()
                pts_ns = capture.last_timing.pts_ns
                dts_ns = capture.last_timing.dts_ns
                timing = capture.last_timing
                if not self.first_frame_ns:
                    self.first_frame_ns = arrival_ns
                    self.first_frame_utc = datetime.now(timezone.utc).isoformat()
                    self.first_frame_pts_ns = pts_ns
                stat = self.controller.stats[self.source_id]
                if timing.decoder_reference_ns > 0 and timing.decoder_out_ns >= timing.decoder_reference_ns:
                    stat["decoder_latency_ns"].append(timing.decoder_out_ns - timing.decoder_reference_ns)
                if timing.decoder_out_ns > 0 and arrival_ns >= timing.decoder_out_ns:
                    stat["decoder_to_appsink_ns"].append(arrival_ns - timing.decoder_out_ns)
                source_bgr = frame[..., :3].copy()
                preprocess_start_ns = time.monotonic_ns()
                inference_frame = self.transform(source_bgr)
                preprocess_end_ns = time.monotonic_ns()
                if getattr(self.controller.args, "native_frame", False):
                    # The native display/evidence cache shares this immutable
                    # source allocation. A background evidence writer can hold
                    # it by reference without copying full-resolution pixels
                    # on the tracking callback.
                    source_bgr.setflags(write=False)
                stat["frame_preprocess_ns"].append(preprocess_end_ns - preprocess_start_ns)
                self.frames += 1
                self.last_pts_ns = pts_ns
                self.last_frame_ns = arrival_ns
                self.last_frame_utc = datetime.now(timezone.utc).isoformat()
                self.state = "ONLINE"
                self.controller.queue_frame(
                    self.source_id, self.frames - 1, inference_frame, pts_ns, dts_ns,
                    arrival_ns,
                    display_image=(source_bgr if self.controller.args.native_frame else inference_frame),
                )
        except Exception as exc:
            self.error = redact(f"{type(exc).__name__}: {exc}")
            self.state = "ERROR"
            self.controller.error_queue.put(f"{self.camera.camera_id}: {self.error}")
            self.stop_event.set()
        finally:
            if capture is not None:
                self._close_capture(capture)
            else:
                self.close_started_ns = time.monotonic_ns()
                self.close_started_utc = datetime.now(timezone.utc).isoformat()
                self.close_return_ns = time.monotonic_ns()
                self.close_return_utc = datetime.now(timezone.utc).isoformat()
            if self.state != "ERROR": self.state = "STOPPED"

    def timing_record(self, first_attempt_ns: int | None, deadline_ns: int | None,
                      max_seconds: float = AUTHORIZED_RTSP_MAX_SECONDS) -> dict[str, Any]:
        attempt_to_close = (
            (self.close_return_ns - self.connection_attempt_ns) / 1e9
            if self.connection_attempt_ns and self.close_return_ns else None
        )
        first_attempt_to_close = (
            (self.close_return_ns - first_attempt_ns) / 1e9
            if first_attempt_ns is not None and self.close_return_ns else None
        )
        return {
            "camera": self.camera.camera_id,
            "source_id": self.source_id,
            "connection_attempt_utc": self.connection_attempt_utc,
            "connection_attempt_monotonic_ns": self.connection_attempt_ns or None,
            "first_decoded_frame_utc": self.first_frame_utc,
            "first_decoded_frame_monotonic_ns": self.first_frame_ns or None,
            "first_source_pts_ns": self.first_frame_pts_ns,
            "last_decoded_frame_utc": self.last_frame_utc,
            "last_decoded_frame_monotonic_ns": self.last_frame_ns or None,
            "last_source_pts_ns": self.last_pts_ns if self.frames else None,
            "frame_count": self.frames,
            "capture_close_started_utc": self.close_started_utc,
            "capture_close_returned_utc": self.close_return_utc,
            "capture_close_returned_monotonic_ns": self.close_return_ns or None,
            "capture_initialized": self.capture_initialized,
            "capture_close_attempted": self.capture_close_attempted,
            "capture_close_succeeded": self.capture_close_succeeded,
            "capture_close_error_type": self.capture_close_error_type,
            "attempt_to_close_seconds": round(attempt_to_close, 6) if attempt_to_close is not None else None,
            "earliest_attempt_to_close_seconds": round(first_attempt_to_close, 6)
                if first_attempt_to_close is not None else None,
            "within_authorized_cap": (
                rtsp_close_within_authorized_cap(first_attempt_ns, self.close_return_ns, max_seconds)
                if first_attempt_ns is not None and self.close_return_ns else None
            ),
            "closed_before_capture_deadline": (
                self.close_return_ns <= deadline_ns
                if deadline_ns is not None and self.close_return_ns else None
            ),
            "error": self.error or None,
        }


class RecordedCaptureWorker(CaptureWorker):
    """Recorded-file input adapter feeding the same native DeepStream path.

    It preserves the original decoded-frame ordinal and MP4 PTS, but uses a
    separate local monotonic playback clock. No camera settings, credentials,
    ownership locks, or RTSP sessions are touched.
    """

    def __init__(self, controller, source_id: int, source: dict[str, Any],
                 transform, *, start_frame: int, duration_sec: float,
                 playback_speed: float = 1.0):
        camera = SimpleNamespace(camera_id=str(source["camera_id"]))
        super().__init__(controller, source_id, camera, None, transform)
        self.name = f"entrance-recorded-{camera.camera_id}"
        self.source = source
        self.start_frame = int(start_frame)
        self.duration_sec = float(duration_sec)
        self.playback_speed = float(playback_speed)
        self.playback_start_mono_ns: int | None = None
        self.playback_first_source_pts_ns: int | None = None
        self.source_eos_seen = False
        self.source_close_state = "NOT_CLOSED"
        self.selected_frame_count = 0

    def run(self) -> None:
        from services.mv3dt_staging.live_tracking.recorded_input import (
            GstRecordedFileCapture,
            playback_deadline_ns,
        )

        capture = None
        last_progress = time.monotonic()
        previous_source_pts = -1
        trace = getattr(self.controller, "startup_trace", None)
        startup_stage = "VALIDATE_RECORDED_SOURCE_OPTIONS"
        try:
            if (self.start_frame < 0 or self.duration_sec <= 0
                    or not math.isfinite(self.playback_speed)
                    or not 0.1 <= self.playback_speed <= 4.0):
                raise ValueError("invalid recorded source frame or duration")
            self.connection_attempt_ns = time.monotonic_ns()
            self.connection_attempt_utc = datetime.now(timezone.utc).isoformat()
            if trace:
                trace.record(
                    "RECORDED_SOURCE_WORKER_STARTED",
                    camera=self.camera.camera_id, source_id=self.source_id,
                    source_sha256=self.source.get("sha256"),
                    source_width=self.source.get("width"),
                    source_height=self.source.get("height"),
                    requested_start_frame=self.start_frame,
                    playback_speed_x=self.playback_speed,
                )
            startup_stage = "CREATE_LOCAL_GSTREAMER_DECODER"
            capture = GstRecordedFileCapture(self.source, appsink_buffers=1)
            self.capture_initialized = True
            if trace:
                try:
                    change, current, pending = capture.pipeline.get_state(0)
                    state_record = {
                        "change_result": int(change), "current_state": int(current),
                        "pending_state": int(pending),
                    }
                except Exception as exc:
                    state_record = {"state_read_error": type(exc).__name__}
                trace.record(
                    "RECORDED_SOURCE_INITIALIZED", camera=self.camera.camera_id,
                    source_id=self.source_id, pipeline_state=state_record,
                    decoder=self.source.get("codec"), appsink_buffers=1,
                )
            self.state = "DECODING_TO_START_FRAME"
            startup_stage = "DECODE_TO_REQUESTED_SOURCE_FRAME"
            while not self.stop_event.is_set():
                pull_started_ns = time.monotonic_ns()
                record = capture.pull(timeout_ms=500)
                pull_return_ns = time.monotonic_ns()
                if record is None:
                    if capture.eos_seen:
                        self.source_eos_seen = True
                        if trace:
                            trace.record(
                                "RECORDED_SOURCE_EOS", camera=self.camera.camera_id,
                                source_id=self.source_id,
                                decoded_frames=capture.decoded_frames,
                            )
                        if self.playback_first_source_pts_ns is None:
                            raise RuntimeError(
                                f"{self.camera.camera_id}: source EOS before requested start frame {self.start_frame}"
                            )
                        self.state = "EOS"
                        break
                    if time.monotonic() - last_progress > 10.0:
                        raise TimeoutError(f"{self.camera.camera_id}: local GStreamer decoder stalled")
                    continue
                last_progress = time.monotonic()
                if record.frame_index <= previous_source_pts:
                    raise RuntimeError(f"{self.camera.camera_id}: decoded source frame index repeated or regressed")
                if previous_source_pts >= 0:
                    prior_pts = int(self.source["pts_ns_by_decoded_frame"][previous_source_pts])
                    if record.source_pts_ns <= prior_pts:
                        raise RuntimeError(f"{self.camera.camera_id}: original source PTS regressed")
                previous_source_pts = record.frame_index
                stat = self.controller.stats[self.source_id]
                stat["recorded_gst_pull_ns"].append(pull_return_ns - pull_started_ns)
                if record.frame_index < self.start_frame:
                    continue

                if self.playback_first_source_pts_ns is None:
                    self.playback_first_source_pts_ns = record.source_pts_ns
                    self.playback_start_mono_ns = time.monotonic_ns()
                    self.first_frame_pts_ns = record.source_pts_ns
                    if trace:
                        trace.record(
                            "FIRST_SOURCE_FRAME", camera=self.camera.camera_id,
                            source_id=self.source_id,
                            source_frame_index=record.frame_index,
                            source_pts_ns=record.source_pts_ns,
                            source_dts_ns=record.source_dts_ns,
                            native_shape=list(record.bgr.shape),
                            source_hash=self.source.get("sha256"),
                        )
                source_elapsed_ns = record.source_pts_ns - self.playback_first_source_pts_ns
                if source_elapsed_ns >= int(self.duration_sec * 1_000_000_000):
                    self.state = "BOUNDED_COMPLETE"
                    break
                deadline_ns = playback_deadline_ns(
                    self.playback_start_mono_ns, self.playback_first_source_pts_ns,
                    record.source_pts_ns, self.playback_speed,
                )
                now_ns = time.monotonic_ns()
                if now_ns < deadline_ns and self.stop_event.wait((deadline_ns - now_ns) / 1e9):
                    break
                arrival_ns = time.monotonic_ns()
                stat["recorded_playback_lateness_ns"].append(max(0, arrival_ns - deadline_ns))
                if not self.first_frame_ns:
                    self.first_frame_ns = arrival_ns
                    self.first_frame_utc = datetime.now(timezone.utc).isoformat()
                source_bgr = record.bgr
                preprocess_start_ns = time.monotonic_ns()
                inference_frame = self.transform(source_bgr)
                preprocess_end_ns = time.monotonic_ns()
                source_bgr.setflags(write=False)
                stat["frame_preprocess_ns"].append(preprocess_end_ns - preprocess_start_ns)
                self.frames += 1
                self.selected_frame_count += 1
                self.last_pts_ns = record.source_pts_ns
                self.last_frame_ns = arrival_ns
                self.last_frame_utc = datetime.now(timezone.utc).isoformat()
                self.state = "ONLINE"
                self.controller.record_recorded_frame(
                    self.source_id, self.source, record, arrival_ns, deadline_ns,
                    self.playback_speed,
                )
                self.controller.queue_frame(
                    self.source_id, record.frame_index, inference_frame,
                    record.source_pts_ns, record.source_dts_ns, arrival_ns,
                    display_image=source_bgr,
                )
            startup_stage = "CLOSE_LOCAL_GSTREAMER_DECODER"
        except Exception as exc:
            self.error = redact(f"{type(exc).__name__}: {exc}")
            self.state = "ERROR"
            if trace:
                trace.record(
                    "RECORDED_CAPTURE_WORKER_FAILED", camera=self.camera.camera_id,
                    source_id=self.source_id, failing_stage=startup_stage,
                    error=self.error,
                )
            self.controller.error_queue.put(f"{self.camera.camera_id}: {self.error}")
            self.stop_event.set()
        finally:
            if capture is not None:
                self.capture_close_attempted = True
                self.close_started_ns = time.monotonic_ns()
                self.close_started_utc = datetime.now(timezone.utc).isoformat()
                if trace:
                    trace.record(
                        "RECORDED_SOURCE_CLOSE_REQUESTED", camera=self.camera.camera_id,
                        source_id=self.source_id, eos_seen=self.source_eos_seen,
                    )
                try:
                    close_result = capture.close()
                    self.source_close_state = str(close_result.get("close_state", "UNKNOWN"))
                    self.source_eos_seen = bool(close_result.get("eos_seen", self.source_eos_seen))
                    self.capture_close_succeeded = self.source_close_state == "NULL"
                    if trace:
                        trace.record(
                            "RECORDED_SOURCE_CLOSED", camera=self.camera.camera_id,
                            source_id=self.source_id, close_result=close_result,
                        )
                    if not self.capture_close_succeeded:
                        raise RuntimeError("recorded GStreamer source did not confirm NULL state")
                except Exception as exc:
                    self.capture_close_succeeded = False
                    self.capture_close_error_type = type(exc).__name__
                    self.error = self.error or f"recorded GStreamer close failed ({self.capture_close_error_type})"
                    self.state = "ERROR"
                    self.controller.error_queue.put(self.error)
                    self.stop_event.set()
                finally:
                    self.close_return_ns = time.monotonic_ns()
                    self.close_return_utc = datetime.now(timezone.utc).isoformat()
            else:
                self.close_started_ns = self.close_return_ns = time.monotonic_ns()
                self.close_started_utc = self.close_return_utc = datetime.now(timezone.utc).isoformat()
            if self.state not in {"ERROR", "EOS", "BOUNDED_COMPLETE"}:
                self.state = "STOPPED"

    def recorded_timing_record(self) -> dict[str, Any]:
        return {
            "camera": self.camera.camera_id,
            "source_id": self.source_id,
            "source_path": self.source["path"],
            "source_sha256": self.source["sha256"],
            "requested_start_frame": self.start_frame,
            "first_emitted_source_frame": self.start_frame if self.first_frame_pts_ns is not None else None,
            "first_emitted_source_pts_ns": self.first_frame_pts_ns,
            "last_emitted_source_pts_ns": self.last_pts_ns if self.frames else None,
            "frames_emitted": self.frames,
            "source_eos_seen": self.source_eos_seen,
            "bounded_duration_sec": self.duration_sec,
            "playback_speed_x": self.playback_speed,
            "duration_semantics": "source media seconds; wall runtime is source duration divided by playback speed",
            "playback_start_monotonic_ns": self.playback_start_mono_ns,
            "gstreamer_close_state": self.source_close_state,
            "capture_close_succeeded": self.capture_close_succeeded,
            "error": self.error or None,
        }


class LiveController:
    def __init__(self, args, output: Path):
        self.args = args
        self.output = output
        self.startup_trace: StartupTrace | None = getattr(args, "startup_trace", None)
        self.capture_stop_seconds = float(getattr(args, "capture_stop_seconds", RTSP_CAPTURE_STOP_SECONDS))
        self.authorized_max_seconds = float(getattr(args, "authorized_max_seconds", AUTHORIZED_RTSP_MAX_SECONDS))
        self.stop_event = threading.Event()
        self.error_queue: queue.Queue[str] = queue.Queue()
        self.frame_queues = {0: queue.Queue(maxsize=1), 1: queue.Queue(maxsize=1)}
        self.frame_lock = threading.Lock()
        self.frame_cache = {0: OrderedDict(), 1: OrderedDict()}
        self.shadow_inference_cache = {0: OrderedDict(), 1: OrderedDict()}
        self.identity_shadow = None
        self.identity_shadow_summary: dict[str, Any] | None = None
        self.latest_frame = {0: None, 1: None}
        self.latest_meta = {0: None, 1: None}
        self.person_class_id = detector_person_class(args.detector)
        self.stats = {
            sid: {"frames_captured": 0, "frames_sent": 0, "queue_drops": 0,
                  "mux_frames": 0, "pgie_frames": 0, "pgie_person_detections": 0,
                  "tracker_frames": 0, "tracker_person_observations": 0,
                  "local_tracker_ids": set(), "source_id_mismatch": 0,
                  "mux_frame_num_nonmonotonic": 0, "display_frame_matched": 0,
                  "input_frame_nonmonotonic": 0,
                  "last_input_frame_by_stage": {"MUX": -1, "PGIE": -1, "TRACK": -1},
                  "display_frame_missing": 0, "last_mux_frame_num": -1,
                  "last_meta_mono_ns": 0, "fps_window": deque(maxlen=300),
                  "capture_times": deque(maxlen=300),
                  "decoder_latency_ns": deque(maxlen=10000),
                  "decoder_to_appsink_ns": deque(maxlen=10000),
                  "recorded_gst_pull_ns": deque(maxlen=10000),
                  "recorded_playback_lateness_ns": deque(maxlen=10000),
                  "frame_preprocess_ns": deque(maxlen=10000),
                  "display_frame_age_frames": deque(maxlen=10000),
                  "tracker_event_handling_ns": deque(maxlen=10000),
                  "shadow_frame_lookup_ns": deque(maxlen=10000),
                  "shadow_submit_frame_ns": deque(maxlen=10000),
                  "shadow_cache_hits": 0,
                  "shadow_cache_misses": 0,
                  "pgie_pad_latency_ns": deque(maxlen=10000),
                  "host_input_to_pgie_ns": deque(maxlen=10000),
                  "tracker_pad_latency_ns": deque(maxlen=10000),
                  "host_input_to_tracker_ns": deque(maxlen=10000)}
            for sid in (0, 1)
        }
        self.pipe: DeepStreamProcess | None = None
        self.settings = None
        self.ownership = None
        self.identity_shadow_prewarm: dict[str, Any] | None = None
        self.capture_workers: list[CaptureWorker] = []
        self.writer_thread: threading.Thread | None = None
        self.started_ns = 0
        self.ready_ns = 0
        self.run_phase = "STARTING"
        self.error = ""
        self._processed_events = 0
        self.stopped_ns = 0
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._gpu_stop = threading.Event()
        self._gpu_thread: threading.Thread | None = None
        self._cpu_stop = threading.Event()
        self._cpu_thread: threading.Thread | None = None
        self._rtsp_watchdog_thread: threading.Thread | None = None
        self.gpu_samples: list[dict[str, Any]] = []
        self.cpu_samples: list[dict[str, Any]] = []
        self.paint_latency_ms: deque[float] = deque(maxlen=10000)
        self.event_records: list[dict[str, Any]] = []
        self.recorded_frame_records: list[dict[str, Any]] = []
        self.recorded_source_records: dict[str, dict[str, Any]] = {}
        self.recorded_playback_origins: dict[str, dict[str, int]] = {}
        self.evidence_capture = None
        self.evidence_summary: dict[str, Any] | None = None
        self._evidence_last_tracker_ids: dict[int, tuple[int, ...] | None] = {0: None, 1: None}
        self.events_path = self.output / "live-events.jsonl"
        self._rtsp_attempt_lock = threading.Lock()
        self._rtsp_deadline_ready = threading.Event()
        self.rtsp_first_attempt_ns: int | None = None
        self.rtsp_first_attempt_utc: str | None = None
        self.rtsp_deadline_ns: int | None = None
        self.rtsp_cap_triggered_ns: int | None = None
        self.rtsp_cap_triggered_utc: str | None = None
        self.rtsp_stop_reason = ""
        self.stop_requested_ns: int | None = None
        self.stop_requested_utc: str | None = None
        self.ownership_released_ns: int | None = None
        self.ownership_released_utc: str | None = None
        self._ownership_release_lock = threading.Lock()
        self._ownership_reaper: threading.Thread | None = None

    def register_rtsp_attempt(self, worker: CaptureWorker) -> None:
        with self._rtsp_attempt_lock:
            if self.rtsp_first_attempt_ns is None or worker.connection_attempt_ns < self.rtsp_first_attempt_ns:
                self.rtsp_first_attempt_ns = worker.connection_attempt_ns
                self.rtsp_first_attempt_utc = worker.connection_attempt_utc
                self.rtsp_deadline_ns = rtsp_deadline_ns(
                    worker.connection_attempt_ns, self.capture_stop_seconds
                )
            self._rtsp_deadline_ready.set()

    def rtsp_deadline_reached(self, now_ns: int | None = None) -> bool:
        if self.rtsp_deadline_ns is None:
            return False
        return (time.monotonic_ns() if now_ns is None else now_ns) >= self.rtsp_deadline_ns

    def _watch_rtsp_deadline(self) -> None:
        while not self.stop_event.is_set() and not self._rtsp_deadline_ready.wait(0.1):
            pass
        while not self.stop_event.is_set():
            deadline = self.rtsp_deadline_ns
            if deadline is None:
                return
            remaining_ns = deadline - time.monotonic_ns()
            if remaining_ns <= 0:
                self.rtsp_cap_triggered_ns = time.monotonic_ns()
                self.rtsp_cap_triggered_utc = datetime.now(timezone.utc).isoformat()
                capture_stop_seconds = getattr(self, "capture_stop_seconds", RTSP_CAPTURE_STOP_SECONDS)
                authorized_max_seconds = getattr(self, "authorized_max_seconds", AUTHORIZED_RTSP_MAX_SECONDS)
                self.rtsp_stop_reason = (
                    f"RTSP frame acquisition hard-stopped after {capture_stop_seconds:.0f}s; "
                    f"{authorized_max_seconds:.0f}s maximum includes teardown"
                )
                self.run_phase = "RTSP_HARD_CAP_STOPPING"
                self.stop_event.set()
                self.stop()
                return
            self.stop_event.wait(min(remaining_ns / 1e9, 0.1))

    def start(self) -> None:
        self.started_ns = time.monotonic_ns()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        os.chdir(ROOT)
        recorded_mode = getattr(self.args, "mode", "live") == "recorded"
        if self.startup_trace:
            self.startup_trace.record(
                "CONTROLLER_START", mode="recorded" if recorded_mode else "physical-rtsp",
                detector=self.args.detector,
                python_pid=os.getpid(),
            )
        if recorded_mode:
            source_records = getattr(self.args, "recorded_source_records", None)
            if not isinstance(source_records, dict) or set(source_records) != set(CAMERAS):
                raise RuntimeError("recorded input preflight did not provide both original source records")
            self.recorded_source_records = source_records
            by_id = {camera: SimpleNamespace(camera_id=camera) for camera in CAMERAS}
        else:
            from services.camera_v11.source_ownership import SourceOwnership
            from services.ml_service.app.config import load_settings
            self.settings = load_settings()
            by_id = {camera.camera_id: camera for camera in self.settings.cameras}
            missing = [camera for camera in CAMERAS if camera not in by_id]
            if missing: raise RuntimeError("authorized camera config lacks required sources: " + ",".join(missing))
        if getattr(self.args, "automatic_reid_shadow", False):
            # Load and hash-check the model and saved control reference before
            # claiming camera ownership or opening either RTSP source.
            from services.mv3dt_room.reid_embedder import OsnetEmbedder
            from services.mv3dt_staging.automatic_provisional_global_id import EXPECTED_MODEL_SHA256
            from services.mv3dt_staging.provisional_reid_shadow import (
                ProvisionalReIdShadow,
                load_preserved_control_reference,
            )
            checkpoint = Path(
                "/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/"
                "deepstream-tracker-3d-multi-view/experiments/deepstream/"
                "dev-room-pn263-global-identity-20260921/models/reid/"
                "osnet_ain_x1_0_msmt17.pth"
            )
            if self.startup_trace:
                self.startup_trace.record("OSNET_PREWARM_BEGIN", requested_device=str(self.args.shadow_reid_device))
            control_max, control_source = load_preserved_control_reference()
            if not checkpoint.is_file():
                raise FileNotFoundError("accepted OSNet-AIN checkpoint is unavailable")
            cfg = {
                "model_name": "osnet_ain_x1_0",
                "model_path": str(checkpoint).lstrip("/"),
                "model_sha256": EXPECTED_MODEL_SHA256,
                "download_if_missing": False,
                "input_height": 256,
                "input_width": 128,
                "device": str(self.args.shadow_reid_device),
                "cpu_threads": 2,
            }
            embedder = OsnetEmbedder(cfg, Path("/"))
            # Explicit bounded pre-load while camera connections are still
            # closed. GPU usage remains covered by the existing device guard.
            import torch
            torch.set_num_threads(2)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            embedder._load()
            prewarm_started = time.perf_counter()
            warmup = embedder.embed_batch([
                np.zeros((256, 128, 3), dtype=np.uint8)
            ])
            if warmup.shape != (1, 512) or not np.isfinite(warmup).all():
                raise RuntimeError("OSNet warmup did not produce one finite 512-D embedding")
            self.identity_shadow_prewarm = {
                "device": str(self.args.shadow_reid_device),
                "model_sha256": EXPECTED_MODEL_SHA256,
                "load_sec": float(embedder.metrics().get("loaded_for_sec", 0.0)),
                "first_forward_ms": float(embedder.metrics().get("last_batch_ms", 0.0)),
                "total_pre_camera_sec": time.perf_counter() - prewarm_started,
                "warmup_input": "synthetic local-only 256x128 BGR patch; no camera frame or identity retained",
            }
            if self.startup_trace:
                self.startup_trace.record("OSNET_PREWARM_COMPLETE", **self.identity_shadow_prewarm)
            self.identity_shadow = ProvisionalReIdShadow(
                embedder,
                control_reference_max=control_max,
                control_reference_source=control_source,
                person_class_id=self.person_class_id,
                sample_every_frames=int(self.args.shadow_reid_sample_every),
                max_crops_per_track=3,
                queue_size=32,
                max_gallery_tracks=128,
                control_history_frames=300,
                batch_size=8,
            )
        if not recorded_mode:
            from services.camera_v11.source_ownership import SourceOwnership
            lock_dir = Path(os.getenv("CAMERA_OWNER_LOCK_DIR", ".runtime/camera-owner-locks"))
            if not lock_dir.is_absolute(): lock_dir = ROOT / lock_dir
            self.ownership = SourceOwnership(CAMERAS, lock_dir)
            self.ownership.__enter__()  # refuse to connect if an existing cooperative owner is active
        self.pipe = DeepStreamProcess(
            mux_only=False, output_root=self.output, detector=self.args.detector,
            yolo_artifacts=Path(self.args.yolo_artifacts).expanduser()
            if self.args.detector == "yolo26m" else None,
            startup_trace=self.startup_trace,
        )
        self.pipe.wait_ready(timeout_sec=30)
        if self.startup_trace:
            self.startup_trace.record("CONTROLLER_RECEIVED_READY", ready_monotonic_ns=time.monotonic_ns())
        if getattr(self.args, "save_visual_evidence", False):
            if not self.args.native_frame:
                raise RuntimeError("visual evidence capture requires native-frame mode")
            from services.mv3dt_staging.live_tracking.visual_evidence import FrameEvidenceCapture
            self.evidence_capture = FrameEvidenceCapture(
                self.output / "visual-evidence",
                sample_every_frames=self.args.evidence_sample_every,
                max_samples_per_camera=self.args.evidence_max_samples_per_camera,
                queue_size=16,
                person_class_id=self.person_class_id,
            )
        if self.args.native_frame:
            from services.mv3dt_staging.live_tracking.native_frame_bridge import NativeFullFovScale
            transforms = {camera_id: NativeFullFovScale(camera_id) for camera_id in CAMERAS}
        else:
            bridge_mod = import_bridge(Path(self.args.bridge).expanduser())
            transforms = {camera_id: bridge_mod.RectificationBridge(camera_id)
                          for camera_id in CAMERAS}
        if recorded_mode:
            self.capture_workers = [
                RecordedCaptureWorker(
                    self, sid, self.recorded_source_records[camera_id], transforms[camera_id],
                    start_frame=int(getattr(self.args, f"start_frame_{camera_id.lower().replace('-', '')}")),
                    duration_sec=float(self.args.duration_sec),
                    playback_speed=float(getattr(self.args, "playback_speed", 1.0)),
                )
                for sid, camera_id in SOURCE_BY_ID.items()
            ]
        else:
            self.capture_workers = [
                CaptureWorker(self, sid, by_id[camera_id], self.settings,
                              transforms[camera_id])
                for sid, camera_id in SOURCE_BY_ID.items()
            ]
        self.writer_thread = threading.Thread(target=self._writer_loop, name="appsrc-writer", daemon=True)
        self.writer_thread.start()
        if not recorded_mode:
            self._rtsp_watchdog_thread = threading.Thread(
                target=self._watch_rtsp_deadline, name="entrance-rtsp-hard-cap", daemon=True
            )
            self._rtsp_watchdog_thread.start()
        if self.startup_trace:
            self.startup_trace.record(
                "RECORDED_CAPTURE_WORKERS_STARTING",
                sources=[{"camera": worker.camera.camera_id,
                          "source_id": worker.source_id,
                          "start_frame": worker.start_frame,
                          "duration_sec": worker.duration_sec}
                         for worker in self.capture_workers
                         if isinstance(worker, RecordedCaptureWorker)],
                worker_count=len(self.capture_workers),
            )
        for worker in self.capture_workers: worker.start()
        self._gpu_thread = threading.Thread(target=self._sample_gpu, name="device-gpu-sampler", daemon=True)
        self._gpu_thread.start()
        self._cpu_thread = threading.Thread(target=self._sample_cpu, name="live-cpu-sampler", daemon=True)
        self._cpu_thread.start()
        self.run_phase = "RECORDED_STARTING" if recorded_mode else "LIVE_CONNECTING"

    def _sample_gpu(self) -> None:
        """Sample device-wide utilization only; this is not process attribution."""
        while not self._gpu_stop.is_set():
            sample = _gpu_sample()
            self.gpu_samples.append(sample)
            if len(self.gpu_samples) > 100:
                del self.gpu_samples[:-100]
            self._gpu_stop.wait(2.0)

    def _sample_cpu(self) -> None:
        """Sample host CPU load, bridge-process CPU use, and bridge RSS via procfs."""
        previous_system = _system_cpu_ticks()
        previous_process = time.process_time()
        previous_wall = time.monotonic()
        while not self._cpu_stop.wait(1.0):
            now_wall = time.monotonic()
            now_process = time.process_time()
            current_system = _system_cpu_ticks()
            system_busy = None
            if previous_system and current_system:
                total_delta = current_system[0] - previous_system[0]
                idle_delta = current_system[1] - previous_system[1]
                if total_delta > 0:
                    system_busy = max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))
            wall_delta = now_wall - previous_wall
            process_percent = (100.0 * (now_process - previous_process) / wall_delta
                               if wall_delta > 0 else None)
            rss_mib = _process_rss_mib()
            self.cpu_samples.append({
                "sample_utc": datetime.now(timezone.utc).isoformat(),
                "system_cpu_busy_percent": round(system_busy, 3) if system_busy is not None else None,
                "bridge_process_cpu_percent_of_one_core": round(process_percent, 3)
                    if process_percent is not None else None,
                "bridge_process_rss_mib": round(rss_mib, 3) if rss_mib is not None else None,
            })
            if len(self.cpu_samples) > 300:
                del self.cpu_samples[:-300]
            previous_system = current_system
            previous_process = now_process
            previous_wall = now_wall

    def queue_frame(self, source_id: int, frame_index: int, image: np.ndarray,
                    source_pts_ns: int, source_dts_ns: int, arrival_ns: int,
                    display_image: np.ndarray | None = None) -> None:
        view_image = image if display_image is None else display_image
        with self.frame_lock:
            cache = self.frame_cache[source_id]
            cache[frame_index] = (view_image, source_pts_ns, source_dts_ns, arrival_ns)
            while len(cache) > 12: cache.popitem(last=False)
            if self.identity_shadow is not None:
                shadow_cache = self.shadow_inference_cache[source_id]
                shadow_cache[frame_index] = image
                while len(shadow_cache) > SHADOW_INFERENCE_CACHE_FRAMES:
                    shadow_cache.popitem(last=False)
            self.latest_frame[source_id] = frame_index
            self.stats[source_id]["frames_captured"] += 1
            self.stats[source_id]["capture_times"].append(arrival_ns / 1e9)
        q = self.frame_queues[source_id]
        try: q.put_nowait((frame_index, image, source_pts_ns, source_dts_ns, arrival_ns))
        except queue.Full:
            try: q.get_nowait()
            except queue.Empty: pass
            self.stats[source_id]["queue_drops"] += 1
            q.put_nowait((frame_index, image, source_pts_ns, source_dts_ns, arrival_ns))

    def record_recorded_frame(self, source_id: int, source: dict[str, Any], frame,
                              arrival_mono_ns: int, playback_deadline_mono_ns: int,
                              playback_speed: float = 1.0) -> None:
        """Persist the exact source frame/PTS to scheduled-playback crosswalk."""
        camera = SOURCE_BY_ID.get(int(source_id))
        if camera is None or source.get("camera_id") != camera:
            raise ValueError("recorded camera/source ID mapping mismatch")
        self.recorded_frame_records.append({
            "source_id": int(source_id),
            "camera_id": camera,
            "source_file_sha256": source["sha256"],
            "source_frame_index": int(frame.frame_index),
            "source_pts_ns": int(frame.source_pts_ns),
            "source_dts_ns": int(frame.source_dts_ns),
            "playback_speed_x": float(playback_speed),
            "appsrc_playback_deadline_monotonic_ns": int(playback_deadline_mono_ns),
            "host_arrival_monotonic_ns": int(arrival_mono_ns),
            "mapping_note": "original source frame ordinal and MP4 timestamps; DeepStream buf_pts is separately generated in appsrc running-time",
        })
        if camera not in self.recorded_playback_origins:
            self.recorded_playback_origins[camera] = {
                "source_first_pts_ns": int(frame.source_pts_ns),
                "playback_origin_monotonic_ns": int(playback_deadline_mono_ns),
            }

    def _writer_loop(self) -> None:
        assert self.pipe is not None
        source_ids = (0, 1)
        cursor = 0
        while not self.stop_event.is_set():
            source_id = source_ids[cursor % 2]
            cursor += 1
            try: item = self.frame_queues[source_id].get(timeout=0.02)
            except queue.Empty: continue
            frame_index, image, pts, dts, arrival = item
            try:
                self.pipe.send_frame(source_id, frame_index, image, pts, dts, arrival)
                self.stats[source_id]["frames_sent"] += 1
                if self.startup_trace and self.stats[source_id]["frames_sent"] == 1:
                    self.startup_trace.record(
                        "HOST_FIRST_APPSRC_FRAME_WRITE", source_id=source_id,
                        camera=SOURCE_BY_ID[source_id], source_frame_index=frame_index,
                        source_pts_ns=pts, source_dts_ns=dts,
                    )
            except Exception as exc:
                self.error = redact(f"DeepStream appsrc writer: {exc}")
                self.error_queue.put(self.error)
                self.stop_event.set()
                break

    def process_events(self) -> list[dict[str, Any]]:
        assert self.pipe is not None
        records = []
        while True:
            try: message = self.error_queue.get_nowait()
            except queue.Empty: break
            if not self.error: self.error = redact(message)
        while True:
            try: record = self.pipe.events.get_nowait()
            except queue.Empty: break
            records.append(record)
            self.event_records.append(record)
            kind = record.get("type")
            if kind == "ERROR":
                self.error = redact(str(record.get("message") or record.get("raw") or "DeepStream error"))
                self.stop_event.set()
                continue
            if kind not in {"MUX", "PGIE", "TRACK"}: continue
            sid = int(record.get("source_id", -1))
            if sid not in self.stats:
                self.error = f"DeepStream emitted unexpected source_id {sid}"
                self.stop_event.set()
                continue
            stat = self.stats[sid]
            if record.get("input_frame_index") is None:
                stat["source_id_mismatch"] += 1
                continue
            input_frame_index = int(record["input_frame_index"])
            previous_index = stat["last_input_frame_by_stage"][kind]
            if input_frame_index <= previous_index:
                stat["input_frame_nonmonotonic"] += 1
            stat["last_input_frame_by_stage"][kind] = input_frame_index
            if kind == "MUX":
                stat["mux_frames"] += 1
                frame_num = int(record.get("frame_num", -1))
                if frame_num <= stat["last_mux_frame_num"]:
                    stat["mux_frame_num_nonmonotonic"] += 1
                stat["last_mux_frame_num"] = frame_num
            elif kind == "PGIE":
                stat["pgie_frames"] += 1
                objects = record.get("objects", [])
                stat["pgie_person_detections"] += sum(
                    int(obj.get("class_id", -1)) == self.person_class_id for obj in objects
                )
                if record.get("pgie_pad_latency_ns") is not None:
                    stat["pgie_pad_latency_ns"].append(int(record["pgie_pad_latency_ns"]))
                if record.get("host_input_to_stage_ns") is not None:
                    stat["host_input_to_pgie_ns"].append(int(record["host_input_to_stage_ns"]))
            elif kind == "TRACK":
                tracker_event_started_ns = time.perf_counter_ns()
                stat["tracker_frames"] += 1
                objects = [obj for obj in record.get("objects", [])
                           if int(obj.get("class_id", -1)) == self.person_class_id]
                stat["tracker_person_observations"] += len(objects)
                if record.get("tracker_pad_latency_ns") is not None:
                    stat["tracker_pad_latency_ns"].append(int(record["tracker_pad_latency_ns"]))
                if record.get("host_input_to_stage_ns") is not None:
                    stat["host_input_to_tracker_ns"].append(int(record["host_input_to_stage_ns"]))
                for obj in objects:
                    track_id = obj.get("tracker_id")
                    if track_id is not None: stat["local_tracker_ids"].add(int(track_id))
                index = input_frame_index
                evidence_frame = None
                evidence_timing = None
                shadow_inference_frame = None
                shadow_lookup_started_ns = time.perf_counter_ns()
                with self.frame_lock:
                    exact = self.frame_cache[sid].get(index)
                    if exact is not None:
                        stat["display_frame_matched"] += 1
                        latest = self.latest_frame[sid]
                        if latest is not None and latest >= index:
                            stat["display_frame_age_frames"].append(latest - index)
                        self.latest_meta[sid] = record
                        stat["last_meta_mono_ns"] = time.monotonic_ns()
                        stat["fps_window"].append(time.monotonic())
                        evidence_frame = exact[0]
                        evidence_timing = (exact[1], exact[2], exact[3])
                        if self.identity_shadow is not None:
                            shadow_inference_frame = self.shadow_inference_cache[sid].get(index)
                    else:
                        stat["display_frame_missing"] += 1
                if self.identity_shadow is not None:
                    stat["shadow_frame_lookup_ns"].append(
                        time.perf_counter_ns() - shadow_lookup_started_ns
                    )
                    if shadow_inference_frame is None:
                        stat["shadow_cache_misses"] += 1
                    else:
                        stat["shadow_cache_hits"] += 1
                if self.evidence_capture is not None:
                    tracker_ids = tuple(sorted(
                        int(obj["tracker_id"]) for obj in objects
                        if obj.get("tracker_id") is not None
                    ))
                    force_evidence = tracker_ids != self._evidence_last_tracker_ids[sid]
                    self._evidence_last_tracker_ids[sid] = tracker_ids
                    if evidence_frame is None:
                        if force_evidence or index % self.args.evidence_sample_every == 0:
                            stat["evidence_frame_missing"] = stat.get("evidence_frame_missing", 0) + 1
                    else:
                        try:
                            self.evidence_capture.submit(
                                sid, index, evidence_frame, record,
                                cached_source_pts_ns=evidence_timing[0],
                                cached_source_dts_ns=evidence_timing[1],
                                cached_arrival_mono_ns=evidence_timing[2],
                                force=force_evidence,
                            )
                        except Exception as exc:
                            stat["evidence_crosswalk_failures"] = stat.get("evidence_crosswalk_failures", 0) + 1
                            self.error = redact(f"visual evidence crosswalk failed: {type(exc).__name__}: {exc}")
                            self.error_queue.put(self.error)
                            self.stop_event.set()
                if self.identity_shadow is not None:
                    if evidence_frame is None or shadow_inference_frame is None:
                        self.identity_shadow.record_join_failure(
                            RuntimeError(
                                "exact display frame expired before OSNet raster lookup"
                                if evidence_frame is None else
                                "exact inference raster expired before OSNet crop extraction"
                            )
                        )
                    else:
                        try:
                            shadow_submit_started_ns = time.perf_counter_ns()
                            self.identity_shadow.submit_frame(
                                source_id=sid,
                                camera_id=SOURCE_BY_ID[sid],
                                frame_index=index,
                                source_pts_ns=int(record["source_pts_ns"]),
                                cached_source_id=sid,
                                cached_frame_index=index,
                                cached_source_pts_ns=int(evidence_timing[0]),
                                inference_frame=shadow_inference_frame,
                                objects=objects,
                            )
                            stat["shadow_submit_frame_ns"].append(
                                time.perf_counter_ns() - shadow_submit_started_ns
                            )
                        except Exception as exc:
                            # OSNet is an optional shadow sidecar; a Re-ID
                            # crop failure must never stop the detector or
                            # camera-local NvDCF pipeline.
                            self.identity_shadow.record_join_failure(exc)
                            stat["shadow_submit_frame_ns"].append(
                                time.perf_counter_ns() - shadow_submit_started_ns
                            )
                stat["tracker_event_handling_ns"].append(
                    time.perf_counter_ns() - tracker_event_started_ns
                )
            self._processed_events += 1
        if self.pipe.process.poll() is not None and not self.stop_event.is_set() and not self.error:
            self.error = f"DeepStream container exited unexpectedly with status {self.pipe.process.returncode}"
            self.stop_event.set()
        return records

    def frame_payload(self, sid: int):
        payload = self.exact_frame_payload(sid)
        if payload is None:
            return None
        frame, meta, arrival, _pts, _index = payload
        return frame, meta, arrival

    def exact_frame_payload(self, sid: int):
        """Return only the exact frame that produced the latest TRACK record."""
        with self.frame_lock:
            meta = self.latest_meta[sid]
            if meta is None: return None
            frame_index = int(meta["input_frame_index"])
            cached = self.frame_cache[sid].get(frame_index)
            if cached is None: return None
            if int(meta.get("source_id", -1)) != sid:
                raise ValueError("tracker source ID differs from exact frame-cache source")
            source_pts = meta.get("source_pts_ns")
            if source_pts is None or int(source_pts) != int(cached[1]):
                raise ValueError("tracker source PTS differs from exact frame-cache PTS")
            # A result older than the capture arrival is still paired only with
            # its own cached raster. It is never painted on a newer video frame.
            return cached[0], meta, cached[3], int(cached[1]), frame_index

    def source_state(self, sid: int) -> str:
        if self.error: return "ERROR"
        worker = self.capture_workers[sid] if len(self.capture_workers) > sid else None
        return worker.state if worker else self.run_phase

    def cam_fps(self, sid: int) -> float:
        times = list(self.stats[sid]["fps_window"])
        if len(times) < 2 or times[-1] <= times[0]: return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def source_fps(self, sid: int) -> float:
        times = list(self.stats[sid]["capture_times"])
        if len(times) < 2 or times[-1] <= times[0]: return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def record_paint_latency(self, value_ms: float) -> None:
        self.paint_latency_ms.append(float(value_ms))

    def _release_ownership_if_workers_stopped(self) -> bool:
        """Release RTSP ownership only after every capture thread has terminated."""
        with self._ownership_release_lock:
            if self.ownership is None:
                return True
            if any(worker.is_alive() for worker in self.capture_workers):
                return False
            self.ownership.__exit__(None, None, None)
            self.ownership = None
            self.ownership_released_ns = time.monotonic_ns()
            self.ownership_released_utc = datetime.now(timezone.utc).isoformat()
            return True

    def _reap_ownership_after_workers_stop(self) -> None:
        """Keep camera locks while a timed-out capture close is alive, then release them."""
        while any(worker.is_alive() for worker in self.capture_workers):
            for worker in self.capture_workers:
                if worker.is_alive():
                    worker.join(timeout=0.1)
        self._release_ownership_if_workers_stopped()

    def _start_ownership_reaper(self) -> None:
        if self._ownership_reaper is not None and self._ownership_reaper.is_alive():
            return
        self._ownership_reaper = threading.Thread(
            target=self._reap_ownership_after_workers_stop,
            name="entrance-camera-lock-reaper", daemon=True,
        )
        self._ownership_reaper.start()

    def stop(self) -> None:
        if not self._stop_lock.acquire(blocking=False):
            return
        try:
            if self._stopped:
                if not self._release_ownership_if_workers_stopped():
                    self._start_ownership_reaper()
                return
            self.stop_requested_ns = time.monotonic_ns()
            self.stop_requested_utc = datetime.now(timezone.utc).isoformat()
            if self.startup_trace:
                self.startup_trace.record(
                    "CONTROLLER_STOP_REQUESTED", phase=self.run_phase,
                    reason=self.error or self.rtsp_stop_reason or "normal shutdown",
                    capture_worker_states=[{
                        "name": worker.name, "alive": worker.is_alive(),
                        "state": worker.state,
                    } for worker in self.capture_workers],
                    deepstream_pid=(self.pipe.process.pid if self.pipe else None),
                )
            self.stop_event.set()
            for worker in self.capture_workers:
                worker.join(timeout=8)
            if self.writer_thread is not None:
                self.writer_thread.join(timeout=3)
            self._gpu_stop.set()
            if self._gpu_thread is not None:
                self._gpu_thread.join(timeout=2)
            self._cpu_stop.set()
            if self._cpu_thread is not None:
                self._cpu_thread.join(timeout=2)
            if self.pipe is not None:
                self.pipe.end_input()
                self.pipe.stop(timeout_sec=10)
                if self.pipe.process.returncode not in (0, None) and not self.error:
                    self.error = f"DeepStream shutdown was not clean (exit status {self.pipe.process.returncode})"
                # Drain records emitted immediately before EOS so the final
                # report and raw sidecar include the pipeline tail.
                self.process_events()
                if (getattr(self.args, "mode", "live") == "recorded"
                        and not self.pipe.done_message_seen.is_set() and not self.error):
                    self.error = "recorded DeepStream shutdown lacked an observed EOS/DONE acknowledgement"
            if self.identity_shadow is not None:
                try:
                    self.identity_shadow_summary = self.identity_shadow.close(timeout_sec=8.0)
                except Exception as exc:
                    self.error = self.error or redact(
                        f"OSNet shadow shutdown failed: {type(exc).__name__}: {exc}"
                    )
                    self.identity_shadow_summary = self.identity_shadow.snapshot()
            if self.evidence_capture is not None:
                try:
                    self.evidence_summary = self.evidence_capture.close(timeout_sec=10.0)
                except Exception as exc:
                    self.error = self.error or redact(
                        f"visual evidence shutdown failed: {type(exc).__name__}: {exc}"
                    )
            workers_alive = [worker.name for worker in self.capture_workers if worker.is_alive()]
            if self.rtsp_first_attempt_ns is not None and self.capture_workers:
                close_times = [worker.close_return_ns for worker in self.capture_workers if worker.close_return_ns]
                if len(close_times) == len(self.capture_workers) and not all(
                    rtsp_close_within_authorized_cap(self.rtsp_first_attempt_ns, value,
                                                     self.authorized_max_seconds)
                    for value in close_times
                ):
                    self.error = self.error or "RTSP teardown exceeded the authorized session cap"
            if self.ownership is not None and not self._release_ownership_if_workers_stopped():
                self.error = self.error or ("capture cleanup timeout; source locks retained until workers terminate: "
                                            + ",".join(workers_alive))
                self._start_ownership_reaper()
            with self.events_path.open("w", encoding="utf-8") as event_file:
                os.chmod(self.events_path, 0o600)
                for record in self.event_records:
                    event_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.run_phase = ("STOPPED_BY_RTSP_HARD_CAP" if self.rtsp_stop_reason and not self.error
                              else "STOPPED" if not self.error else "ERROR")
            self.stopped_ns = time.monotonic_ns()
            self._stopped = True
            if self.startup_trace:
                self.startup_trace.record(
                    "CONTROLLER_STOP_COMPLETED", phase=self.run_phase,
                    error=self.error or None,
                    stopped_monotonic_ns=self.stopped_ns,
                    deepstream_exit_code=(self.pipe.process.poll() if self.pipe else None),
                    deepstream_done_message=(self.pipe.done_message_seen.is_set() if self.pipe else None),
                    capture_workers=[{
                        "name": worker.name, "alive": worker.is_alive(),
                        "state": worker.state,
                        "capture_close_succeeded": getattr(worker, "capture_close_succeeded", None),
                    } for worker in self.capture_workers],
                    writer_thread_alive=(self.writer_thread.is_alive() if self.writer_thread else None),
                    gpu_sampler_alive=(self._gpu_thread.is_alive() if self._gpu_thread else None),
                    cpu_sampler_alive=(self._cpu_thread.is_alive() if self._cpu_thread else None),
                )
        finally:
            self._stop_lock.release()


def preflight_recorded_sources(args) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Verify original file hashes, stream metadata, and GStreamer PTS at each start frame."""
    from services.mv3dt_staging.live_tracking.recorded_input import (
        GstRecordedFileCapture,
        inspect_recording,
    )

    paths = {"CAM-02": Path(args.cam02_video), "CAM-05": Path(args.cam05_video)}
    starts = {"CAM-02": int(args.start_frame_cam02), "CAM-05": int(args.start_frame_cam05)}
    sources: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for camera in CAMERAS:
        source = inspect_recording(camera, paths[camera])
        start = starts[camera]
        if start < 0 or start + math.ceil(float(args.duration_sec) * source["nominal_fps"]) >= source["frame_count"]:
            raise ValueError(f"{camera}: requested recorded segment is outside the verified original file")
        reader = GstRecordedFileCapture(source)
        target = None
        selected_pts = selected_dts = None
        try:
            for expected_index in range(start + 1):
                target = reader.pull(timeout_ms=5000)
                if target is None or target.frame_index != expected_index:
                    raise RuntimeError(
                        f"{camera}: GStreamer source-frame {expected_index} decode preflight failed"
                    )
                expected_pts = int(source["pts_ns_by_decoded_frame"][expected_index])
                if target.source_pts_ns != expected_pts:
                    raise RuntimeError(
                        f"{camera}: GStreamer source PTS disagrees with the recorded packet crosswalk "
                        f"at decoded frame {expected_index}"
                    )
                if (target.bgr.shape != (source["height"], source["width"], 3)
                        or target.bgr.dtype != np.uint8):
                    raise RuntimeError(f"{camera}: selected source-frame geometry/pixel format is invalid")
                if expected_index == start:
                    selected_pts = target.source_pts_ns
                    selected_dts = target.source_dts_ns
                target = None
            close_result = reader.close()
            if close_result["close_state"] != "NULL":
                raise RuntimeError(f"{camera}: GStreamer preflight did not close cleanly")
        finally:
            if not reader.closed:
                reader.close()
        summaries[camera] = {
            key: value for key, value in source.items()
            if key not in {"pts_ns_by_decoded_frame", "dts_ns_by_decoded_frame"}
        }
        summaries[camera].update({
            "requested_start_frame": start,
            "gstreamer_verified_first_source_pts_ns": selected_pts,
            "gstreamer_verified_first_source_dts_ns": selected_dts,
            "requested_segment_pts_ns_from_packet_crosswalk": int(source["pts_ns_by_decoded_frame"][start]),
            "segment_frame_pts_crosscheck": "runtime GStreamer decoder must match the packet-PTS source-frame map exactly",
            "gstreamer_preflight_decoded_frames": reader.decoded_frames,
            "gstreamer_verified_start_frame_index": start,
            "gstreamer_verified_start_frame_pts_ns": selected_pts,
            "gstreamer_verified_start_frame_dts_ns": selected_dts,
            "gstreamer_preflight_close_state": reader.close_state,
            "gstreamer_eos_observed": reader.eos_seen,
        })
        sources[camera] = source
    return sources, summaries


def run_live(args) -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                                   QHBoxLayout, QWidget, QPushButton, QComboBox)
    if args.native_frame:
        from services.mv3dt_staging.live_tracking.native_frame_bridge import detector_box_to_source
    from services.mv3dt_staging.live_map_projection import (
        FrameJoinError,
        LiveNativeMapProjection,
        choose_marker_label_position,
    )
    from services.mv3dt_staging.map_fusion import (
        ExperimentalMapFusion, empty_identity_snapshot, paint_map_fusion_markers,
    )

    recorded_mode = getattr(args, "mode", "live") == "recorded"
    if args.output:
        output = Path(args.output).expanduser().resolve()
    elif recorded_mode:
        output = ROOT / ".runtime" / f"entrance-recorded-yolo-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    else:
        output = Path(tempfile.mkdtemp(prefix="entrance-live-track-"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty live evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    startup_trace = StartupTrace(output / "startup-diagnostics.jsonl")
    args.startup_trace = startup_trace
    startup_trace.record(
        "RUNNER_STARTED", python_pid=os.getpid(), mode=getattr(args, "mode", "live"),
        detector=args.detector, display=os.environ.get("DISPLAY"),
        qt_qpa_platform=os.environ.get("QT_QPA_PLATFORM"), output_dir=str(output),
        native_frame=bool(args.native_frame),
        camera_mapping={"source_id_0": "CAM-02", "source_id_1": "CAM-05"},
        recorded_duration_sec=(float(args.duration_sec) if recorded_mode else None),
        recorded_playback_speed_x=(float(getattr(args, "playback_speed", 1.0))
                                   if recorded_mode else None),
    )
    recorded_source_summaries = None
    if recorded_mode:
        startup_trace.record(
            "RECORDED_SOURCE_PREFLIGHT_BEGIN",
            sources={"CAM-02": str(Path(args.cam02_video).expanduser()),
                     "CAM-05": str(Path(args.cam05_video).expanduser())},
            start_frames={"CAM-02": args.start_frame_cam02,
                          "CAM-05": args.start_frame_cam05},
        )
        try:
            recorded_sources, recorded_source_summaries = preflight_recorded_sources(args)
            args.recorded_source_records = recorded_sources
            startup_trace.record(
                "RECORDED_SOURCE_PREFLIGHT_PASSED",
                sources={camera: {
                    "camera_id": record.get("camera_id"),
                    "source_sha256": record.get("sha256"),
                    "width": record.get("width"), "height": record.get("height"),
                    "nominal_fps": record.get("nominal_fps"),
                    "frame_count": record.get("frame_count"),
                    "verified_start_pts_ns": recorded_source_summaries[camera].get(
                        "gstreamer_verified_start_frame_pts_ns"),
                } for camera, record in recorded_sources.items()},
            )
        except Exception as exc:
            startup_trace.record(
                "RECORDED_SOURCE_PREFLIGHT_FAILED",
                error=redact(f"{type(exc).__name__}: {exc}"),
            )
            report = {
                "status": "RECORDED_SOURCE_PREFLIGHT_BLOCKED",
                "reason": redact(f"{type(exc).__name__}: {exc}"),
                "input_mode": "RECORDED_ORIGINAL_NATIVE_SCALE_ONLY",
                "physical_rtsp_opened": False,
                "camera_ownership_locks_claimed": False,
                "deepstream_started": False,
                "output_dir": str(output),
                "startup_diagnostics": str(startup_trace.path),
            }
            report_path = output / "live-run.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            os.chmod(report_path, 0o600)
            print(json.dumps(report, indent=2))
            return 5
    layout_path = Path(args.layout).expanduser() if args.layout else Path(
        "/tmp/amc-nvidia-skill-stage.pd7zJs/projects/project_20261018_080650_1744/manual_adjustment/layout.png"
    )
    map_projection = None
    map_candidate_record = None
    map_meta = None
    if args.experimental_native_alignment:
        startup_trace.record(
            "MAP_ALIGNMENT_PREFLIGHT_BEGIN",
            alignment_path=str(Path(args.experimental_native_alignment).expanduser()),
            layout_path=str(layout_path),
        )
        try:
            if not args.native_frame:
                raise ValueError("experimental native alignment requires --native-frame")
            from services.mv3dt_staging.native_map_preview import (
                file_sha256, load_validated_candidate,
            )
            alignment_path = Path(args.experimental_native_alignment).expanduser().resolve()
            alignment_doc = json.loads(alignment_path.read_text(encoding="utf-8"))
            source_paths = {
                key: Path(alignment_doc["source_files"][key]["path"]).expanduser().resolve()
                for key in ("cam02", "cam05")
            }
            matrices, map_diagnostics, map_meta = load_validated_candidate(
                alignment_path, video_paths=source_paths, layout_path=layout_path,
            )
            map_projection = LiveNativeMapProjection(
                matrices, layout_width=map_meta["width"], layout_height=map_meta["height"],
                person_class_id=detector_person_class(args.detector),
            )
            map_candidate_record = {
                "status": "LOADED_HASH_AND_COORDINATE_VERIFIED_EXPERIMENTAL",
                "alignment_path": str(alignment_path),
                "alignment_sha256": file_sha256(alignment_path),
                "layout_path": str(layout_path.resolve()),
                "layout_sha256": file_sha256(layout_path),
                "source_hashes": {key: file_sha256(path) for key, path in source_paths.items()},
                "camera_mapping": alignment_doc.get("camera_mapping"),
                "coordinate_space": alignment_doc.get("coordinate_space"),
                "camera_coordinate_dimensions": alignment_doc.get("camera_coordinate_dimensions"),
                "matrices_row_major": {key: matrix.tolist() for key, matrix in matrices.items()},
                "fit_and_heldout_diagnostics_map_pixels_only": map_diagnostics,
                "map_geometry_status": "EXPERIMENTAL / UNCALIBRATED",
                "metric_accuracy": "UNVERIFIED",
                "cross_camera_synchronization": "UNVERIFIED",
            }
            startup_trace.record(
                "MAP_ALIGNMENT_PREFLIGHT_PASSED",
                alignment_sha256=map_candidate_record["alignment_sha256"],
                layout_sha256=map_candidate_record["layout_sha256"],
                source_hashes=map_candidate_record["source_hashes"],
                coordinate_space=map_candidate_record["coordinate_space"],
            )
        except Exception as exc:
            startup_trace.record(
                "MAP_ALIGNMENT_PREFLIGHT_FAILED",
                error=redact(f"{type(exc).__name__}: {exc}"),
            )
            report = {
                "status": "MAP_PREFLIGHT_BLOCKED",
                "reason": redact(f"{type(exc).__name__}: {exc}"),
                "rtsp_connections_opened": False,
                "deepstream_started": False,
                "map_candidate_path": str(args.experimental_native_alignment),
                "output_dir": str(output),
                "startup_diagnostics": str(startup_trace.path),
            }
            report_path = output / "live-run.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            os.chmod(report_path, 0o600)
            print(json.dumps(report, indent=2))
            return 4
    yolo_asset_hashes = None
    frozen_pgie_config_sha256 = None
    if args.detector == "yolo26m":
        from services.mv3dt_staging.live_tracking.yolo_candidate import (
            FROZEN_PGIE, sha256_file, validate_yolo_assets,
        )
        yolo_asset_hashes = validate_yolo_assets(Path(args.yolo_artifacts).expanduser())
        frozen_pgie_config_sha256 = sha256_file(FROZEN_PGIE)
        startup_trace.record(
            "DETECTOR_ASSET_VALIDATION_PASSED", detector=args.detector,
            artifact_sha256=yolo_asset_hashes,
            frozen_pgie_config_sha256=frozen_pgie_config_sha256,
        )
    gpu_baseline = collect_gpu_baseline(samples=3, interval_sec=1.0)
    startup_trace.record(
        "GPU_PREFLIGHT_COMPLETE", samples=gpu_baseline,
        safe_for_smoke=safe_for_smoke(gpu_baseline),
        host_meminfo=read_host_memory_summary(),
    )
    if not safe_for_smoke(gpu_baseline):
        report = {
            "status": "GPU_PREFLIGHT_BLOCKED",
            "detector": args.detector,
            "gpu_baseline": gpu_baseline,
            "rtsp_connections_opened": False,
            "deepstream_started": False,
            "reason": "three fresh device-total GPU samples were not all available and below 95%",
            "yolo_asset_sha256": yolo_asset_hashes,
            "frozen_yolo_pgie_config_sha256": frozen_pgie_config_sha256,
            "output_dir": str(output),
            "startup_diagnostics": str(startup_trace.path),
            "map_candidate": map_candidate_record,
        }
        report_path = output / "live-run.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        os.chmod(report_path, 0o600)
        print(json.dumps(report, indent=2))
        return 3
    startup_trace.record("QAPPLICATION_CREATE_BEGIN", display=os.environ.get("DISPLAY"))
    app = QApplication(sys.argv[:1])
    startup_trace.record("QAPPLICATION_CREATED", application_name=app.applicationName())

    class VideoPane(QWidget):
        def __init__(self, camera: str, person_class_id: int, latency_callback,
                     native_frame: bool = False, recorded_mode: bool = False):
            super().__init__()
            self.camera = camera
            self.person_class_id = person_class_id
            self.latency_callback = latency_callback
            self.native_frame = native_frame
            self.recorded_mode = bool(recorded_mode)
            self.image = QImage()
            self.objects = []
            self.info = "Waiting for an inference-matched frame"
            self.arrival_ns = 0
            self.frame_index = -1
            self.frame_key = None
            self.latency_frame_index = -1
            self._shown_index = -1
            self.setMinimumSize(440, 380)

        def set_payload(self, frame, meta, arrival_ns, *,
                        map_out_of_bounds_ids=(), map_jump_ids=(),
                        shadow_global_ids=None):
            h, w = frame.shape[:2]
            qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
            self.image = qimg.copy()
            raw_objects = [obj for obj in meta.get("objects", [])
                           if int(obj.get("class_id", -1)) == self.person_class_id]
            # DeepStream boxes are in the full-frame 1920x1080 scale-only
            # inference space. Convert copies to native pixels for the raw
            # source-resolution UI; raw event records remain untouched.
            self.objects = ([detector_box_to_source(obj, self.camera) for obj in raw_objects]
                            if self.native_frame else raw_objects)
            self.shadow_global_ids = shadow_global_ids or {}
            self.map_out_of_bounds_ids = {str(value) for value in map_out_of_bounds_ids}
            self.map_jump_ids = {str(value) for value in map_jump_ids}
            self.arrival_ns = int(arrival_ns)
            pts = meta.get("source_pts_ns")
            pts_text = "UNAVAILABLE" if pts in (None, -1) else f"{int(pts) / 1e9:.3f}s"
            frame_idx = meta.get("input_frame_index", "?")
            self.frame_index = int(frame_idx) if frame_idx != "?" else -1
            try:
                self.frame_key = (
                    int(meta["source_id"]), int(meta["input_frame_index"]),
                    int(meta["source_pts_ns"]),
                )
            except (KeyError, TypeError, ValueError):
                self.frame_key = None
            gst_fn = meta.get("frame_num", "?")
            local = ", ".join(str(o.get("tracker_id")) for o in self.objects if o.get("tracker_id") is not None)
            geometry = f"native source view {w}x{h}" if self.native_frame else f"view {w}x{h}"
            self.info = (f"{geometry} · Source ID {meta.get('source_id')} · input frame {frame_idx} / mux frame {gst_fn} · "
                         f"source PTS {pts_text} / mux buf_pts {int(meta.get('buf_pts', 0))/1e9:.3f}s · "
                         f"person observations {len(self.objects)} · local NvDCF IDs {local or 'NO_PEOPLE'}")
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor("#101820"))
            margin, header = 8, 32
            p.setPen(QColor("#f3f7fa")); p.setFont(QFont("Sans Serif", 12, QFont.Weight.Bold))
            mode = source_panel_mode_label(
                native_frame=self.native_frame,
                recorded_mode=self.recorded_mode,
            )
            p.drawText(margin, 22, f"{self.camera} — {mode}")
            area = self.rect().adjusted(margin, header, -margin, -40)
            if not self.image.isNull():
                size = self.image.size(); scale = min(area.width()/size.width(), area.height()/size.height())
                width, height = size.width()*scale, size.height()*scale
                target = (area.x()+(area.width()-width)/2, area.y()+(area.height()-height)/2, width, height)
                p.drawImage(int(target[0]), int(target[1]), self.image.scaled(int(width), int(height),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                sx, sy = width/size.width(), height/size.height()
                overlay_font = QFont("Sans Serif", 9, QFont.Weight.Bold)
                overlay_metrics = QFontMetrics(overlay_font)
                p.setFont(overlay_font)
                p.setPen(QPen(QColor("#20f080"), 2))
                overlay_objects = []
                frame_rect = (
                    int(target[0]), int(target[1]), int(width), int(height),
                )
                for obj in self.objects:
                    box = clamp_box_to_frame(
                        frame_rect,
                        (
                            target[0] + float(obj.get("left", 0))*sx,
                            target[1] + float(obj.get("top", 0))*sy,
                            float(obj.get("width", 0))*sx,
                            float(obj.get("height", 0))*sy,
                        ),
                    )
                    if box is None:
                        continue
                    x, y, w, h, box_was_clamped = box
                    p.drawRect(int(x), int(y), int(w), int(h))
                    tid = obj.get("tracker_id")
                    anchor_x, anchor_y = x + w / 2, y + h
                    if tid is not None and str(tid) in self.map_out_of_bounds_ids:
                        p.setPen(QPen(QColor("#ff3434"), 3))
                        p.drawEllipse(int(anchor_x - 7), int(anchor_y - 7), 14, 14)
                    else:
                        p.setPen(QPen(QColor("#3cbfff" if self.camera == "CAM-02" else "#ffbe3c"), 2))
                        p.drawEllipse(int(anchor_x - 4), int(anchor_y - 4), 8, 8)
                    if tid is not None and str(tid) in self.map_jump_ids:
                        p.setPen(QPen(QColor("#ff3434"), 3))
                        p.drawEllipse(int(anchor_x - 11), int(anchor_y - 11), 22, 22)
                    overlay_objects.append({
                        "left": x, "top": y, "width": w, "height": h,
                        "tracker_id": tid,
                        "confidence": float(obj.get("confidence", 0.0)),
                        "box_was_clamped": box_was_clamped,
                    })
                paint_camera_track_labels(
                    p, frame_rect, overlay_objects, self.shadow_global_ids,
                    overlay_metrics, Qt.TextElideMode.ElideRight,
                    QColor(0, 0, 0, 190), QColor("#ffffff"),
                )
                paint_camera_provisional_badge(
                    p, frame_rect, self.shadow_global_ids, self.camera,
                    {str(obj.get("tracker_id")) for obj in self.objects
                     if obj.get("tracker_id") is not None},
                    overlay_metrics, Qt.TextElideMode.ElideRight,
                    QColor("#542578"), QColor("#ffffff"),
                )
            p.setPen(QColor("#d8e3eb")); p.setFont(QFont("Sans Serif", 9))
            p.drawText(self.rect().adjusted(8, self.height()-36, -8, -4), Qt.AlignmentFlag.AlignLeft, self.info)
            p.end()
            if self.arrival_ns and self.frame_index >= 0 and self.frame_index != self.latency_frame_index:
                latency_ms = max(0.0, (time.monotonic_ns() - self.arrival_ns) / 1e6)
                self.latency_frame_index = self.frame_index
                self.info += f" · decoded→Qt paint {latency_ms:.1f} ms"
                self.latency_callback(latency_ms)
                self.update()

    class MapPane(QWidget):
        def __init__(self, projection):
            super().__init__()
            self.pixmap = QPixmap(str(layout_path)) if layout_path.is_file() else QPixmap()
            self.projection = projection
            self.states = {camera: None for camera in CAMERAS}
            self.shadow_global_ids = {camera: {} for camera in CAMERAS}
            self.identity_snapshot = empty_identity_snapshot(time.monotonic_ns())
            self.map_mode = getattr(args, "map_mode", "local")
            self.fusion = (ExperimentalMapFusion(
                layout_width=map_meta["width"], layout_height=map_meta["height"])
                if map_meta is not None else None)
            self.last_fusion_snapshot = None
            self.last_rendered_markers = None
            self.fusion_diagnostics = deque(maxlen=512)
            self._last_fusion_signature = None
            self.setMinimumSize(420, 380)

        def set_state(self, camera_id, state, shadow_global_ids=None):
            self.states[camera_id] = state
            self.update()

        def set_identity_snapshot(self, snapshot):
            self.identity_snapshot = snapshot
            self.shadow_global_ids = snapshot["assignments"]
            self.update()

        def set_mode(self, mode):
            self.map_mode = mode
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self); p.fillRect(self.rect(), QColor("#151c22"))
            p.setPen(QColor("#ffffff")); p.setFont(QFont("Sans Serif", 12, QFont.Weight.Bold))
            p.drawText(8, 22, "Entrance Room · EXPERIMENTAL / UNCALIBRATED")
            area = self.rect().adjusted(8, 34, -8, -132)
            image_box = None
            if not self.pixmap.isNull():
                pix = self.pixmap.scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
                x = area.x()+(area.width()-pix.width())//2; y = area.y()+(area.height()-pix.height())//2
                p.setOpacity(0.85); p.drawPixmap(x, y, pix); p.setOpacity(1.0)
                if map_meta is not None:
                    image_box = (x, y, pix.width(), pix.height(),
                                 pix.width() / map_meta["width"],
                                 pix.height() / map_meta["height"])
            state_lines = []
            current_states = {}
            for camera in CAMERAS:
                state = (self.projection.snapshot(camera) if self.projection is not None
                         else {"status": "DISABLED", "markers": []})
                status = state.get("status", "WAITING")
                current_states[camera] = state
                pts = state.get("source_pts_ns")
                frame = state.get("frame_index")
                age = state.get("age_ms")
                pts_text = "PTS —" if pts is None else f"PTS {int(pts) / 1e9:.3f}s"
                state_lines.append(
                    f"{camera}: {status} · frame {frame if frame is not None else '—'} · {pts_text}"
                    + (f" · age {age:.0f}ms" if age is not None else "")
                )
            labels_hidden, group_warnings = [], []
            if self.fusion is not None and image_box is not None:
                snapshot = self.fusion.update(current_states, self.identity_snapshot,
                                              mode=self.map_mode)
                self.last_fusion_snapshot = snapshot
                rendered = paint_map_fusion_markers(p, snapshot, image_box)
                self.last_rendered_markers = rendered
                labels_hidden = rendered["labels_hidden"]
                signature = (self.map_mode, snapshot["identity_revision"],
                    tuple((m["key"], tuple(tuple(k) for k in m["contributor_frame_keys"]))
                          for m in rendered["markers"]))
                if signature != self._last_fusion_signature:
                    self._last_fusion_signature = signature
                    self.fusion_diagnostics.append({
                        "snapshot_monotonic_ns": snapshot["snapshot_monotonic_ns"],
                        "mode": self.map_mode, "identity_revision": snapshot["identity_revision"],
                        "groups": snapshot["groups"], "rejected": snapshot["rejected"],
                        "rendered": rendered,
                    })
                fused = sum(len(m["contributors"]) > 1 for m in snapshot["markers"])
                state_lines.append(f"{self.map_mode} · {len(snapshot['markers'])} markers · {fused} fused groups")
                group_warnings = [f"{g['provisional_global_id']}? SPLIT: {','.join(g['reasons'])}"
                                  for g in snapshot["groups"] if g["decision"] == "SPLIT"]
                self.setToolTip("\n".join(
                    f"{m['label']}: " + "; ".join(
                        f"{o['camera_id']} local {o['local_tracker_id']} frame {o['frame_index']} "
                        f"PTS {o['source_pts_ns']} layout {o['layout_xy']}"
                        for o in m["contributors"])
                    for m in snapshot["markers"]))
            p.setBrush(Qt.BrushStyle.NoBrush)
            warning_parts = list(group_warnings)
            if labels_hidden:
                warning_parts.append("label space exhausted " + ",".join(labels_hidden))
            if self.projection is not None:
                for camera in CAMERAS:
                    state = self.states.get(camera) or {}
                    if state.get("out_of_bounds_ids"):
                        warning_parts.append(f"{camera} OOB {','.join(state['out_of_bounds_ids'])}")
                    if state.get("jump_candidate_ids"):
                        warning_parts.append(f"{camera} JUMP? {','.join(state['jump_candidate_ids'])}")
                    if state.get("duplicate_local_ids"):
                        warning_parts.append(f"{camera} DUPLICATE local obs")
                    if state.get("invalid_ids"):
                        warning_parts.append(f"{camera} INVALID {','.join(state['invalid_ids'])}")
            p.fillRect(self.rect().adjusted(8, self.height()-128, -8, -8), QColor("#713b12"))
            p.setPen(QColor("#fff0d1")); p.setFont(QFont("Sans Serif", 8, QFont.Weight.Bold))
            details = "\n".join(state_lines + [
                "Experimental pixel mean; host-arrival timing only; exposure sync UNVERIFIED.",
                "Map-pixel fit/held-out diagnostics only; metric accuracy UNVERIFIED.",
                "Warnings: " + ("; ".join(warning_parts) if warning_parts else "none in latest matched frames"),
            ])
            p.drawText(self.rect().adjusted(14, self.height()-124, -12, -10),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                       | Qt.TextFlag.TextWordWrap, details)

    class Window(QMainWindow):
        def __init__(self):
            super().__init__(); self.controller = None; self.run_started = 0.0
            self._first_preview_recorded = set()
            self.smoke_passed = False; self.review_started = 0.0; self.finish_started = 0.0
            self.review_deadline_ns = 0
            self.smoke_baseline = None
            self.shutdown_started = 0.0
            self.active_preview_screenshot_path = None
            self.global_fusion_screenshot_path = None
            self.global_fusion_visual_evidence = []
            self.provisional_identity_visual_evidence = []
            self.review_skipped_reason = ""
            mode = ("DEMO / RECORDED VIDEO — NOT LIVE" if recorded_mode else
                    "RAW NATIVE VIEW; no rectification" if args.native_frame else "rectified input")
            self.setWindowTitle(f"Entrance Room — {mode} · {args.detector} / local NvDCF")
            self.resize(1780, 920)
            central = QWidget(); layout = QVBoxLayout(central)
            mode_banner = (
                "DEMO / RECORDED VIDEO — NOT LIVE · original frames; source-PTS-paced scale-only inference"
                if recorded_mode else
                "RAW CAMERA VIEW; scale-only inference; EXPERIMENTAL map; camera-local IDs only"
                if map_projection is not None else
                "RAW CAMERA VIEW; full-FOV scale-only 1920x1080 inference; no map/global IDs"
                if args.native_frame else "no global IDs / map projection"
            )
            shadow_banner = (
                " · OSNet provisional shadow only; candidates are unverified and nonpersistent"
                if getattr(args, "automatic_reid_shadow", False) else ""
            )
            banner_prefix = "STARTING RECORDED DEMO" if recorded_mode else "STARTING ISOLATED LIVE"
            self.banner = QLabel(f"{banner_prefix} {args.detector.upper()} + NvDCF — {mode_banner}{shadow_banner}")
            self.banner.setStyleSheet("background:#25435b;color:white;padding:9px;font-weight:700;font-size:16px")
            latency_callback = lambda value: self.controller.record_paint_latency(value) if self.controller else None
            row = QHBoxLayout(); self.panes = {
                "CAM-02": VideoPane("CAM-02", detector_person_class(args.detector), latency_callback,
                                    native_frame=args.native_frame, recorded_mode=recorded_mode),
                "CAM-05": VideoPane("CAM-05", detector_person_class(args.detector), latency_callback,
                                    native_frame=args.native_frame, recorded_mode=recorded_mode)}
            self.map_pane = MapPane(map_projection)
            for camera in CAMERAS: row.addWidget(self.panes[camera], 2)
            row.addWidget(self.map_pane, 2 if map_projection is not None else 1)
            self.status = QLabel("Starting local DeepStream process…")
            self.status.setWordWrap(True)
            self.status.setStyleSheet("background:#202b33;color:#e3eef4;padding:8px")
            self.stop_button = QPushButton("Stop recorded benchmark" if recorded_mode else "Stop live session")
            self.stop_button.clicked.connect(self._stop_async)
            self.map_mode_selector = QComboBox()
            self.map_mode_selector.addItem("Local / Diagnostic", "local")
            self.map_mode_selector.addItem("Experimental Global Fusion", "experimental-global")
            self.map_mode_selector.setCurrentIndex(
                1 if getattr(args, "map_mode", "local") == "experimental-global" else 0)
            self.map_mode_selector.currentIndexChanged.connect(
                lambda _index: self.map_pane.set_mode(self.map_mode_selector.currentData()))
            self.map_mode_selector.setEnabled(map_projection is not None)
            controls = QHBoxLayout(); controls.addWidget(self.status, 1)
            controls.addWidget(self.map_mode_selector); controls.addWidget(self.stop_button)
            layout.addWidget(self.banner); layout.addLayout(row, 1); layout.addLayout(controls)
            self.setCentralWidget(central)
            self.timer = QTimer(self); self.timer.setInterval(50); self.timer.timeout.connect(self.refresh)
            self.timer.start()

        def refresh(self):
            if self.controller is None:
                if not self.run_started:
                    self.run_started = time.monotonic()
                    threading.Thread(target=self._start_controller, daemon=True).start()
                return
            records = self.controller.process_events()
            if self.controller.rtsp_stop_reason and not recorded_mode:
                self.banner.setText("RTSP HARD CAP REACHED — CLOSING CAMERA AND DEEPSTREAM PIPELINES")
                self.status.setText(self.controller.rtsp_stop_reason)
                self._stop_async()
                return
            gpu_stop_reason = live_gpu_guard_reason(self.controller.gpu_samples)
            if gpu_stop_reason:
                self.controller.error = gpu_stop_reason
                self._stop_async(); return
            identity_snapshot = (
                self.controller.identity_shadow.display_identity_snapshot()
                if self.controller.identity_shadow is not None else
                empty_identity_snapshot(time.monotonic_ns())
            )
            shadow_map = identity_snapshot["assignments"]
            self.map_pane.set_identity_snapshot(identity_snapshot)
            # A withdrawal must clear labels even if a camera has no new frame.
            for camera in CAMERAS:
                if getattr(self.panes[camera], "shadow_global_ids", {}) != shadow_map.get(camera, {}):
                    self.panes[camera].shadow_global_ids = shadow_map.get(camera, {})
                    self.panes[camera].update()
            for camera, sid in (("CAM-02", 0), ("CAM-05", 1)):
                try:
                    payload = self.controller.exact_frame_payload(sid)
                except Exception as exc:
                    self.controller.error = redact(f"exact display-frame join failed: {type(exc).__name__}: {exc}")
                    self.controller.stop_event.set()
                    self._stop_async()
                    return
                if payload:
                    frame, meta, arrival, cached_pts, cached_index = payload
                    current = getattr(self.panes[camera], "_shown_index", -1)
                    if cached_index > current:
                        map_state = None
                        if map_projection is not None:
                            try:
                                map_state = map_projection.update(
                                    camera, meta,
                                    cached_source_id=sid,
                                    cached_frame_index=cached_index,
                                    cached_source_pts_ns=cached_pts,
                                    # Age is measured from actual frame arrival,
                                    # not the time a delayed TRACK event reached Qt.
                                    now_mono_ns=arrival,
                                )
                            except FrameJoinError as exc:
                                self.controller.error = redact(f"map frame join rejected: {exc}")
                                self.controller.stop_event.set()
                                self._stop_async()
                                return
                            self.map_pane.set_state(camera, map_state, shadow_map.get(camera, {}))
                        self.panes[camera].set_payload(
                            frame, meta, arrival,
                            map_out_of_bounds_ids=(map_state or {}).get("out_of_bounds_ids", ()),
                            map_jump_ids=(map_state or {}).get("jump_candidate_ids", ()),
                            shadow_global_ids=shadow_map.get(camera, {}),
                        )
                        self.panes[camera]._shown_index = cached_index
                        if camera not in self._first_preview_recorded:
                            self._first_preview_recorded.add(camera)
                            startup_trace.record(
                                "FIRST_PYSIDE6_PREVIEW_FRAME", camera=camera,
                                source_id=meta.get("source_id"),
                                source_frame_index=meta.get("input_frame_index"),
                                source_pts_ns=meta.get("source_pts_ns"),
                                mux_frame_num=meta.get("frame_num"),
                                person_objects=len(self.panes[camera].objects),
                                camera_local_ids=[obj.get("tracker_id")
                                                  for obj in self.panes[camera].objects
                                                  if obj.get("tracker_id") is not None],
                            )
            if self.controller.error:
                self.banner.setText("LIVE TEST STOPPING — ERROR")
                self.status.setText(self.controller.error)
                self._stop_async()
                return
            states = [self.controller.source_state(sid) for sid in (0, 1)]
            stats = self.controller.stats
            ready_cameras = all(state == "ONLINE" for state in states)
            if ready_cameras and self.controller.ready_ns == 0:
                self.controller.ready_ns = time.monotonic_ns()
                self.controller.run_phase = "15-SECOND SMOKE"
                self.smoke_baseline = {
                    sid: {key: stats[sid][key] for key in
                          ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")}
                    for sid in (0, 1)}
            if not self.controller.ready_ns:
                self.banner.setText(f"CONNECTING CAMERAS — {states[0]} / {states[1]}")
                readiness_cap_ns = 120_000_000_000 if recorded_mode else 30_000_000_000
                if self.controller.started_ns and time.monotonic_ns() - self.controller.started_ns > readiness_cap_ns:
                    self.controller.error = (
                        "recorded decode-to-start-frame readiness timed out after 120 seconds"
                        if recorded_mode else
                        "camera readiness timed out after 30 seconds; closing both subscriptions"
                    )
                    self._stop_async(); return
            elapsed = ((time.monotonic_ns() - self.controller.ready_ns)/1e9) if self.controller.ready_ns else 0.0
            detections = sum(stats[sid]["pgie_person_detections"] for sid in (0, 1))
            tracked = sum(stats[sid]["tracker_person_observations"] for sid in (0, 1))
            fps = {camera: self.controller.cam_fps(sid) for camera, sid in (("CAM-02", 0), ("CAM-05", 1))}
            decoded_fps = {camera: self.controller.source_fps(sid) for camera, sid in (("CAM-02", 0), ("CAM-05", 1))}
            gpu = next((sample for sample in reversed(self.controller.gpu_samples)
                        if sample.get("status") == "AVAILABLE"), None)
            gpu_text = (f"GPU device-total {gpu['gpu_utilization_percent']}% · VRAM {gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB"
                        if gpu else "GPU/VRAM UNAVAILABLE")
            shadow_text = "OSNet shadow OFF"
            if self.controller.identity_shadow is not None:
                shadow_state = self.controller.identity_shadow.snapshot()
                shadow_text = (
                    f"OSNet shadow candidates {len(shadow_state['active_provisional_ids'])} "
                    f"· crops {shadow_state['counters']['features_generated']} "
                    f"· sample drops {shadow_state['counters']['queue_full_drops']} "
                    f"· no persistent writes"
                )
            people_text = f"person observations {tracked}" if detections else "NO_PEOPLE (no person detections observed)"
            source_label = "DEMO / RECORDED — NOT LIVE" if recorded_mode else "LIVE"
            self.status.setText(
                f"{source_label} · CAM-02 {states[0]} decoded {decoded_fps['CAM-02']:.1f} FPS / NvDCF {fps['CAM-02']:.1f} FPS "
                f"({stats[0]['frames_captured']} received, {stats[0]['frames_sent']} sent, {stats[0]['queue_drops']} host queue drops) · "
                f"CAM-05 {states[1]} decoded {decoded_fps['CAM-05']:.1f} FPS / NvDCF {fps['CAM-05']:.1f} FPS "
                f"({stats[1]['frames_captured']} received, {stats[1]['frames_sent']} sent, {stats[1]['queue_drops']} host queue drops) · "
                f"PGIE frames {stats[0]['pgie_frames']}/{stats[1]['pgie_frames']} · NvDCF frames {stats[0]['tracker_frames']}/{stats[1]['tracker_frames']} · "
                f"{people_text} · local IDs CAM-02 {len(stats[0]['local_tracker_ids'])}, CAM-05 {len(stats[1]['local_tracker_ids'])} · "
                f"{shadow_text} · {gpu_text} · elapsed {elapsed:.1f}s"
            )
            if (recorded_mode and map_projection is not None
                    and self.controller.identity_shadow is not None
                    and (self.active_preview_screenshot_path is None
                         or (self.map_pane.map_mode == "experimental-global"
                             and self.global_fusion_screenshot_path is None))
                    and all(self.panes[camera]._shown_index >= 0 for camera in CAMERAS)):
                shadow_map = self.controller.identity_shadow.assignment_map()
                visible_groups = visible_provisional_identity_groups(
                    shadow_map,
                    {
                        camera: {
                            str(obj.get("tracker_id"))
                            for obj in self.panes[camera].objects
                            if obj.get("tracker_id") is not None
                        }
                        for camera in CAMERAS
                    },
                    {camera: self.panes[camera].frame_key for camera in CAMERAS},
                    {camera: self.map_pane.states.get(camera) for camera in CAMERAS},
                    {
                        camera: self.panes[camera].shadow_global_ids
                        for camera in CAMERAS
                    },
                    self.map_pane.shadow_global_ids,
                )
                if visible_groups:
                    captured = self.grab()
                    fusion_snapshot = self.map_pane.last_fusion_snapshot or {}
                    eligible = [m for m in fusion_snapshot.get("markers", [])
                                if m.get("provisional_global_id")
                                and len(m["contributors"]) > 1
                                and all((o["source_id"], o["frame_index"], o["source_pts_ns"])
                                        == self.panes[o["camera_id"]].frame_key
                                        for o in m["contributors"])]
                    if self.active_preview_screenshot_path is None:
                        active_path = output / "entrance-provisional-global-id-review.png"
                        if captured.save(str(active_path)):
                            os.chmod(active_path, 0o600)
                            self.active_preview_screenshot_path = str(active_path)
                            fused_ids = {m["provisional_global_id"] for m in eligible}
                            self.provisional_identity_visual_evidence = [
                                {**row, "map_points_fused": row["provisional_global_id"] in fused_ids,
                                 "fusion_coordinate_space": "EXPERIMENTAL_LAYOUT_PIXELS"}
                                for row in visible_groups]
                    if (self.map_pane.map_mode == "experimental-global" and eligible
                            and self.global_fusion_screenshot_path is None):
                        fusion_path = output / "entrance-global-fusion-review.png"
                        if captured.save(str(fusion_path)):
                            os.chmod(fusion_path, 0o600)
                            self.global_fusion_screenshot_path = str(fusion_path)
                            self.global_fusion_visual_evidence = eligible
            if self.controller.ready_ns and not self.smoke_passed and elapsed >= args.smoke_seconds:
                smoke_ok = smoke_gate_passed(stats, self.smoke_baseline)
                if not smoke_ok:
                    self.controller.error = (
                        "recorded YOLO26m smoke failed: both sources did not reach mux, PGIE, NvDCF and exact display-frame joins"
                        if recorded_mode else
                        f"{args.smoke_seconds:.0f}-second smoke failed: both sources did not reach mux, PGIE, and NvDCF"
                    )
                    self._stop_async(); return
                self.smoke_passed = True
                action = post_smoke_action(smoke_only=args.smoke_only,
                                           gpu_samples=self.controller.gpu_samples)
                if action == "STOP_SMOKE_ONLY":
                    self.controller.run_phase = "SMOKE_PASS_STOPPED_AS_REQUESTED"
                    self.banner.setText(
                        f"{args.smoke_seconds:.0f}-SECOND RECORDED SMOKE PASS — STOPPING BY REQUEST"
                        if recorded_mode else
                        f"{args.smoke_seconds:.0f}-SECOND SMOKE PASS — STOPPING BY REQUEST"
                    )
                    self._stop_async(); return
                if action == "STOP_RESOURCE_GUARD":
                    self.review_skipped_reason = "extended review withheld: device-total GPU telemetry unavailable or >=95% in at least two of the last three samples"
                    self.controller.run_phase = "SMOKE_PASS_REVIEW_SKIPPED_RESOURCE_GUARD"
                    self.banner.setText("15-SECOND SMOKE PASS — EXTENDED REVIEW WITHHELD BY GPU SAFETY GUARD")
                    self._stop_async(); return
                self.review_started = time.monotonic()
                now_ns = time.monotonic_ns()
                planned_review_end_ns = now_ns + int(args.review_seconds * 1_000_000_000)
                self.review_deadline_ns = min(
                    planned_review_end_ns,
                    (self.controller.rtsp_deadline_ns if not recorded_mode else None)
                    or planned_review_end_ns,
                )
                self.controller.run_phase = (
                    f"{args.review_seconds:.0f}-SECOND RECORDED COMBINED REVIEW"
                    if recorded_mode else f"{args.review_seconds:.0f}-SECOND VISUAL REVIEW"
                )
                remaining = max(0.0, (self.review_deadline_ns - now_ns) / 1e9)
                self.banner.setText(
                    (f"RECORDED SMOKE PASS — {remaining:.1f}s COMBINED OFFLINE REVIEW · NOT LIVE"
                     if recorded_mode else
                     f"{args.smoke_seconds:.0f}-SECOND SMOKE PASS — LIVE REVIEW LIMITED TO {remaining:.1f}s BY RTSP HARD CAP")
                )
            elif self.smoke_passed:
                review_remaining = max(0.0, (self.review_deadline_ns-time.monotonic_ns())/1e9)
                self.banner.setText(
                    (f"RECORDED DEMO / NOT LIVE · {review_remaining:.1f}s remaining · LOCAL NvDCF"
                     if recorded_mode else
                     f"LIVE VISUAL REVIEW · {review_remaining:.1f}s remaining · LOCAL NvDCF")
                    + (" + UNVERIFIED OSNet SHADOW" if self.controller.identity_shadow is not None else " ONLY")
                )
                if review_remaining <= 0:
                    self.banner.setText(
                        "RECORDED BENCHMARK COMPLETE — DEEPSTREAM STOPPING"
                        if recorded_mode else "LIVE SESSION COMPLETE — RTSP AND DEEPSTREAM STOPPING"
                    )
                    self._stop_async()

        def _start_controller(self):
            startup_trace.record("CONTROLLER_THREAD_ENTERED", python_pid=os.getpid())
            controller = LiveController(args, output)
            controller.gpu_samples.extend(gpu_baseline)
            try:
                controller.start()
            except Exception as exc:
                controller.error = redact(f"{type(exc).__name__}: {exc}")
                controller.run_phase = "ERROR"
                startup_trace.record(
                    "CONTROLLER_START_FAILED", error=controller.error,
                    deepstream_exists=controller.pipe is not None,
                )
                controller.stop()
            startup_trace.record(
                "CONTROLLER_THREAD_RETURNED", error=controller.error or None,
                phase=controller.run_phase,
            )
            self.controller = controller

        def _stop_async(self):
            if self.finish_started: return
            self.finish_started = time.monotonic()
            if self.controller is not None:
                threading.Thread(target=self.controller.stop, daemon=True).start()
            self._shutdown_timer = QTimer(self)
            self._shutdown_timer.setInterval(200)
            self._shutdown_timer.timeout.connect(self._poll_shutdown)
            self._shutdown_timer.start()

        def _poll_shutdown(self):
            if self.controller is None:
                app.quit(); return
            if self.controller._stopped:
                self._shutdown_timer.stop()
                QTimer.singleShot(1000, app.quit)
            elif self.finish_started and time.monotonic() - self.finish_started > 45:
                self._shutdown_timer.stop()
                app.quit()

        def closeEvent(self, event):  # noqa: N802
            if self.controller is not None:
                self.controller.stop()
            event.accept()

    window = Window(); window.show()
    app.exec()
    if window.controller is not None:
        window.controller.stop()
        window.controller.process_events()
    map_screenshot_path = None
    if map_projection is not None:
        screenshot_path = output / "entrance-live-map-final.png"
        if window.grab().save(str(screenshot_path)):
            os.chmod(screenshot_path, 0o600)
            map_screenshot_path = str(screenshot_path)
    stats = window.controller.stats if window.controller else {}
    recorded_frame_map_path = None
    if recorded_mode and window.controller is not None:
        frame_map = output / "recorded-frame-map.jsonl"
        with frame_map.open("w", encoding="utf-8") as stream:
            os.chmod(frame_map, 0o600)
            for row in window.controller.recorded_frame_records:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        recorded_frame_map_path = str(frame_map)
    fusion_diagnostics_path = None
    fusion_records = list(window.map_pane.fusion_diagnostics)
    if window.map_pane.fusion is not None:
        fusion_path = output / "map-fusion-events.jsonl"
        with fusion_path.open("w", encoding="utf-8") as stream:
            os.chmod(fusion_path, 0o600)
            for row in fusion_records:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        fusion_diagnostics_path = str(fusion_path)
    selected_image = (window.controller.pipe.image if window.controller and window.controller.pipe else None)
    selected_image_id = (window.controller.pipe.image_id if window.controller and window.controller.pipe else None)
    staged_yolo_config_sha256 = None
    if args.detector == "yolo26m":
        from services.mv3dt_staging.live_tracking.yolo_candidate import sha256_file
        staged_config = output / "pgie-yolo26m-b2-engine-only.txt"
        if staged_config.is_file():
            staged_yolo_config_sha256 = sha256_file(staged_config)
    report = {
        "detector": args.detector,
        "detector_person_class_id": detector_person_class(args.detector),
        "image": selected_image,
        "image_id": selected_image_id,
        "yolo_asset_sha256": yolo_asset_hashes,
        "frozen_yolo_pgie_config_sha256": frozen_pgie_config_sha256,
        "staged_yolo_pgie_config_sha256": staged_yolo_config_sha256,
        "pipeline": (f"host GStreamer local-file decode + raw full-FOV scale-only 1920x1080 BGR appsrc → DeepStream 9.1 NVMM + legacy nvstreammux → {args.detector} → local NvDCF"
                     if recorded_mode else
                     f"host RTSP decode + raw full-FOV scale-only 1920x1080 BGR appsrc → DeepStream 9.1 NVMM + legacy nvstreammux → {args.detector} → local NvDCF"
                     if args.native_frame else
                     f"host RTSP decode + empirical rectification → BGR appsrc → DeepStream 9.1 NVMM + legacy nvstreammux → {args.detector} → local NvDCF"),
        "input_mode": ("RECORDED_ORIGINAL_RAW_FULL_FOV_SCALE_ONLY" if recorded_mode else
                       "RAW_UNRECTIFIED_FULL_FOV_SCALE_ONLY" if args.native_frame else "RECTIFIED"),
        "physical_rtsp_opened": False if recorded_mode else None,
        "camera_ownership_locks_claimed": False if recorded_mode else None,
        "source_geometry": ({camera: {"width": recorded_source_summaries[camera]["width"],
                                       "height": recorded_source_summaries[camera]["height"]}
                             for camera in CAMERAS} if recorded_mode else
                            {camera: {"width": NATIVE_SIZE[camera][0], "height": NATIVE_SIZE[camera][1]}
                             for camera in CAMERAS}),
        "recorded_sources": recorded_source_summaries,
        "recorded_playback": ({
            "mode": "local GStreamer qtdemux→h264parse→avdec_h264→videoconvert→BGR appsink",
            "paced_by": "original source PTS deltas mapped to each file's independent local monotonic playback origin",
            "playback_speed_x": float(getattr(args, "playback_speed", 1.0)),
            "duration_sec_semantics": "source media time; wall duration is divided by playback_speed_x",
            "original_source_frame_index": "presentation-order video-packet PTS index; runtime exact GstBuffer PTS cross-check",
            "original_source_pts": "retained in source_pts_ns and exact source-frame map",
            "pipeline_buf_pts": "generated separately in DeepStream appsrc running-time; not copied from original file PTS",
            "cross_camera_sync": "UNVERIFIED; sources use independent playback origins and offsets",
            "bounded_queues": {"per_camera_latest_input_frames": 1, "appsink_buffers": 1,
                                "display_frame_cache": 12,
                                "shadow_inference_frame_cache": SHADOW_INFERENCE_CACHE_FRAMES,
                                "osnet_jobs": 32},
            "frame_map_path": recorded_frame_map_path,
            "frame_map_rows": len(window.controller.recorded_frame_records)
                if window.controller is not None else 0,
            "gstreamer_capture_timing_by_camera": {
                worker.camera.camera_id: worker.recorded_timing_record()
                for worker in window.controller.capture_workers
                if isinstance(worker, RecordedCaptureWorker)
            } if window.controller is not None else {},
        } if recorded_mode else None),
        "native_mode_transform": ({"operation": "cv2.resize INTER_AREA; no crop/pad/undistortion/projective warp",
                                   "detector_canvas": "1920x1080",
                                   "scale_x_y": {"CAM-02": 0.6, "CAM-05": 0.75},
                                   "viewer": "original decoded native-resolution BGR frame; DeepStream boxes mapped back by inverse scale"}
                                  if args.native_frame else None),
        "map": ("EXPERIMENTAL / UNCALIBRATED; native homographies applied only to exact source/frame/PTS tracker observations; metric accuracy and cross-camera synchronization unverified"
                if map_projection is not None else
                "disabled; geometry and source-clock synchronization unverified"),
        "map_candidate": map_candidate_record,
        "map_projection_runtime": ({
            "counters_by_camera": map_projection.counters,
            "local_ids_camera_scoped": True,
            "out_of_layout_markers_clamped": False,
            "stale_tracker_results_joined_to_newer_frames": False,
            "live_ui_screenshot": window.active_preview_screenshot_path or map_screenshot_path,
            "active_frame_ui_screenshot": window.active_preview_screenshot_path,
            "post_stop_ui_screenshot": map_screenshot_path,
            "frame_join_contract": "source_id + input_frame_index + exact source_pts_ns from the same cached raw frame",
            "marker_freshness_seconds_from_frame_arrival": map_projection.marker_max_age_ns / 1e9,
        } if map_projection is not None else None),
        "map_fusion": ({
            "mode": window.map_pane.map_mode,
            "policy": window.map_pane.fusion.policy(),
            "diagnostics_path": fusion_diagnostics_path,
            "bounded_diagnostic_updates": len(fusion_records),
            "updates_with_multi_camera_fused_marker": sum(any(
                marker["key"].startswith("global:")
                and len(marker["contributor_frame_keys"]) > 1
                for marker in record["rendered"]["markers"]) for record in fusion_records),
            "split_group_diagnostics": [group for record in fusion_records
                                        for group in record["groups"] if group["decision"] == "SPLIT"],
            "active_frame_screenshot": window.global_fusion_screenshot_path,
            "screenshot_contributing_observations": window.global_fusion_visual_evidence,
            "identity_correctness": "UNVERIFIED",
            "metric_world_coordinate_fusion": False,
        } if window.map_pane.fusion is not None else None),
        "global_ids": (
            "OSNet provisional candidate labels only; no MV3DT IDs, cross-run persistence, or production writes"
            if args.automatic_reid_shadow else
            "NOT_RUN; no MV3DT association enabled"
        ),
        "persistent_identity": "NOT ENABLED",
        "identity_shadow": window.controller.identity_shadow_summary
            if window.controller and window.controller.identity_shadow_summary else None,
        "provisional_identity_visual_evidence": window.provisional_identity_visual_evidence,
        "identity_shadow_prewarm": window.controller.identity_shadow_prewarm
            if window.controller else None,
        "smoke_passed": window.smoke_passed,
        "extended_review_skipped_reason": window.review_skipped_reason,
        "started_utc": window.controller.started_utc if window.controller else None,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "session_duration_sec": round((window.controller.stopped_ns-window.controller.ready_ns)/1e9, 3)
            if window.controller and window.controller.ready_ns and window.controller.stopped_ns else None,
        "total_runtime_sec": round((window.controller.stopped_ns-window.controller.started_ns)/1e9, 3)
            if window.controller and window.controller.started_ns and window.controller.stopped_ns else None,
        "status": window.controller.error or window.controller.run_phase if window.controller else "START_FAILED",
        "deepstream_shutdown": ({
            "exit_code": window.controller.pipe.process.returncode,
            "eos_done_message_observed": window.controller.pipe.done_message_seen.is_set(),
            "status": next((row.get("status") for row in reversed(window.controller.event_records)
                            if row.get("type") == "DONE"), None),
            "recorded_mode_requires_eos_ack": recorded_mode,
            "result": ("PASS" if window.controller.pipe.process.returncode == 0
                       and window.controller.pipe.done_message_seen.is_set()
                       else "FAIL"),
        } if window.controller and window.controller.pipe else "NOT_STARTED"),
        "rtsp_timing": (None if recorded_mode else {
            "authorized_max_seconds": window.controller.authorized_max_seconds,
            "capture_stop_seconds_from_first_attempt": window.controller.capture_stop_seconds,
            "teardown_reserve_seconds": round(
                window.controller.authorized_max_seconds - window.controller.capture_stop_seconds, 3
            ),
            "first_connection_attempt_utc": window.controller.rtsp_first_attempt_utc,
            "first_connection_attempt_monotonic_ns": window.controller.rtsp_first_attempt_ns,
            "capture_deadline_monotonic_ns": window.controller.rtsp_deadline_ns,
            "hard_cap_triggered_utc": window.controller.rtsp_cap_triggered_utc,
            "hard_cap_triggered_monotonic_ns": window.controller.rtsp_cap_triggered_ns,
            "stop_reason": window.controller.rtsp_stop_reason or None,
            "cameras": {
                worker.camera.camera_id: worker.timing_record(
                    window.controller.rtsp_first_attempt_ns,
                    window.controller.rtsp_deadline_ns,
                    window.controller.authorized_max_seconds,
                ) for worker in window.controller.capture_workers
            },
            "latest_camera_close_monotonic_ns": max(
                (worker.close_return_ns for worker in window.controller.capture_workers),
                default=0,
            ) or None,
            "first_attempt_to_last_camera_close_seconds": (
                round((max(worker.close_return_ns for worker in window.controller.capture_workers)
                       - window.controller.rtsp_first_attempt_ns) / 1e9, 6)
                if window.controller.rtsp_first_attempt_ns is not None
                and all(worker.close_return_ns for worker in window.controller.capture_workers)
                else None
            ),
            "all_cameras_closed_within_authorized_cap": (
                all(rtsp_close_within_authorized_cap(
                    window.controller.rtsp_first_attempt_ns, worker.close_return_ns,
                    window.controller.authorized_max_seconds,
                ) for worker in window.controller.capture_workers)
                if window.controller.rtsp_first_attempt_ns is not None
                and window.controller.capture_workers
                and all(worker.close_return_ns for worker in window.controller.capture_workers)
                else None
            ),
            "ownership_released_utc": window.controller.ownership_released_utc,
            "ownership_released_monotonic_ns": window.controller.ownership_released_ns,
        } if window.controller else None),
        "visual_evidence": window.controller.evidence_summary if window.controller else None,
        "cameras": {}, "deepstream_stdout": str(output / "deepstream.stdout.log"),
        "deepstream_stderr": str(output / "deepstream.stderr.log"), "output_dir": str(output),
        "startup_diagnostics": str(startup_trace.path),
        "startup_diagnostics_error": startup_trace.error,
    }
    for camera, sid in (("CAM-02", 0), ("CAM-05", 1)):
        if stats:
            stat = stats[sid]
            report["cameras"][camera] = {
                key: (len(value) if isinstance(value, set) else value)
                for key, value in stat.items() if not isinstance(value, deque)
            }
            report["cameras"][camera]["measured_processed_fps"] = round(window.controller.cam_fps(sid), 3)
            report["cameras"][camera]["measured_decoded_fps"] = round(window.controller.source_fps(sid), 3)
            report["cameras"][camera]["sent_not_observed_at_mux"] = max(
                0, stat["frames_sent"] - stat["mux_frames"])
            report["cameras"][camera]["stage_latency_ms"] = {
                "nvdec_input_to_output": percentile_summary_ns(stat["decoder_latency_ns"]),
                "nvdec_output_to_appsink": percentile_summary_ns(stat["decoder_to_appsink_ns"]),
                "recorded_gstreamer_pull_wall_time": percentile_summary_ns(stat["recorded_gst_pull_ns"]),
                "recorded_playback_schedule_lateness": percentile_summary_ns(stat["recorded_playback_lateness_ns"]),
                "frame_preprocess": percentile_summary_ns(stat["frame_preprocess_ns"]),
                "appsink_arrival_to_pgie_output": percentile_summary_ns(stat["host_input_to_pgie_ns"]),
                "pgie_pad_elapsed": percentile_summary_ns(stat["pgie_pad_latency_ns"]),
                "appsink_arrival_to_tracker_output": percentile_summary_ns(stat["host_input_to_tracker_ns"]),
                "tracker_pad_elapsed": percentile_summary_ns(stat["tracker_pad_latency_ns"]),
                "tracker_event_handling_including_exact_lookup_and_enqueue": percentile_summary_ns(
                    stat["tracker_event_handling_ns"]),
                "shadow_exact_raster_cache_lookup": percentile_summary_ns(stat["shadow_frame_lookup_ns"]),
                "shadow_submit_frame_scheduling": percentile_summary_ns(stat["shadow_submit_frame_ns"]),
            }
            ages = sorted(stat["display_frame_age_frames"])
            report["cameras"][camera]["exact_display_frame_age_frames"] = (
                {"samples": len(ages), "p50": float(np.percentile(ages, 50)),
                 "p95": float(np.percentile(ages, 95)), "p99": float(np.percentile(ages, 99)),
                 "max": max(ages)} if ages else "NOT_MEASURED"
            )
    latency = sorted(window.controller.paint_latency_ms) if window.controller else []
    if latency:
        report["decoded_to_qt_paint_ms"] = {
            "samples": len(latency), "p50": round(float(np.percentile(latency, 50)), 3),
            "p95": round(float(np.percentile(latency, 95)), 3),
            "p99": round(float(np.percentile(latency, 99)), 3), "max": round(latency[-1], 3),
        }
    else:
        report["decoded_to_qt_paint_ms"] = "NOT_MEASURED"
    if window.controller:
        track_rows = [record for record in window.controller.event_records if record.get("type") == "TRACK"]
        pgie_rows = [record for record in window.controller.event_records if record.get("type") == "PGIE"]
        report["same_frame_high_iou_box_candidates"] = {
            "threshold_iou": 0.90,
            "pgie_pairs": high_overlap_pairs(pgie_rows, class_id=window.controller.person_class_id),
            "tracker_pairs": high_overlap_pairs(track_rows, class_id=window.controller.person_class_id),
            "interpretation": "overlap candidates only; not visually adjudicated false duplicates",
        }
        report["local_id_frame_spans"] = {
            "meaning": "camera-local NvDCF IDs and observed input-frame spans; no person-identity claim",
            "by_source_id": track_frame_summary(track_rows, class_id=window.controller.person_class_id),
        }
        report["stage_frame_records"] = {
            stage: {
                "source_id_0": sum(1 for row in window.controller.event_records
                                    if row.get("type") == stage and row.get("source_id") == 0),
                "source_id_1": sum(1 for row in window.controller.event_records
                                    if row.get("type") == stage and row.get("source_id") == 1),
                "missing_input_frame_index": sum(1 for row in window.controller.event_records
                                                 if row.get("type") == stage and row.get("input_frame_index") is None),
            }
            for stage in ("MUX", "PGIE", "TRACK")
        }
    report["gpu_preflight_samples"] = gpu_baseline
    report["gpu_device_total_samples"] = window.controller.gpu_samples if window.controller else []
    cpu_samples = window.controller.cpu_samples if window.controller else []
    cpu_summary = {}
    for key in ("system_cpu_busy_percent", "bridge_process_cpu_percent_of_one_core", "bridge_process_rss_mib"):
        values = [float(sample[key]) for sample in cpu_samples if sample.get(key) is not None]
        cpu_summary[key] = ({
            "samples": len(values), "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3), "max": round(max(values), 3),
        } if values else "NOT_MEASURED")
    report["cpu"] = {
        "scope": "host aggregate plus Python bridge process; DeepStream container CPU is not process-attributed",
        "summary": cpu_summary,
        "samples": cpu_samples,
    }
    report["camera_mapping"] = {"DeepStream source_id 0": "CAM-02", "DeepStream source_id 1": "CAM-05"}
    report["drop_counter_scope"] = (
        "queue_drops counts host latest-frame queue replacements; sent_not_observed_at_mux is appsrc/queue frames not seen at mux by EOS; "
        "display_frame_missing counts tracker outputs whose exact source frame had already left the 12-frame UI cache. "
        "A missing-cache record is not drawn with mismatched pixels/boxes."
    )
    report["timing_note"] = (
        "Original source PTS and presentation-order frame indices are preserved and cross-checked against the FFprobe packet-PTS map. Playback schedules frames using original source-PTS deltas on independent local monotonic origins. DeepStream appsrc buf_pts is generated separately in pipeline running-time and is stored alongside, never substituted for, source_pts_ns. Cross-camera exposure alignment is not asserted."
        if recorded_mode else
        "Each source receives increasing pipeline-running-time PTS; original RTSP PTS/DTS and host arrival monotonic timestamps are sidecar metadata. "
        "No cross-camera exposure alignment is asserted. RTSP attempt, first/last decoded frame, close-return, and ownership-release timestamps are recorded. "
        f"Frame acquisition stops at {window.controller.capture_stop_seconds:.0f} seconds from the first RTSP attempt, reserving "
        f"{window.controller.authorized_max_seconds - window.controller.capture_stop_seconds:.0f} seconds inside the "
        f"{window.controller.authorized_max_seconds:.0f}-second authorization cap for teardown."
        if window.controller else "RTSP capture was not started."
    )
    if args.save_visual_evidence and window.controller:
        summary = window.controller.evidence_summary
        crosswalk_misses = sum(
            int(stat.get("evidence_frame_missing", 0))
            + int(stat.get("evidence_crosswalk_failures", 0))
            for stat in window.controller.stats.values()
        )
        if summary is None:
            report["visual_evidence_gate"] = {
                "status": "FAIL", "reason": "writer summary unavailable at shutdown",
                "crosswalk_misses": crosswalk_misses,
            }
        else:
            per_camera = summary["camera_counts"]
            evidence_losses = sum(
                int(counts.get(key, 0))
                for counts in per_camera.values()
                for key in ("queue_dropped", "cap_skipped", "writer_error_skipped", "write_failed")
            )
            uncommitted = sum(max(0, int(counts["submitted"]) - int(counts["written"]))
                              for counts in per_camera.values())
            accepted = (summary.get("error") is None and evidence_losses == 0
                        and uncommitted == 0 and crosswalk_misses == 0)
            report["visual_evidence_gate"] = {
                "status": "PASS" if accepted else "PARTIAL",
                "queue_dropped_cap_skipped_or_writer_failed": evidence_losses,
                "submitted_not_written": uncommitted,
                "crosswalk_misses": crosswalk_misses,
                "writer_error": summary.get("error"),
                "tracking_continues_independently_of_evidence_queue": True,
            }
    report_path = output / "live-run.json"
    startup_trace.record(
        "RUNNER_FINISHED", status=report.get("status"),
        smoke_passed=report.get("smoke_passed"),
        deepstream_shutdown=report.get("deepstream_shutdown"),
    )
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    os.chmod(report_path, 0o600)
    print(json.dumps(report, indent=2, default=str))
    return 0 if window.smoke_passed and not (window.controller and window.controller.error) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    mux = sub.add_parser("mux-smoke", help="recorded-frame appsrc→legacy nvstreammux metadata test")
    mux.add_argument("--cam02-video", required=True)
    mux.add_argument("--cam05-video", required=True)
    mux.add_argument("--frames", type=int, default=8)
    mux.add_argument("--inference", action="store_true", help="also exercise the existing batch-2 PeopleNet Transformer and stock NvDCF")
    mux.add_argument("--bridge", default=str(BRIDGE_DEFAULT))
    mux.add_argument("--output")

    def add_viewer_args(command, *, recorded: bool) -> None:
        if recorded:
            command.add_argument("--cam02-video", default=str(
                ROOT / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_1.mp4"))
            command.add_argument("--cam05-video", default=str(
                ROOT / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_2.mp4"))
            command.add_argument("--start-frame-cam02", type=int, default=1314)
            command.add_argument("--start-frame-cam05", type=int, default=2506)
            command.add_argument("--duration-sec", type=float, default=12.0,
                                 help="maximum source-PTS playback duration per camera")
            command.add_argument("--playback-speed", type=float, default=1.0,
                                 help="recorded source-time pace multiplier (1.0=real-time, bounded 0.1..4.0)")
            command.set_defaults(native_frame=True)
        else:
            command.add_argument("--bridge", default=str(BRIDGE_DEFAULT))
            command.add_argument("--native-frame", action="store_true",
                                 help="bypass rectification; use raw full-FOV native camera frames, scale only for inference, and display overlays on original-resolution frames")
        command.add_argument("--layout")
        command.add_argument("--experimental-native-alignment", default=None,
                             help="hash-verified LIVE_NATIVE_ALIGNMENT JSON; staging display only, never production")
        command.add_argument("--automatic-reid-shadow", action="store_true",
                             help="optional in-memory OSNet candidate labels; no global-ID persistence or production writes")
        command.add_argument("--map-mode", choices=("local", "experimental-global"), default="local",
                             help="map display only: independent local observations or guarded experimental provisional-ID fusion")
        command.add_argument("--shadow-reid-device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--shadow-reid-sample-every", type=int, default=20)
        command.add_argument("--output")
        command.add_argument("--detector", choices=("peoplenet", "yolo26m"),
                             default="yolo26m" if recorded else "peoplenet")
        command.add_argument("--yolo-artifacts", default=(
            str(ROOT / ".runtime/sixcam-approved-bbox-20261006-r12-profile/models")
            if recorded else None),
            help="hash-verified frozen YOLO26m artifact directory under this checkout's .runtime")
        command.add_argument("--smoke-only", action="store_true",
                             help="stop after smoke and EOS/cleanup; do not continue to the review portion")
        if recorded:
            command.add_argument("--continue-after-smoke", action="store_false", dest="smoke_only",
                                 help="after a passing smoke and resource gate, continue for review-seconds")
            command.set_defaults(smoke_only=True)
        command.add_argument("--smoke-seconds", type=float, default=8.0 if recorded else 15.0)
        command.add_argument("--review-seconds", type=float, default=4.0 if recorded else 45.0)
        if not recorded:
            command.add_argument("--capture-stop-seconds", type=float, default=RTSP_CAPTURE_STOP_SECONDS,
                                 help="stop camera acquisition this many seconds after first RTSP attempt")
            command.add_argument("--authorized-max-seconds", type=float, default=AUTHORIZED_RTSP_MAX_SECONDS,
                                 help="hard maximum including camera teardown")
        command.add_argument("--save-visual-evidence", action="store_true",
                             help="write bounded lossless raw/annotated frame pairs with exact metadata")
        command.add_argument("--evidence-sample-every", type=int, default=6)
        command.add_argument("--evidence-max-samples-per-camera", type=int, default=180)

    live = sub.add_parser("live", help="one bounded live inference and Qt review")
    add_viewer_args(live, recorded=False)
    recorded = sub.add_parser("recorded", help="original CAM-02/CAM-05 files → the existing YOLO/NvDCF/Qt pipeline; never opens RTSP")
    add_viewer_args(recorded, recorded=True)
    args = parser.parse_args()
    if args.mode == "mux-smoke":
        if args.frames < 2 or args.frames > 30: parser.error("--frames must be 2..30")
        return run_mux_smoke(args)
    if args.mode == "live":
        try:
            validate_live_duration(capture_stop_seconds=args.capture_stop_seconds,
                                   authorized_max_seconds=args.authorized_max_seconds,
                                   smoke_seconds=args.smoke_seconds,
                                   review_seconds=args.review_seconds)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if not 0 < args.smoke_seconds <= args.duration_sec <= 180:
            parser.error("recorded duration must be <=180s and must cover the positive smoke interval")
        if (not math.isfinite(args.playback_speed)
                or not 0.1 <= args.playback_speed <= 4.0):
            parser.error("recorded --playback-speed must be finite and between 0.1 and 4.0")
        if args.review_seconds < 0 or (not args.smoke_only and args.smoke_seconds + args.review_seconds > args.duration_sec + 1e-6):
            parser.error("recorded smoke plus review must fit inside --duration-sec")
    if args.save_visual_evidence and not args.native_frame:
        parser.error("--save-visual-evidence requires --native-frame")
    if args.experimental_native_alignment and (not args.native_frame or not args.layout):
        parser.error("--experimental-native-alignment requires --native-frame and explicit --layout")
    if args.map_mode == "experimental-global" and not (
            args.automatic_reid_shadow and args.experimental_native_alignment):
        parser.error("experimental-global map mode requires accepted OSNet shadow bindings and native alignment")
    if args.automatic_reid_shadow and (
        not args.native_frame or not args.layout or not args.experimental_native_alignment
        or args.detector != "yolo26m"
    ):
        parser.error(
            "--automatic-reid-shadow requires YOLO26m, native-frame mode, explicit layout, "
            "and the verified experimental native alignment"
        )
    if args.shadow_reid_sample_every < 1:
        parser.error("--shadow-reid-sample-every must be positive")
    if args.evidence_sample_every < 1 or args.evidence_max_samples_per_camera < 1:
        parser.error("visual-evidence sampling bounds must be positive")
    if args.detector == "yolo26m" and not args.yolo_artifacts:
        parser.error("--detector yolo26m requires --yolo-artifacts")
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
