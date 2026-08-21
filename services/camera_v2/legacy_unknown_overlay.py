from __future__ import annotations

"""GPU-v2/Core-v1 yellow Unknown presentation for the current old-UI detector.

Detector/tracker behavior is intentionally left untouched.  This module only
exposes the already-confirmed Core-v1 visual track IDs to OSD and renders the old
``Unknown_C{camera}_{track}`` labels in the final post-tiler coordinate system.
"""

import ctypes
import math
import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_unknown_overlay.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v2"
LIB_PATH = BUILD_DIR / "libcamera_v2_unknown_overlay.so"


def _deepstream_root() -> Path:
    env = os.getenv("DEEPSTREAM_ROOT")
    if env:
        path = Path(env)
        if (path / "sources/includes/gstnvdsmeta.h").exists():
            return path
    candidates = [Path("/opt/nvidia/deepstream/deepstream")]
    candidates.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
    for path in candidates:
        if (path / "sources/includes/gstnvdsmeta.h").exists() and (path / "lib").exists():
            return path
    raise RuntimeError("DeepStream headers were not found under /opt/nvidia/deepstream")


def _ensure_library() -> Path:
    if not SOURCE.exists():
        raise RuntimeError(f"legacy Unknown overlay source missing: {SOURCE}")
    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for the legacy Unknown overlay")
    pkg_flags = shlex.split(
        subprocess.check_output(
            [pkg, "--cflags", "--libs", "gstreamer-1.0", "glib-2.0"],
            text=True,
        ).strip()
    )
    common = [
        gcc,
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c11",
        str(SOURCE),
        "-o",
        str(LIB_PATH),
        f"-I{include_dir}",
        f"-L{lib_dir}",
        f"-Wl,-rpath,{lib_dir}",
        *pkg_flags,
    ]
    errors = []
    for libs in (["-lnvds_meta", "-lnvdsgst_meta"], ["-lnvds_meta"], ["-lnvdsgst_meta", "-lnvds_meta"]):
        result = subprocess.run([*common, *libs], capture_output=True, text=True, check=False)
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "unknown compiler error").strip())
    raise RuntimeError("legacy Unknown overlay compile failed: " + " | ".join(errors[-2:]))


class _NativeUnknownOverlay:
    def __init__(self) -> None:
        self.path = _ensure_library()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v2_add_unknown_tracks.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
        ]
        self.lib.camera_v2_add_unknown_tracks.restype = ctypes.c_int

    def add(self, gst_buffer, source_id: int, rows) -> int:
        if not rows:
            return 0
        flat = []
        track_ids = []
        for x1, y1, x2, y2, confidence, track_id in rows:
            flat.extend((float(x1), float(y1), float(x2), float(y2), float(confidence)))
            track_ids.append(int(track_id))
        float_type = ctypes.c_float * len(flat)
        id_type = ctypes.c_uint64 * len(track_ids)
        payload = float_type(*flat)
        ids = id_type(*track_ids)
        return int(
            self.lib.camera_v2_add_unknown_tracks(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                ctypes.cast(payload, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ids, ctypes.POINTER(ctypes.c_uint64)),
                ctypes.c_int(len(track_ids)),
            )
        )


def _box_error(a, b) -> float:
    aw = max(8.0, float(a.x2) - float(a.x1))
    ah = max(8.0, float(a.y2) - float(a.y1))
    return (
        abs(float(a.x1) - float(b.x1)) / aw
        + abs(float(a.y1) - float(b.y1)) / ah
        + abs(float(a.x2) - float(b.x2)) / aw
        + abs(float(a.y2) - float(b.y2)) / ah
    )


def _visible_tracks(box_manager, cid: str, now: float):
    """Return the exact visible Core-v1 boxes paired with their stable local IDs."""
    with box_manager.lock:
        tracker = box_manager._trackers.get(cid)
        if tracker is None:
            return []

        visible = tracker.visible(float(now), target_time=float(now))
        if not visible:
            return []

        with tracker._lock:
            predicted = []
            target = float(now)
            for track in tracker._tracks.values():
                source_age = target - float(track.last_observation)
                if source_age < -1e-6 or source_age > float(tracker.hold_sec):
                    continue
                if int(track.hits) < int(tracker.strong_confirm_hits):
                    continue
                box, _horizon = tracker._visible_prediction(track, target)
                predicted.append((int(track.track_id), box))

        rows = []
        used = set()
        for box in visible:
            best = None
            best_error = float("inf")
            for track_id, predicted_box in predicted:
                if track_id in used:
                    continue
                error = _box_error(box, predicted_box)
                if error < best_error:
                    best_error = error
                    best = track_id
            if best is None:
                continue
            used.add(best)
            rows.append(
                (
                    float(box.x1),
                    float(box.y1),
                    float(box.x2),
                    float(box.y2),
                    float(box.confidence),
                    int(best),
                )
            )
        return rows


def _to_wall(self, source_id: int, rows):
    if not rows:
        return []
    try:
        focus = int(self.tiler.get_property("show-source"))
    except Exception:
        focus = -1

    wall_w = float(max(1, int(self.wall_width)))
    wall_h = float(max(1, int(self.wall_height)))
    src_w = float(max(1, int(self.frame_width)))
    src_h = float(max(1, int(self.frame_height)))

    if focus >= 0:
        if int(source_id) != focus:
            return []
        left = 0.0
        top = 0.0
        tile_w = wall_w
        tile_h = wall_h
    else:
        columns = max(1, int(getattr(self, "tiler_columns", 2)))
        grid_rows = max(1, int(getattr(self, "tiler_rows", 3)))
        column = int(source_id) % columns
        row = int(source_id) // columns
        if row >= grid_rows:
            return []
        tile_w = wall_w / float(columns)
        tile_h = wall_h / float(grid_rows)
        left = float(column) * tile_w
        top = float(row) * tile_h

    sx = tile_w / src_w
    sy = tile_h / src_h
    mapped = []
    for x1, y1, x2, y2, confidence, track_id in rows:
        wx1 = left + max(0.0, min(src_w, float(x1))) * sx
        wy1 = top + max(0.0, min(src_h, float(y1))) * sy
        wx2 = left + max(0.0, min(src_w, float(x2))) * sx
        wy2 = top + max(0.0, min(src_h, float(y2))) * sy
        if wx2 <= wx1 or wy2 <= wy1:
            continue
        mapped.append((wx1, wy1, wx2, wy2, float(confidence), int(track_id)))
    return mapped


def _unknown_overlay_probe(self, _pad, info):
    buffer = info.get_buffer()
    if buffer is None:
        return self.Gst.PadProbeReturn.OK

    renderer = getattr(self, "_legacy_unknown_renderer", None)
    if renderer is None:
        return self.Gst.PadProbeReturn.OK

    now = time.monotonic()
    requested = 0
    added = 0
    labels = []
    for cid, source_id in self.camera_index.items():
        rows = _visible_tracks(self.boxes, cid, now)
        wall_rows = _to_wall(self, int(source_id), rows)
        if not wall_rows:
            continue
        requested += len(wall_rows)
        result = renderer.add(buffer, int(source_id), wall_rows)
        if result > 0:
            added += int(result)
            labels.extend(
                f"Unknown_C{int(source_id)+1}_{int(row[5]):02d}" for row in wall_rows[:result]
            )

    if added:
        with self.det_lock:
            self.meta_boxes += added

    budget = int(getattr(self, "_legacy_unknown_log_budget", 30))
    if requested and budget > 0:
        self._legacy_unknown_log_budget = budget - 1
        try:
            focus = int(self.tiler.get_property("show-source"))
        except Exception:
            focus = -1
        print(
            "LEGACY_UNKNOWN_OVERLAY "
            f"focus={focus} requested={requested} injected={added} "
            f"labels=[{' '.join(labels[:8])}] stage=post-tiler-pre-osd",
            flush=True,
        )
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    from . import detection

    cls = detection.CameraDetectionV2
    if not getattr(cls, "_old_ui_overlay_installed", False):
        raise RuntimeError("legacy Unknown overlay requires old UI detection backend first")
    if getattr(cls, "_legacy_unknown_overlay_installed", False):
        return

    # The already-installed old-UI init wrapper reads this class method when the
    # runtime instance is created, so replacing it here cleanly removes the green
    # raw overlay instead of stacking another rectangle on top.
    cls._old_ui_post_tiler_overlay_probe = _unknown_overlay_probe

    original_init = cls.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._legacy_unknown_renderer = _NativeUnknownOverlay()
        self._legacy_unknown_log_budget = 30
        self._set_if(self.osd, "display-bbox", True)
        self._set_if(self.osd, "display-text", True)
        print(
            "LEGACY_UNKNOWN_OVERLAY_READY source=gpu-v2-clean/native_label_style "
            "unknown=yellow label=Unknown_C{camera}_{track} "
            "track_ids=core-v1-visual grid+fullscreen=post-tiler",
            flush=True,
        )

    cls.__init__ = wrapped_init
    cls._legacy_unknown_overlay_installed = True
