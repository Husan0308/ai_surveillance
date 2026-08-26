from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_meta_bridge.c")
LABEL_SOURCE = Path(__file__).with_name("native_label_style.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v2"
LIB_PATH = BUILD_DIR / "libcamera_v2_meta.so"


class _TrackRow(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("frame_num", ctypes.c_uint64),
        ("source_id", ctypes.c_uint32),
        ("left", ctypes.c_float),
        ("top", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("tracker_confidence", ctypes.c_float),
    ]


class _GlobalLabel(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("source_id", ctypes.c_uint32),
        ("global_id", ctypes.c_uint32),
        ("state_code", ctypes.c_uint32),
    ]


def _deepstream_root() -> Path:
    env = os.getenv("DEEPSTREAM_ROOT")
    if env:
        path = Path(env)
        if (path / "sources/includes/gstnvdsmeta.h").exists():
            return path

    candidates = [Path("/opt/nvidia/deepstream/deepstream")]
    candidates.extend(
        sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True)
    )
    for path in candidates:
        if (path / "sources/includes/gstnvdsmeta.h").exists() and (path / "lib").exists():
            return path
    raise RuntimeError("DeepStream headers were not found under /opt/nvidia/deepstream")


def ensure_bridge() -> Path:
    """Build only metadata/label helpers used by clean tracking and Global ID."""
    sources = (SOURCE, LABEL_SOURCE)
    for source in sources:
        if not source.exists():
            raise RuntimeError(f"metadata bridge source missing: {source}")

    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    newest = max(source.stat().st_mtime for source in sources)
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= newest:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for metadata bridge")
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
        str(LABEL_SOURCE),
        "-o",
        str(LIB_PATH),
        f"-I{include_dir}",
        f"-L{lib_dir}",
        f"-Wl,-rpath,{lib_dir}",
        *pkg_flags,
    ]
    errors: list[str] = []
    for libs in (
        ["-lnvds_meta", "-lnvdsgst_meta"],
        ["-lnvds_meta"],
        ["-lnvdsgst_meta", "-lnvds_meta"],
    ):
        result = subprocess.run(
            [*common, *libs],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "compiler error").strip())
    raise RuntimeError("metadata bridge compile failed: " + " | ".join(errors[-2:]))


class NativeMetaBridge:
    def __init__(self) -> None:
        self.path = ensure_bridge()
        self.lib = ctypes.CDLL(str(self.path))

        self.lib.camera_v2_add_boxes.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.camera_v2_add_boxes.restype = ctypes.c_int
        self.lib.camera_v2_add_tracked_boxes.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.camera_v2_add_tracked_boxes.restype = ctypes.c_int
        self.lib.camera_v2_apply_detector_result.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.camera_v2_apply_detector_result.restype = ctypes.c_int
        self.lib.camera_v2_style_and_count_tracked.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_style_and_count_tracked.restype = ctypes.c_int
        self.lib.camera_v2_apply_local_track_style.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_apply_local_track_style.restype = ctypes.c_int
        self.lib.camera_v2_copy_tracks.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(_TrackRow),
            ctypes.c_int,
        ]
        self.lib.camera_v2_copy_tracks.restype = ctypes.c_int
        self.lib.camera_v2_apply_global_track_style.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(_GlobalLabel),
            ctypes.c_int,
        ]
        self.lib.camera_v2_apply_global_track_style.restype = ctypes.c_int
        self.lib.camera_v2_expand_display_boxes.argtypes = [
            ctypes.c_uint64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.camera_v2_expand_display_boxes.restype = ctypes.c_int
        self.lib.camera_v2_add_wall_rects.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.camera_v2_add_wall_rects.restype = ctypes.c_int
        self.lib.camera_v2_add_wall_tracks.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.camera_v2_add_wall_tracks.restype = ctypes.c_int
        self.lib.camera_v2_count_tracked.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_count_tracked.restype = ctypes.c_int
        self.lib.camera_v2_shadow_promoted_total.argtypes = []
        self.lib.camera_v2_shadow_promoted_total.restype = ctypes.c_uint64

    @staticmethod
    def _payload(boxes):
        if not boxes:
            return None, None
        flat: list[float] = []
        for x1, y1, x2, y2, conf in boxes:
            flat.extend((float(x1), float(y1), float(x2), float(y2), float(conf)))
        payload = (ctypes.c_float * len(flat))(*flat)
        return payload, ctypes.cast(payload, ctypes.POINTER(ctypes.c_float))

    def add_boxes(self, gst_buffer, source_id: int, boxes) -> int:
        payload, pointer = self._payload(boxes)
        _ = payload
        if pointer is None:
            return 0
        return int(
            self.lib.camera_v2_add_boxes(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                pointer,
                ctypes.c_int(len(boxes)),
            )
        )

    def add_tracked_boxes(self, gst_buffer, source_id: int, boxes) -> int:
        if not boxes:
            return 0
        flat: list[float] = []
        valid = 0
        for track_id, x1, y1, x2, y2, conf in boxes:
            if int(track_id) < 0:
                continue
            flat.extend(
                (
                    float(track_id),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                )
            )
            valid += 1
        if not flat:
            return 0
        payload = (ctypes.c_float * len(flat))(*flat)
        pointer = ctypes.cast(payload, ctypes.POINTER(ctypes.c_float))
        return int(
            self.lib.camera_v2_add_tracked_boxes(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                pointer,
                ctypes.c_int(valid),
            )
        )

    def apply_detector_result(self, gst_buffer, source_id: int, boxes) -> int:
        payload, pointer = self._payload(boxes)
        _ = payload
        if pointer is None:
            pointer = ctypes.POINTER(ctypes.c_float)()
        return int(
            self.lib.camera_v2_apply_detector_result(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                pointer,
                ctypes.c_int(len(boxes)),
            )
        )

    def style_and_count_tracked(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v2_style_and_count_tracked(
                ctypes.c_uint64(hash(gst_buffer))
            )
        )

    def apply_local_track_style(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v2_apply_local_track_style(
                ctypes.c_uint64(hash(gst_buffer))
            )
        )

    def copy_tracks(self, gst_buffer, max_rows: int = 128) -> list[dict]:
        max_rows = max(1, min(512, int(max_rows)))
        payload = (_TrackRow * max_rows)()
        count = int(
            self.lib.camera_v2_copy_tracks(
                ctypes.c_uint64(hash(gst_buffer)),
                payload,
                ctypes.c_int(max_rows),
            )
        )
        if count <= 0:
            return []
        return [
            {
                "object_id": int(row.object_id),
                "frame_num": int(row.frame_num),
                "source_id": int(row.source_id),
                "left": float(row.left),
                "top": float(row.top),
                "width": float(row.width),
                "height": float(row.height),
                "confidence": float(row.confidence),
                "tracker_confidence": float(row.tracker_confidence),
            }
            for row in payload[: min(count, max_rows)]
        ]

    @staticmethod
    def _state_code(state: str) -> int:
        value = str(state or "").upper()
        if value == "CONFIRMED":
            return 2
        if value == "SUSPECT":
            return 3
        return 1

    def apply_global_track_style(self, gst_buffer, mappings: list[dict]) -> int:
        rows = []
        for mapping in mappings:
            gid = int(mapping.get("global_id") or 0)
            if gid <= 0:
                continue
            rows.append(
                _GlobalLabel(
                    object_id=int(mapping["object_id"]),
                    source_id=int(mapping["source_id"]),
                    global_id=gid,
                    state_code=self._state_code(mapping.get("state", "TENTATIVE")),
                )
            )
        if not rows:
            return 0
        payload = (_GlobalLabel * len(rows))(*rows)
        return int(
            self.lib.camera_v2_apply_global_track_style(
                ctypes.c_uint64(hash(gst_buffer)),
                payload,
                ctypes.c_int(len(rows)),
            )
        )

    def expand_display_boxes(
        self,
        gst_buffer,
        *,
        side_margin: float = 0.08,
        top_margin: float = 0.04,
        bottom_margin: float = 0.10,
    ) -> int:
        return int(
            self.lib.camera_v2_expand_display_boxes(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_float(float(side_margin)),
                ctypes.c_float(float(top_margin)),
                ctypes.c_float(float(bottom_margin)),
            )
        )

    def add_wall_rects(self, gst_buffer, boxes) -> int:
        payload, pointer = self._payload(boxes)
        _ = payload
        if pointer is None:
            return 0
        return int(
            self.lib.camera_v2_add_wall_rects(
                ctypes.c_uint64(hash(gst_buffer)),
                pointer,
                ctypes.c_int(len(boxes)),
            )
        )

    def add_wall_tracks(self, gst_buffer, tracks) -> int:
        if not tracks:
            return 0
        flat: list[float] = []
        for track_id, x1, y1, x2, y2, conf in tracks:
            flat.extend(
                (
                    float(track_id),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                )
            )
        payload = (ctypes.c_float * len(flat))(*flat)
        pointer = ctypes.cast(payload, ctypes.POINTER(ctypes.c_float))
        return int(
            self.lib.camera_v2_add_wall_tracks(
                ctypes.c_uint64(hash(gst_buffer)),
                pointer,
                ctypes.c_int(len(tracks)),
            )
        )

    def count_tracked(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v2_count_tracked(ctypes.c_uint64(hash(gst_buffer)))
        )

    def shadow_promoted_total(self) -> int:
        return int(self.lib.camera_v2_shadow_promoted_total())
