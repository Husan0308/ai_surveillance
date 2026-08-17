from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_meta_bridge.c")
SMOOTHER_SOURCE = Path(__file__).with_name("native_display_smoother.c")
LABEL_SOURCE = Path(__file__).with_name("native_label_style.c")
HEATMAP_SOURCE = Path(__file__).with_name("native_heatmap.c")
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
    candidates.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
    for path in candidates:
        if (path / "sources/includes/gstnvdsmeta.h").exists() and (path / "lib").exists():
            return path
    raise RuntimeError("DeepStream headers were not found under /opt/nvidia/deepstream")


def ensure_bridge() -> Path:
    """Build metadata helpers required by detection/tracking/ReID/heatmap."""
    sources = (SOURCE, SMOOTHER_SOURCE, LABEL_SOURCE, HEATMAP_SOURCE)
    for src in sources:
        if not src.exists():
            raise RuntimeError(f"metadata bridge source missing: {src}")

    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    newest_source_mtime = max(src.stat().st_mtime for src in sources)
    rebuild = not LIB_PATH.exists() or LIB_PATH.stat().st_mtime < newest_source_mtime
    if not rebuild:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required to build the DeepStream metadata bridge")

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
        str(SMOOTHER_SOURCE),
        str(LABEL_SOURCE),
        str(HEATMAP_SOURCE),
        "-o",
        str(LIB_PATH),
        f"-I{include_dir}",
        f"-L{lib_dir}",
        f"-Wl,-rpath,{lib_dir}",
        *pkg_flags,
    ]

    attempts = [
        ["-lnvds_meta", "-lnvdsgst_meta"],
        ["-lnvds_meta"],
        ["-lnvdsgst_meta", "-lnvds_meta"],
    ]
    errors: list[str] = []
    for libs in attempts:
        result = subprocess.run([*common, *libs], capture_output=True, text=True, check=False)
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "unknown compiler error").strip())

    raise RuntimeError("metadata bridge compile failed: " + " | ".join(errors[-2:]))


class NativeMetaBridge:
    def __init__(self) -> None:
        path = ensure_bridge()
        self.lib = ctypes.CDLL(str(path))

        self.lib.camera_v2_add_boxes.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.camera_v2_add_boxes.restype = ctypes.c_int

        self.lib.camera_v2_apply_detector_result.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.camera_v2_apply_detector_result.restype = ctypes.c_int

        self.lib.camera_v2_style_and_count_tracked.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_style_and_count_tracked.restype = ctypes.c_int
        self.lib.camera_v2_smooth_display_boxes.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_smooth_display_boxes.restype = ctypes.c_int
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

        self.lib.camera_v2_count_tracked.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_count_tracked.restype = ctypes.c_int
        self.lib.camera_v2_shadow_promoted_total.argtypes = []
        self.lib.camera_v2_shadow_promoted_total.restype = ctypes.c_uint64

        self.lib.camera_v2_heatmap_configure.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint,
        ]
        self.lib.camera_v2_heatmap_configure.restype = None
        self.lib.camera_v2_heatmap_reset.argtypes = []
        self.lib.camera_v2_heatmap_reset.restype = None
        self.lib.camera_v2_heatmap_update.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_heatmap_update.restype = ctypes.c_int
        self.lib.camera_v2_heatmap_render.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self.lib.camera_v2_heatmap_render.restype = ctypes.c_int
        self.lib.camera_v2_heatmap_rendered_points_total.argtypes = []
        self.lib.camera_v2_heatmap_rendered_points_total.restype = ctypes.c_uint64
        self.path = path

    @staticmethod
    def _payload(boxes: list[tuple[float, float, float, float, float]]):
        if not boxes:
            return None, None
        flat: list[float] = []
        for x1, y1, x2, y2, conf in boxes:
            flat.extend((float(x1), float(y1), float(x2), float(y2), float(conf)))
        array_type = ctypes.c_float * len(flat)
        payload = array_type(*flat)
        return payload, ctypes.cast(payload, ctypes.POINTER(ctypes.c_float))

    def add_boxes(self, gst_buffer, source_id: int, boxes: list[tuple[float, float, float, float, float]]) -> int:
        if not boxes:
            return 0
        payload, pointer = self._payload(boxes)
        _ = payload
        return int(
            self.lib.camera_v2_add_boxes(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                pointer,
                ctypes.c_int(len(boxes)),
            )
        )

    def apply_detector_result(self, gst_buffer, source_id: int, boxes: list[tuple[float, float, float, float, float]]) -> int:
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
        buffer_ptr = ctypes.c_uint64(hash(gst_buffer))
        count = int(self.lib.camera_v2_style_and_count_tracked(buffer_ptr))
        if count >= 0:
            self.lib.camera_v2_smooth_display_boxes(buffer_ptr)
        return count

    def apply_local_track_style(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v2_apply_local_track_style(
                ctypes.c_uint64(hash(gst_buffer))
            )
        )

    def copy_tracks(self, gst_buffer, max_rows: int = 128) -> list[dict]:
        max_rows = max(1, min(512, int(max_rows)))
        array_type = _TrackRow * max_rows
        payload = array_type()
        count = int(
            self.lib.camera_v2_copy_tracks(
                ctypes.c_uint64(hash(gst_buffer)),
                payload,
                ctypes.c_int(max_rows),
            )
        )
        if count <= 0:
            return []
        output: list[dict] = []
        for row in payload[: min(count, max_rows)]:
            output.append(
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
            )
        return output

    @staticmethod
    def _state_code(state: str) -> int:
        value = str(state or "").upper()
        if value == "CONFIRMED":
            return 2
        if value == "SUSPECT":
            return 3
        return 1

    def apply_global_track_style(self, gst_buffer, mappings: list[dict]) -> int:
        if not mappings:
            return 0
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
        array_type = _GlobalLabel * len(rows)
        payload = array_type(*rows)
        return int(
            self.lib.camera_v2_apply_global_track_style(
                ctypes.c_uint64(hash(gst_buffer)),
                payload,
                ctypes.c_int(len(rows)),
            )
        )

    def configure_heatmap(
        self,
        *,
        deposit: float = 0.008,
        decay: float = 0.99990,
        low_threshold: float = 0.015,
        yellow_threshold: float = 0.28,
        red_threshold: float = 0.62,
        max_points_per_source: int = 24,
    ) -> None:
        self.lib.camera_v2_heatmap_configure(
            ctypes.c_float(float(deposit)),
            ctypes.c_float(float(decay)),
            ctypes.c_float(float(low_threshold)),
            ctypes.c_float(float(yellow_threshold)),
            ctypes.c_float(float(red_threshold)),
            ctypes.c_uint(int(max_points_per_source)),
        )

    def reset_heatmap(self) -> None:
        self.lib.camera_v2_heatmap_reset()

    def heatmap_update(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v2_heatmap_update(ctypes.c_uint64(hash(gst_buffer)))
        )

    def heatmap_render(
        self,
        gst_buffer,
        *,
        wall_width: int,
        wall_height: int,
        rows: int = 2,
        columns: int = 3,
        source_count: int = 6,
    ) -> int:
        return int(
            self.lib.camera_v2_heatmap_render(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(wall_width)),
                ctypes.c_uint(int(wall_height)),
                ctypes.c_uint(int(rows)),
                ctypes.c_uint(int(columns)),
                ctypes.c_uint(int(source_count)),
            )
        )

    def heatmap_rendered_points_total(self) -> int:
        return int(self.lib.camera_v2_heatmap_rendered_points_total())

    def shadow_promoted_total(self) -> int:
        return int(self.lib.camera_v2_shadow_promoted_total())

    def count_tracked(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v2_count_tracked(ctypes.c_uint64(hash(gst_buffer)))
        )
