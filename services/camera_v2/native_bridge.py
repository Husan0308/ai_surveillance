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


def _deepstream_root() -> Path:
    env = os.getenv("DEEPSTREAM_ROOT")
    if env:
        p = Path(env)
        if (p / "sources/includes/gstnvdsmeta.h").exists():
            return p

    candidates = [Path("/opt/nvidia/deepstream/deepstream")]
    candidates.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
    for p in candidates:
        if (p / "sources/includes/gstnvdsmeta.h").exists() and (p / "lib").exists():
            return p
    raise RuntimeError("DeepStream headers were not found under /opt/nvidia/deepstream")


def ensure_bridge() -> Path:
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

        self.lib.camera_v2_apply_identity_style.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_apply_identity_style.restype = ctypes.c_int

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

    def apply_detector_result(
        self,
        gst_buffer,
        source_id: int,
        boxes: list[tuple[float, float, float, float, float]],
    ) -> int:
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
            # Geometry smoothing/display-gap bridging runs first. Identity styling
            # runs last so label chips stay attached to the final visible bbox,
            # including bounded display-only hold boxes.
            self.lib.camera_v2_smooth_display_boxes(buffer_ptr)
            self.lib.camera_v2_apply_identity_style(buffer_ptr)
        return count

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
        return int(self.lib.camera_v2_heatmap_update(ctypes.c_uint64(hash(gst_buffer))))

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
        return int(self.lib.camera_v2_count_tracked(ctypes.c_uint64(hash(gst_buffer))))
