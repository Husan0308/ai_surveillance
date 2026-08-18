from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_pose_heatmap.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v2"
LIB_PATH = BUILD_DIR / "libcamera_v2_pose_heatmap.so"


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


def ensure_pose_heatmap_bridge() -> Path:
    if not SOURCE.exists():
        raise RuntimeError(f"pose heatmap source missing: {SOURCE}")
    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for pose heatmap bridge")
    pkg_flags = shlex.split(
        subprocess.check_output(
            [pkg, "--cflags", "--libs", "gstreamer-1.0", "glib-2.0"],
            text=True,
        ).strip()
    )
    base = [
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
    attempts = [
        ["-lnvds_meta", "-lnvdsgst_meta"],
        ["-lnvds_meta"],
        ["-lnvdsgst_meta", "-lnvds_meta"],
    ]
    errors: list[str] = []
    for libs in attempts:
        result = subprocess.run([*base, *libs], capture_output=True, text=True, check=False)
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "unknown compiler error").strip())
    raise RuntimeError("pose heatmap bridge compile failed: " + " | ".join(errors[-2:]))


class PoseHeatmapBridge:
    def __init__(self) -> None:
        self.path = ensure_pose_heatmap_bridge()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v2_pose_heatmap_configure.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint,
        ]
        self.lib.camera_v2_pose_heatmap_configure.restype = None
        self.lib.camera_v2_pose_heatmap_reset.argtypes = []
        self.lib.camera_v2_pose_heatmap_reset.restype = None
        self.lib.camera_v2_pose_heatmap_update_anchor.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.camera_v2_pose_heatmap_update_anchor.restype = ctypes.c_int
        self.lib.camera_v2_pose_heatmap_render.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.lib.camera_v2_pose_heatmap_render.restype = ctypes.c_int
        self.lib.camera_v2_pose_heatmap_rendered_points_total.argtypes = []
        self.lib.camera_v2_pose_heatmap_rendered_points_total.restype = ctypes.c_uint64

    def configure(
        self,
        *,
        deposit: float,
        decay: float,
        low_threshold: float,
        yellow_threshold: float,
        red_threshold: float,
        max_points_per_source: int,
    ) -> None:
        self.lib.camera_v2_pose_heatmap_configure(
            ctypes.c_float(float(deposit)),
            ctypes.c_float(float(decay)),
            ctypes.c_float(float(low_threshold)),
            ctypes.c_float(float(yellow_threshold)),
            ctypes.c_float(float(red_threshold)),
            ctypes.c_uint(int(max_points_per_source)),
        )

    def reset(self) -> None:
        self.lib.camera_v2_pose_heatmap_reset()

    def update_ankle(
        self,
        *,
        source_id: int,
        object_id: int,
        tick: int,
        nx: float,
        ny: float,
        confidence: float,
    ) -> int:
        return int(
            self.lib.camera_v2_pose_heatmap_update_anchor(
                ctypes.c_uint(int(source_id)),
                ctypes.c_uint64(int(object_id)),
                ctypes.c_uint64(int(tick)),
                ctypes.c_float(float(nx)),
                ctypes.c_float(float(ny)),
                ctypes.c_float(float(confidence)),
            )
        )

    def render(
        self,
        gst_buffer,
        *,
        wall_width: int,
        wall_height: int,
        rows: int,
        columns: int,
        source_count: int,
        focus_source: int = -1,
    ) -> int:
        return int(
            self.lib.camera_v2_pose_heatmap_render(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(wall_width)),
                ctypes.c_uint(int(wall_height)),
                ctypes.c_uint(int(rows)),
                ctypes.c_uint(int(columns)),
                ctypes.c_uint(int(source_count)),
                ctypes.c_int(int(focus_source)),
            )
        )

    def rendered_points_total(self) -> int:
        return int(self.lib.camera_v2_pose_heatmap_rendered_points_total())
