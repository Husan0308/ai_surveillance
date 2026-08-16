from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_qt_heatmap.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v2"
LIB_PATH = BUILD_DIR / "libcamera_v2_qt_heatmap.so"


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


def ensure_qt_heatmap_bridge() -> Path:
    if not SOURCE.exists():
        raise RuntimeError(f"Qt heatmap source missing: {SOURCE}")
    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required to build Qt heatmap bridge")
    pkg_flags = shlex.split(
        subprocess.check_output([pkg, "--cflags", "--libs", "gstreamer-1.0", "glib-2.0"], text=True).strip()
    )
    common = [
        gcc, "-shared", "-fPIC", "-O2", "-std=c11", str(SOURCE),
        "-o", str(LIB_PATH), f"-I{include_dir}", f"-L{lib_dir}",
        f"-Wl,-rpath,{lib_dir}", *pkg_flags,
    ]
    errors: list[str] = []
    for libs in (["-lnvds_meta", "-lnvdsgst_meta"], ["-lnvds_meta"], ["-lnvdsgst_meta", "-lnvds_meta"]):
        result = subprocess.run([*common, *libs], capture_output=True, text=True, check=False)
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "unknown compiler error").strip())
    raise RuntimeError("Qt heatmap bridge compile failed: " + " | ".join(errors[-2:]))


class QtHeatmapBridge:
    def __init__(self) -> None:
        self.path = ensure_qt_heatmap_bridge()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v2_qt_heatmap_configure.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_uint,
        ]
        self.lib.camera_v2_qt_heatmap_configure.restype = None
        self.lib.camera_v2_qt_heatmap_reset.argtypes = []
        self.lib.camera_v2_qt_heatmap_reset.restype = None
        self.lib.camera_v2_qt_heatmap_process.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.lib.camera_v2_qt_heatmap_process.restype = ctypes.c_int
        self.lib.camera_v2_qt_heatmap_current_count.argtypes = [ctypes.c_uint]
        self.lib.camera_v2_qt_heatmap_current_count.restype = ctypes.c_uint
        self.lib.camera_v2_qt_heatmap_updates_total.argtypes = []
        self.lib.camera_v2_qt_heatmap_updates_total.restype = ctypes.c_uint64
        self.lib.camera_v2_qt_heatmap_points_total.argtypes = []
        self.lib.camera_v2_qt_heatmap_points_total.restype = ctypes.c_uint64
        self.lib.camera_v2_qt_heatmap_points_last.argtypes = []
        self.lib.camera_v2_qt_heatmap_points_last.restype = ctypes.c_int

    def configure(self, *, deposit: float = 0.0025, decay: float = 0.99992,
                  low: float = 0.003, yellow: float = 0.070, red: float = 0.180,
                  max_points: int = 18) -> None:
        self.lib.camera_v2_qt_heatmap_configure(
            ctypes.c_float(deposit), ctypes.c_float(decay), ctypes.c_float(low),
            ctypes.c_float(yellow), ctypes.c_float(red), ctypes.c_uint(max_points),
        )

    def reset(self) -> None:
        self.lib.camera_v2_qt_heatmap_reset()

    def process(self, gst_buffer, visible: bool) -> int:
        return int(self.lib.camera_v2_qt_heatmap_process(
            ctypes.c_uint64(hash(gst_buffer)), ctypes.c_int(1 if visible else 0)
        ))

    def current_count(self, source_id: int) -> int:
        return int(self.lib.camera_v2_qt_heatmap_current_count(ctypes.c_uint(source_id)))

    def updates_total(self) -> int:
        return int(self.lib.camera_v2_qt_heatmap_updates_total())

    def points_total(self) -> int:
        return int(self.lib.camera_v2_qt_heatmap_points_total())

    def points_last(self) -> int:
        return int(self.lib.camera_v2_qt_heatmap_points_last())
