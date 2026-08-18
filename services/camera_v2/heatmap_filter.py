from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_heatmap_filter.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v2"
LIB_PATH = BUILD_DIR / "libcamera_v2_heatmap_filter.so"


def _deepstream_root() -> Path:
    env = os.getenv("DEEPSTREAM_ROOT", "").strip()
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


def ensure_heatmap_filter() -> Path:
    if not SOURCE.exists():
        raise RuntimeError(f"heatmap filter source missing: {SOURCE}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for the heatmap filter")

    ds = _deepstream_root()
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
        f"-I{ds / 'sources/includes'}",
        f"-L{ds / 'lib'}",
        f"-Wl,-rpath,{ds / 'lib'}",
        *pkg_flags,
    ]

    errors: list[str] = []
    for libs in (["-lnvds_meta", "-lnvdsgst_meta"], ["-lnvds_meta"]):
        result = subprocess.run(
            [*common, *libs], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "compile failed").strip())

    raise RuntimeError("heatmap filter compile failed: " + " | ".join(errors[-2:]))


class NativeHeatmapFilter:
    def __init__(self) -> None:
        self.path = ensure_heatmap_filter()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v2_heatmap_filter.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint32,
        ]
        self.lib.camera_v2_heatmap_filter.restype = ctypes.c_int

    def apply(
        self,
        gst_buffer,
        *,
        wall_width: int,
        wall_height: int,
        rows: int,
        columns: int,
        enabled_mask: int,
    ) -> int:
        return int(
            self.lib.camera_v2_heatmap_filter(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(wall_width)),
                ctypes.c_uint(int(wall_height)),
                ctypes.c_uint(int(rows)),
                ctypes.c_uint(int(columns)),
                ctypes.c_uint32(int(enabled_mask) & 0xFFFFFFFF),
            )
        )
