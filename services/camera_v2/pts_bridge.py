from __future__ import annotations

import ctypes
import shlex
import shutil
import subprocess
from pathlib import Path

from .native_bridge import _deepstream_root

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_pts_bridge.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v2"
LIB_PATH = BUILD_DIR / "libcamera_v95_pts.so"


class _FramePtsRow(ctypes.Structure):
    _fields_ = [
        ("source_id", ctypes.c_uint32),
        ("pad_index", ctypes.c_uint32),
        ("frame_num", ctypes.c_uint64),
        ("buf_pts", ctypes.c_uint64),
    ]


def ensure_pts_bridge() -> Path:
    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for V9.5 PTS bridge")
    pkg_flags = shlex.split(
        subprocess.check_output(
            [pkg, "--cflags", "--libs", "gstreamer-1.0", "glib-2.0"],
            text=True,
        ).strip()
    )
    cmd = [
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
        "-lnvds_meta",
        "-lnvdsgst_meta",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not LIB_PATH.exists():
        raise RuntimeError(
            "V9.5 PTS bridge compile failed: "
            + (result.stderr or result.stdout or "compiler error").strip()
        )
    return LIB_PATH


class FramePtsBridge:
    def __init__(self) -> None:
        self.path = ensure_pts_bridge()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v95_copy_frame_pts.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(_FramePtsRow),
            ctypes.c_int,
        ]
        self.lib.camera_v95_copy_frame_pts.restype = ctypes.c_int

    def copy(self, gst_buffer, max_rows: int = 32) -> list[dict]:
        max_rows = max(1, min(64, int(max_rows)))
        payload = (_FramePtsRow * max_rows)()
        count = int(
            self.lib.camera_v95_copy_frame_pts(
                ctypes.c_uint64(hash(gst_buffer)),
                payload,
                ctypes.c_int(max_rows),
            )
        )
        if count <= 0:
            return []
        return [
            {
                "source_id": int(row.source_id),
                "pad_index": int(row.pad_index),
                "frame_num": int(row.frame_num),
                "buf_pts": int(row.buf_pts),
            }
            for row in payload[: min(count, max_rows)]
        ]
