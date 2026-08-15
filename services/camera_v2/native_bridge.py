from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_meta_bridge.c")
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
    if not SOURCE.exists():
        raise RuntimeError(f"metadata bridge source missing: {SOURCE}")
    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    rebuild = not LIB_PATH.exists() or LIB_PATH.stat().st_mtime < SOURCE.stat().st_mtime
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
        self.lib.camera_v2_count_tracked.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_count_tracked.restype = ctypes.c_int
        self.path = path

    def add_boxes(self, gst_buffer, source_id: int, boxes: list[tuple[float, float, float, float, float]]) -> int:
        if not boxes:
            return 0
        flat: list[float] = []
        for x1, y1, x2, y2, conf in boxes:
            flat.extend((float(x1), float(y1), float(x2), float(y2), float(conf)))
        array_type = ctypes.c_float * len(flat)
        payload = array_type(*flat)
        return int(
            self.lib.camera_v2_add_boxes(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                payload,
                ctypes.c_int(len(boxes)),
            )
        )

    def count_tracked(self, gst_buffer) -> int:
        return int(self.lib.camera_v2_count_tracked(ctypes.c_uint64(hash(gst_buffer))))
