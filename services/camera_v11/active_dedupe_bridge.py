from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("active_dedupe_bridge.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v11"
LIB_PATH = BUILD_DIR / "libcamera_v11_active_dedupe.so"


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
    if not SOURCE.exists():
        raise RuntimeError(f"active dedupe bridge source missing: {SOURCE}")

    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for active dedupe bridge")

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
    errors: list[str] = []
    for libs in (
        ["-lnvds_meta", "-lnvdsgst_meta"],
        ["-lnvds_meta"],
        ["-lnvdsgst_meta", "-lnvds_meta"],
    ):
        result = subprocess.run(
            [*common, *libs], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "compiler error").strip())
    raise RuntimeError("active dedupe bridge compile failed: " + " | ".join(errors[-2:]))


class ActiveDedupeBridge:
    def __init__(self) -> None:
        self.path = ensure_bridge()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v11_hide_track_ids.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
        ]
        self.lib.camera_v11_hide_track_ids.restype = ctypes.c_int

    def hide_track_ids(self, gst_buffer, source_id: int, ids) -> int:
        values = sorted({int(value) for value in ids if int(value) >= 0})
        if not values:
            return 0
        payload = (ctypes.c_uint64 * len(values))(*values)
        return int(
            self.lib.camera_v11_hide_track_ids(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                payload,
                ctypes.c_int(len(values)),
            )
        )
