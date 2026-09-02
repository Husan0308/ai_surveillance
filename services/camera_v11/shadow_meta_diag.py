from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "services" / "camera_v2" / "native_shadow_diag.c"
BUILD_DIR = ROOT / ".runtime" / "camera_v11"
LIB_PATH = BUILD_DIR / "libcamera_v11_shadow_diag.so"


class _ShadowRow(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("frame_num", ctypes.c_uint32),
        ("source_id", ctypes.c_uint32),
        ("left", ctypes.c_float),
        ("top", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("age", ctypes.c_uint32),
        ("tracker_state", ctypes.c_uint32),
        ("visibility", ctypes.c_float),
    ]


def _deepstream_root() -> Path:
    env = os.getenv("DEEPSTREAM_ROOT")
    if env:
        path = Path(env)
        if (path / "sources/includes/nvds_tracker_meta.h").exists():
            return path

    candidates = [Path("/opt/nvidia/deepstream/deepstream")]
    candidates.extend(
        sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True)
    )
    for path in candidates:
        include_dir = path / "sources/includes"
        if (include_dir / "nvds_tracker_meta.h").exists() and (path / "lib").exists():
            return path
    raise RuntimeError("DeepStream tracker metadata headers were not found")


def ensure_shadow_diag() -> Path:
    if not SOURCE.exists():
        raise RuntimeError(f"shadow diagnostic source missing: {SOURCE}")

    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for shadow metadata diagnostic")

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
    raise RuntimeError("shadow metadata diagnostic compile failed: " + " | ".join(errors[-2:]))


class ShadowMetaDiag:
    STATE_NAMES = {
        0: "EMPTY",
        1: "ACTIVE",
        2: "INACTIVE",
        3: "TENTATIVE",
        4: "PROJECTED",
        5: "QUASIACTIVE",
    }

    def __init__(self) -> None:
        self.path = ensure_shadow_diag()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v2_copy_shadow_tracks.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(_ShadowRow),
            ctypes.c_int,
        ]
        self.lib.camera_v2_copy_shadow_tracks.restype = ctypes.c_int

    def copy_shadow_tracks(self, gst_buffer, max_rows: int = 128) -> list[dict]:
        max_rows = max(1, min(512, int(max_rows)))
        payload = (_ShadowRow * max_rows)()
        count = int(
            self.lib.camera_v2_copy_shadow_tracks(
                ctypes.c_uint64(hash(gst_buffer)),
                payload,
                ctypes.c_int(max_rows),
            )
        )
        if count < 0:
            raise RuntimeError(f"copy shadow tracks returned {count}")
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
                "age": int(row.age),
                "tracker_state": int(row.tracker_state),
                "state": self.STATE_NAMES.get(int(row.tracker_state), f"STATE_{int(row.tracker_state)}"),
                "visibility": float(row.visibility),
            }
            for row in payload[: min(count, max_rows)]
        ]
