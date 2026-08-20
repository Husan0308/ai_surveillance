from __future__ import annotations

import ctypes
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("native_boxes.c")
BUILD_DIR = ROOT / ".runtime" / "vision_v3"
LIB_PATH = BUILD_DIR / "libvision_v3_boxes.so"


def _deepstream_root() -> Path:
    candidates = [Path("/opt/nvidia/deepstream/deepstream")]
    candidates.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
    for path in candidates:
        if (path / "sources/includes/gstnvdsmeta.h").exists() and (path / "lib").exists():
            return path
    raise RuntimeError("DeepStream headers not found under /opt/nvidia/deepstream")


def ensure_bridge() -> Path:
    if not SOURCE.exists():
        raise RuntimeError(f"missing native bridge source: {SOURCE}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for Vision V3 metadata bridge")

    ds = _deepstream_root()
    flags = shlex.split(
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
        f"-I{ds / 'sources/includes'}",
        f"-L{ds / 'lib'}",
        f"-Wl,-rpath,{ds / 'lib'}",
        *flags,
        "-lnvds_meta",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not LIB_PATH.exists():
        detail = (result.stderr or result.stdout or "unknown compiler error").strip()
        raise RuntimeError("Vision V3 metadata bridge compile failed: " + detail)
    return LIB_PATH


class NativeBoxBridge:
    def __init__(self) -> None:
        self.path = ensure_bridge()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.vision_v3_add_person_boxes.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.vision_v3_add_person_boxes.restype = ctypes.c_int

    def add_person_boxes(self, gst_buffer, source_id: int, boxes: list[tuple[float, float, float, float, float]]) -> int:
        if not boxes:
            return 0
        flat: list[float] = []
        for x1, y1, x2, y2, conf in boxes:
            flat.extend((float(x1), float(y1), float(x2), float(y2), float(conf)))
        array_type = ctypes.c_float * len(flat)
        payload = array_type(*flat)
        pointer = ctypes.cast(payload, ctypes.POINTER(ctypes.c_float))
        return int(
            self.lib.vision_v3_add_person_boxes(
                ctypes.c_uint64(hash(gst_buffer)),
                ctypes.c_uint(int(source_id)),
                pointer,
                ctypes.c_int(len(boxes)),
            )
        )
