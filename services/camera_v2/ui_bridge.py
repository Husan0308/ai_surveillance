from __future__ import annotations

import ctypes
import shlex
import shutil
import subprocess
from pathlib import Path

from .native_bridge import BUILD_DIR, _deepstream_root

SOURCE = Path(__file__).with_name("native_ui_bridge.c")
LIB_PATH = BUILD_DIR / "libcamera_v2_ui.so"


class _TrackRow(ctypes.Structure):
    _fields_ = [
        ("source_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("object_id", ctypes.c_uint64),
        ("left", ctypes.c_float),
        ("top", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("tracker_confidence", ctypes.c_float),
    ]


def ensure_ui_bridge() -> Path:
    if not SOURCE.exists():
        raise RuntimeError(f"UI metadata bridge source missing: {SOURCE}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    ds = _deepstream_root()
    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required to build the UI metadata bridge")

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

    # Keep the UI bridge as tolerant as the proven detector metadata bridge.
    # DeepStream package variants do not always expose both link names equally.
    attempts = (
        ("-lnvds_meta", "-lnvdsgst_meta"),
        ("-lnvds_meta",),
        ("-lnvdsgst_meta", "-lnvds_meta"),
    )
    errors: list[str] = []
    for libs in attempts:
        result = subprocess.run([*common, *libs], capture_output=True, text=True, check=False)
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "unknown compiler error").strip())

    raise RuntimeError("UI metadata bridge compile failed: " + " | ".join(errors[-2:]))


class NativeUIBridge:
    MAX_TRACKS = 96

    def __init__(self) -> None:
        self.path = ensure_ui_bridge()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.camera_v2_snapshot_tracks.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(_TrackRow),
            ctypes.c_int,
        ]
        self.lib.camera_v2_snapshot_tracks.restype = ctypes.c_int

    def snapshot_tracks(self, gst_buffer) -> list[dict]:
        rows = (_TrackRow * self.MAX_TRACKS)()
        count = int(
            self.lib.camera_v2_snapshot_tracks(
                ctypes.c_uint64(hash(gst_buffer)), rows, ctypes.c_int(self.MAX_TRACKS)
            )
        )
        if count <= 0:
            return []
        return [
            {
                "source_id": int(row.source_id),
                "object_id": int(row.object_id),
                "left": float(row.left),
                "top": float(row.top),
                "width": float(row.width),
                "height": float(row.height),
                "confidence": float(row.confidence),
                "tracker_confidence": float(row.tracker_confidence),
            }
            for row in rows[:count]
        ]
