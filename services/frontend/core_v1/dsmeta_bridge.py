from __future__ import annotations

import ctypes
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("dsmeta_bridge.c")
BUILD_DIR = ROOT / "data/runtime"
LIBRARY = BUILD_DIR / "libdsmeta_bridge.so"


def _deepstream_root() -> Path:
    candidates = [
        Path(os.environ.get("DEEPSTREAM_ROOT", "")),
        Path("/opt/nvidia/deepstream/deepstream"),
        Path("/opt/nvidia/deepstream/deepstream-7.1"),
    ]
    candidates.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-7.1*")))
    for candidate in candidates:
        if not str(candidate):
            continue
        include = candidate / "sources/includes"
        lib = candidate / "lib"
        if include.is_dir() and lib.is_dir():
            return candidate.resolve()
    raise RuntimeError("DeepStream 7.1 root with sources/includes and lib was not found")


def _pkg_config_flags() -> list[str]:
    command = [
        "pkg-config",
        "--cflags",
        "--libs",
        "gstreamer-1.0",
        "glib-2.0",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return shlex.split(result.stdout.strip())


def _build_if_needed(ds_root: Path) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if LIBRARY.exists() and LIBRARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return

    command = [
        "gcc",
        "-O2",
        "-shared",
        "-fPIC",
        str(SOURCE),
        "-o",
        str(LIBRARY),
        f"-I{ds_root / 'sources/includes'}",
        f"-L{ds_root / 'lib'}",
        f"-Wl,-rpath,{ds_root / 'lib'}",
        "-lnvdsgst_meta",
        "-lnvds_meta",
    ]
    command.extend(_pkg_config_flags())
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "gcc failed").strip()
        raise RuntimeError(f"failed to build native DeepStream metadata bridge: {detail}")


class DeepStreamMetaBridge:
    def __init__(self):
        ds_root = _deepstream_root()
        _build_if_needed(ds_root)
        self.ds_root = ds_root
        self.library_path = LIBRARY
        self.lib = ctypes.CDLL(str(LIBRARY))

        self.lib.dsmeta_add_person_boxes.argtypes = [
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.dsmeta_add_person_boxes.restype = ctypes.c_int
        self.lib.dsmeta_count_objects.argtypes = [ctypes.c_size_t]
        self.lib.dsmeta_count_objects.restype = ctypes.c_int

    def add_person_boxes(
        self,
        gst_buffer,
        source_index: int,
        boxes: list[dict],
        source_width: int,
        source_height: int,
        detection_width: int,
        detection_height: int,
    ) -> int:
        if not boxes:
            return 0
        count = len(boxes)
        coords = (ctypes.c_float * (count * 4))()
        confs = (ctypes.c_float * count)()
        for index, item in enumerate(boxes):
            xyxy = item.get("xyxy") or [0.0, 0.0, 1.0, 1.0]
            coords[index * 4 + 0] = float(xyxy[0])
            coords[index * 4 + 1] = float(xyxy[1])
            coords[index * 4 + 2] = float(xyxy[2])
            coords[index * 4 + 3] = float(xyxy[3])
            confs[index] = float(item.get("confidence") or 0.0)
        return int(
            self.lib.dsmeta_add_person_boxes(
                ctypes.c_size_t(hash(gst_buffer)),
                int(source_index),
                coords,
                confs,
                count,
                int(source_width),
                int(source_height),
                int(detection_width),
                int(detection_height),
            )
        )

    def count_objects(self, gst_buffer) -> int:
        return int(self.lib.dsmeta_count_objects(ctypes.c_size_t(hash(gst_buffer))))
