from __future__ import annotations

"""DeepStream frame-contract adapter for sparse external RF-DETR inference.

RF-DETR intentionally runs at a few Hz per camera while NvDCF tracks every live
mux frame. DeepStream needs each frame to state that external inference has
completed, even when that particular frame has no fresh detector objects. This
module adds that missing bInferDone contract without replacing the existing
RF-DETR result injection, NvDCF configuration, tiler, OSD, or display path.
"""

import ctypes
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("native_sparse_tracker_contract.c")
BUILD_DIR = ROOT / ".runtime" / "camera_v2"
LIB_PATH = BUILD_DIR / "libcamera_v2_sparse_tracker_contract.so"


def _deepstream_root() -> Path:
    candidates = [Path("/opt/nvidia/deepstream/deepstream")]
    candidates.extend(
        sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True)
    )
    for path in candidates:
        if (path / "sources/includes/gstnvdsmeta.h").exists() and (path / "lib").exists():
            return path
    raise RuntimeError("DeepStream headers were not found under /opt/nvidia/deepstream")


def ensure_sparse_tracker_bridge() -> Path:
    """Compile the tiny bInferDone helper when missing or stale."""
    if not SOURCE.exists():
        raise RuntimeError(f"sparse tracker contract source missing: {SOURCE}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIB_PATH

    gcc = shutil.which("gcc")
    pkg = shutil.which("pkg-config")
    if not gcc or not pkg:
        raise RuntimeError("gcc and pkg-config are required for sparse NvDCF bridge")

    ds = _deepstream_root()
    include_dir = ds / "sources/includes"
    lib_dir = ds / "lib"
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
    attempts = (
        ("-lnvds_meta", "-lnvdsgst_meta"),
        ("-lnvds_meta",),
        ("-lnvdsgst_meta", "-lnvds_meta"),
    )
    errors: list[str] = []
    for libs in attempts:
        result = subprocess.run(
            [*common, *libs],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and LIB_PATH.exists():
            return LIB_PATH
        errors.append((result.stderr or result.stdout or "unknown compiler error").strip())

    raise RuntimeError(
        "sparse NvDCF bridge compile failed: " + " | ".join(errors[-2:])
    )


class _SparseTrackerMarker:
    def __init__(self) -> None:
        path = ensure_sparse_tracker_bridge()
        self.lib = ctypes.CDLL(str(path))
        self.lib.camera_v2_mark_batch_infer_done.argtypes = [ctypes.c_uint64]
        self.lib.camera_v2_mark_batch_infer_done.restype = ctypes.c_int
        self.path = path

    def mark_batch_infer_done(self, gst_buffer) -> int:
        return int(
            self.lib.camera_v2_mark_batch_infer_done(
                ctypes.c_uint64(hash(gst_buffer))
            )
        )


_MARKER: _SparseTrackerMarker | None = None


def _marker() -> _SparseTrackerMarker:
    global _MARKER
    if _MARKER is None:
        _MARKER = _SparseTrackerMarker()
    return _MARKER


def install_sparse_tracker_contract() -> Path:
    """Wrap CameraPersonTrackingFinal's mux probe with bInferDone marking.

    The wrapper runs first on every nvstreammux output buffer. The original probe
    then applies any fresh RF-DETR detections exactly as before. No detector box,
    tracker state, ID, or display metadata is synthesized by this adapter.
    """
    from .person_tracking_final import CameraPersonTrackingFinal

    marker = _marker()
    if getattr(CameraPersonTrackingFinal, "_sparse_tracker_contract_installed", False):
        return marker.path

    original_inject = CameraPersonTrackingFinal._inject_detector_probe
    original_print_stats = CameraPersonTrackingFinal._print_stats

    def _inject_with_infer_done(self, pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            marked = marker.mark_batch_infer_done(buffer)
            if marked >= 0:
                self._sparse_contract_batches = int(
                    getattr(self, "_sparse_contract_batches", 0)
                ) + 1
                self._sparse_contract_frames = int(
                    getattr(self, "_sparse_contract_frames", 0)
                ) + int(marked)
            else:
                self._sparse_contract_errors = int(
                    getattr(self, "_sparse_contract_errors", 0)
                ) + 1
        return original_inject(self, pad, info)

    def _print_stats_with_contract(self) -> bool:
        keep = original_print_stats(self)
        print(
            "CAMERA_TRACK_INPUT "
            f"infer_done_batches={int(getattr(self, '_sparse_contract_batches', 0))} "
            f"infer_done_frames={int(getattr(self, '_sparse_contract_frames', 0))} "
            f"errors={int(getattr(self, '_sparse_contract_errors', 0))} "
            "mode=sparse-external-detector",
            flush=True,
        )
        return keep

    CameraPersonTrackingFinal._inject_detector_probe = _inject_with_infer_done
    CameraPersonTrackingFinal._print_stats = _print_stats_with_contract
    CameraPersonTrackingFinal._sparse_tracker_contract_installed = True
    return marker.path
