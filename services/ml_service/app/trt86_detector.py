from __future__ import annotations

import json
import os
import select
import subprocess
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INPUT_W = 672
CONTENT_H = 378
INPUT_H = 384
FRAME_BYTES = INPUT_W * INPUT_H * 3


def _absolute_without_resolving_symlink(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.absolute()


def _read_json(proc: subprocess.Popen[str], timeout: float) -> dict:
    if proc.stdout is None:
        raise RuntimeError("TRT86 worker stdout unavailable")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"TRT86 worker timeout after {timeout:.1f}s")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"TRT86 worker closed rc={proc.poll()}")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TRT86 worker emitted non-JSON stdout: {line[:180]!r}") from exc


@dataclass
class DetectionResult:
    boxes: list[list[float]]
    prep_ms: float
    trt_ms: float
    sidecar_ms: float
    roundtrip_ms: float


class TRT86DetectorClient:
    """Own one isolated TRT8.6 worker and one fixed BGR letterbox SHM segment."""

    def __init__(self) -> None:
        self.python = _absolute_without_resolving_symlink(
            os.environ.get("ML_DETECTOR_TRT86_PYTHON", ROOT / ".venv-trt86/bin/python")
        )
        self.worker = Path(
            os.environ.get(
                "ML_DETECTOR_TRT86_WORKER",
                ROOT / "scripts/yolo26_trt86_shm_worker_v4.py",
            )
        ).resolve()
        self.engine = Path(
            os.environ.get(
                "ML_DETECTOR_TRT86_ENGINE",
                ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
            )
        ).resolve()
        for path, label in (
            (self.python, "python"),
            (self.worker, "worker"),
            (self.engine, "engine"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"TRT86 {label} missing: {path}")

        self.request_id = 0
        self.proc: subprocess.Popen[str] | None = None
        self.shm: shared_memory.SharedMemory | None = None
        self.frame: np.ndarray | None = None
        child_env = os.environ.copy()
        child_env.pop("PYTHONHOME", None)
        child_env.pop("PYTHONPATH", None)
        child_env["PYTHONNOUSERSITE"] = "1"

        try:
            self.proc = subprocess.Popen(
                [str(self.python), "-I", str(self.worker), "--engine", str(self.engine)],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                env=child_env,
            )
            ready = _read_json(self.proc, 30.0)
            if ready.get("type") != "ready":
                raise RuntimeError(f"bad TRT86 handshake: {ready}")
            if not str(ready.get("tensorrt", "")).startswith("8.6.1"):
                raise RuntimeError(f"TensorRT 8.6.1 required: {ready}")
            if tuple(ready.get("input_shape", ())) != (1, 3, INPUT_H, INPUT_W):
                raise RuntimeError(f"unexpected TRT86 input shape: {ready}")
            if tuple(ready.get("output_shape", ())) != (1, 300, 6):
                raise RuntimeError(f"unexpected TRT86 output shape: {ready}")

            self.shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
            self.frame = np.ndarray((INPUT_H, INPUT_W, 3), dtype=np.uint8, buffer=self.shm.buf)
            self.frame.fill(114)
        except BaseException:
            self._stop_proc()
            if self.shm is not None:
                try:
                    self.shm.close()
                finally:
                    try:
                        self.shm.unlink()
                    except FileNotFoundError:
                        pass
                self.shm = None
            raise

        print(
            "ML_DETECTOR_READY "
            f"engine={self.engine} worker={self.worker.name} python={self.python} "
            f"python_real={self.python.resolve()} backend=trt86-sidecar-shm-v4 "
            "input=672x378+3px/3px-pad114 isolated=1",
            flush=True,
        )

    def _stop_proc(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('{"cmd":"stop"}\n')
                proc.stdin.flush()
                proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def infer(self, frame378: np.ndarray, conf: float, max_det: int) -> DetectionResult:
        if frame378.shape != (CONTENT_H, INPUT_W, 3):
            raise RuntimeError(f"bad detector frame shape={frame378.shape}")
        if frame378.dtype != np.uint8:
            raise RuntimeError(f"bad detector frame dtype={frame378.dtype}")
        if self.frame is None or self.shm is None or self.proc is None:
            raise RuntimeError("TRT86 detector client is not ready")

        self.frame[:3, :, :] = 114
        self.frame[3:381, :, :] = frame378
        self.frame[381:, :, :] = 114
        self.request_id += 1
        req = {
            "id": self.request_id,
            "shm_name": self.shm.name,
            "conf": float(conf),
            "max_det": int(max_det),
        }
        if self.proc.stdin is None:
            raise RuntimeError("TRT86 worker stdin unavailable")
        started = time.perf_counter()
        self.proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        response = _read_json(self.proc, 5.0)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        if response.get("id") != self.request_id:
            raise RuntimeError("TRT86 response ID mismatch")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "TRT86 inference failed")))

        boxes: list[list[float]] = []
        for row in response.get("boxes", []):
            if not isinstance(row, (list, tuple)) or len(row) != 5:
                raise RuntimeError(f"invalid TRT86 detection row: {row!r}")
            boxes.append([float(v) for v in row])
        return DetectionResult(
            boxes=boxes,
            prep_ms=float(response.get("prep_ms", 0.0)),
            trt_ms=float(response.get("trt_ms", 0.0)),
            sidecar_ms=float(response.get("total_ms", 0.0)),
            roundtrip_ms=roundtrip_ms,
        )

    def close(self) -> None:
        self._stop_proc()
        self.proc = None
        if self.shm is not None:
            try:
                self.shm.close()
            finally:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
            self.shm = None
        self.frame = None
