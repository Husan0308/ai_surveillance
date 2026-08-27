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

ROOT = Path(__file__).resolve().parents[2]
INPUT_W = 672
INPUT_H = 384
FRAME_BYTES = INPUT_W * INPUT_H * 3


def _absolute(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.absolute()


def _read_json(proc: subprocess.Popen[str], timeout: float) -> dict:
    if proc.stdout is None:
        raise RuntimeError("V11 TRT86 b1 worker stdout unavailable")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"V11 TRT86 b1 worker timeout after {timeout:.1f}s")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"V11 TRT86 b1 worker closed rc={proc.poll()}")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"V11 TRT86 b1 worker emitted non-JSON: {line[:180]!r}") from exc


@dataclass(frozen=True)
class Batch1Result:
    boxes: list[list[float]]
    shm_copy_ms: float
    prep_ms: float
    trt_ms: float
    sidecar_ms: float
    roundtrip_ms: float


class Batch1TRT86Client:
    """Isolated TensorRT 8.6 batch-1 client for latency-first V11 detection."""

    def __init__(self) -> None:
        self.python = _absolute(os.environ.get("V11_TRT86_PYTHON", ROOT / ".venv-trt86/bin/python"))
        self.worker = _absolute(
            os.environ.get("V11_TRT86_B1_WORKER", ROOT / "scripts/yolo26_trt86_shm_worker.py")
        )
        self.engine = _absolute(
            os.environ.get(
                "V11_TRT86_B1_ENGINE",
                ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
            )
        )
        for path, label in ((self.python, "python"), (self.worker, "worker"), (self.engine, "batch1 engine")):
            if not path.is_file():
                raise FileNotFoundError(f"V11 TRT86 {label} missing: {path}")

        self.proc: subprocess.Popen[str] | None = None
        self.shm: shared_memory.SharedMemory | None = None
        self.frame: np.ndarray | None = None
        self.request_id = 0
        child_env = os.environ.copy()
        child_env.pop("PYTHONHOME", None)
        child_env.pop("PYTHONPATH", None)
        child_env["PYTHONNOUSERSITE"] = "1"

        try:
            self.shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
            self.frame = np.ndarray((INPUT_H, INPUT_W, 3), dtype=np.uint8, buffer=self.shm.buf)
            self.frame.fill(114)
            self.proc = subprocess.Popen(
                [str(self.python), str(self.worker), "--engine", str(self.engine)],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                env=child_env,
            )
            ready = _read_json(self.proc, 60.0)
            if ready.get("type") != "ready":
                raise RuntimeError(f"bad V11 TRT86 b1 handshake: {ready}")
            if not str(ready.get("tensorrt", "")).startswith("8.6.1"):
                raise RuntimeError(f"TensorRT 8.6.1 required: {ready}")
            if tuple(ready.get("input_shape", ())) != (1, 3, 384, 672):
                raise RuntimeError(f"unexpected V11 b1 input: {ready}")
            if tuple(ready.get("output_shape", ())) != (1, 300, 6):
                raise RuntimeError(f"unexpected V11 b1 output: {ready}")
        except BaseException:
            self.close()
            raise

    def infer(self, camera_id: str, frame: np.ndarray, *, conf: float, max_det: int) -> Batch1Result:
        if self.proc is None or self.proc.stdin is None or self.shm is None or self.frame is None:
            raise RuntimeError("V11 TRT86 b1 client is not ready")
        if frame.shape != (INPUT_H, INPUT_W, 3) or frame.dtype != np.uint8:
            raise RuntimeError(
                f"{camera_id}: expected uint8 {INPUT_H}x{INPUT_W}x3, got {frame.shape}/{frame.dtype}"
            )

        copy_started = time.perf_counter()
        np.copyto(self.frame, frame, casting="no")
        shm_copy_ms = (time.perf_counter() - copy_started) * 1000.0

        self.request_id += 1
        request = {
            "id": self.request_id,
            "shm_name": self.shm.name,
            "conf": float(conf),
            "max_det": int(max_det),
        }
        started = time.perf_counter()
        self.proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        response = _read_json(self.proc, 5.0)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        if response.get("id") != self.request_id:
            raise RuntimeError("V11 TRT86 b1 response ID mismatch")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "V11 TRT86 b1 inference failed")))

        clean: list[list[float]] = []
        for row in response.get("boxes") or []:
            if not isinstance(row, (list, tuple)) or len(row) != 5:
                raise RuntimeError(f"{camera_id}: invalid detection row {row!r}")
            clean.append([float(v) for v in row])

        return Batch1Result(
            boxes=clean,
            shm_copy_ms=shm_copy_ms,
            prep_ms=float(response.get("prep_ms", 0.0)),
            trt_ms=float(response.get("trt_ms", 0.0)),
            sidecar_ms=float(response.get("total_ms", roundtrip_ms)),
            roundtrip_ms=roundtrip_ms,
        )

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None:
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
        self.frame = None
        shm = self.shm
        self.shm = None
        if shm is not None:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
