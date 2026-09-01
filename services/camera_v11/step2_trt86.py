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
CONTENT_H = 378
INPUT_H = 384
FRAME_BYTES = INPUT_W * INPUT_H * 3


def _absolute(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.absolute()


def _read_json(process: subprocess.Popen[str], timeout: float) -> dict:
    if process.stdout is None:
        raise RuntimeError("TRT86 worker stdout unavailable")
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"TRT86 worker response timeout after {timeout:.1f}s")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"TRT86 worker exited rc={process.poll()}")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TRT86 worker emitted invalid JSON: {line[:160]!r}") from exc


@dataclass(frozen=True)
class TRTResult:
    boxes: list[list[float]]
    stages: dict[str, float]
    roundtrip_ms: float


class Step2TRT86Client:
    """One isolated TRT8.6 worker and one persistent latest-frame SHM slot."""

    def __init__(self) -> None:
        self.python = _absolute(
            os.environ.get("V11_STEP2_TRT86_PYTHON", ROOT / ".venv-trt86/bin/python")
        )
        self.worker = _absolute(
            os.environ.get(
                "V11_STEP2_TRT86_WORKER",
                ROOT / "scripts/yolo26_trt86_step2_worker.py",
            )
        )
        self.engine = _absolute(
            os.environ.get(
                "V11_STEP2_ENGINE",
                ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
            )
        )
        for path, label in (
            (self.python, "TRT86 Python"),
            (self.worker, "TRT86 worker"),
            (self.engine, "FP32 engine"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} missing: {path}")

        self.process: subprocess.Popen[str] | None = None
        self.shm: shared_memory.SharedMemory | None = None
        self.frame: np.ndarray | None = None
        self.content: np.ndarray | None = None
        self.request_id = 0
        child_env = os.environ.copy()
        child_env.pop("PYTHONHOME", None)
        child_env.pop("PYTHONPATH", None)
        child_env["PYTHONNOUSERSITE"] = "1"

        try:
            self.process = subprocess.Popen(
                [str(self.python), "-I", str(self.worker), "--engine", str(self.engine)],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                env=child_env,
                # The application owns terminal signals and performs the worker's
                # ordered JSON shutdown; do not deliver Ctrl+C to both processes.
                start_new_session=True,
            )
            ready = _read_json(self.process, 30.0)
            if ready.get("type") != "ready":
                raise RuntimeError(f"bad TRT86 worker handshake: {ready}")
            if not str(ready.get("tensorrt", "")).startswith("8.6.1"):
                raise RuntimeError(f"TensorRT 8.6.1 required: {ready}")
            if tuple(ready.get("input_shape", ())) != (1, 3, INPUT_H, INPUT_W):
                raise RuntimeError(f"unexpected TRT86 input shape: {ready}")
            if tuple(ready.get("output_shape", ())) != (1, 300, 6):
                raise RuntimeError(f"unexpected TRT86 output shape: {ready}")

            self.shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
            self.frame = np.ndarray((INPUT_H, INPUT_W, 3), dtype=np.uint8, buffer=self.shm.buf)
            self.frame.fill(114)
            self.content = self.frame[3:381]
            print(
                "CAMERA_V11_STEP2_TRT_READY "
                f"engine={self.engine} precision=fp32 batch=1 isolated=1 "
                f"transport={ready.get('transport')} stream={ready.get('stream')} "
                f"priority={ready.get('priority_least')}/range={ready.get('priority_greatest')}..{ready.get('priority_least')}",
                flush=True,
            )
        except BaseException:
            self.close()
            raise

    def infer_preloaded(self, conf: float, max_det: int) -> TRTResult:
        if self.process is None or self.process.stdin is None or self.shm is None:
            raise RuntimeError("TRT86 worker is not ready")
        self.request_id += 1
        request = {
            "id": self.request_id,
            "shm_name": self.shm.name,
            "conf": float(conf),
            "max_det": int(max_det),
        }
        started = time.perf_counter()
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = _read_json(self.process, 5.0)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        if response.get("id") != self.request_id:
            raise RuntimeError("TRT86 worker response ID mismatch")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "TRT86 inference failed")))
        boxes = response.get("boxes") or []
        clean: list[list[float]] = []
        for row in boxes:
            if not isinstance(row, list) or len(row) != 5:
                raise RuntimeError(f"invalid TRT86 detection row: {row!r}")
            clean.append([float(value) for value in row])
        raw_stages = response.get("stages") or {}
        stages = {str(key): float(value) for key, value in raw_stages.items()}
        return TRTResult(boxes=clean, stages=stages, roundtrip_ms=roundtrip_ms)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write('{"cmd":"stop"}\n')
                    process.stdin.flush()
                    process.wait(timeout=2.0)
            except Exception:
                try:
                    process.terminate()
                    process.wait(timeout=1.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        self.content = None
        self.frame = None
        if self.shm is not None:
            try:
                self.shm.close()
            finally:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
            self.shm = None
