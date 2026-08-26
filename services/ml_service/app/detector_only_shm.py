from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from services.camera_service.app.shm_frame import LatestFrameMmapReader


ROOT = Path(__file__).resolve().parents[3]
INPUT_W = 672
CONTENT_H = 378
INPUT_H = 384
FRAME_BYTES = INPUT_W * INPUT_H * 3
DEFAULT_CAMERAS = "CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06"


def _absolute_without_resolving_symlink(value: str | os.PathLike[str]) -> Path:
    """Return an absolute path while preserving the final symlink.

    A Python venv commonly implements bin/python as a symlink to the base
    interpreter. Path.resolve() would dereference that link and bypass the venv's
    pyvenv.cfg/site-packages when the resolved /usr/bin/python is launched.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.absolute()


def _percentile(values: deque[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    index = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * q))))
    return float(rows[index])


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
    """Own one TRT8.6 worker and one fixed BGR letterbox SHM segment."""

    def __init__(self) -> None:
        # IMPORTANT: never .resolve() the venv Python path. On Linux it is usually
        # a symlink; dereferencing it launches /usr/bin/python directly and loses
        # the TRT8.6 venv site-packages.
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

        # -I isolates the TRT8.6 worker from PYTHONPATH/user-site contamination.
        # The worker explicitly inserts the repository root before importing the
        # local scripts package, so isolated mode does not break local imports.
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

            # Allocate the bridge SHM only after the worker handshake succeeds.
            # A broken interpreter can no longer leave a resource_tracker leak.
            self.shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
            self.frame = np.ndarray(
                (INPUT_H, INPUT_W, 3), dtype=np.uint8, buffer=self.shm.buf
            )
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
            f"python_real={self.python.resolve()} "
            "backend=trt86-sidecar-shm-v4 input=672x378+3px/3px-pad114 isolated=1",
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
            raise RuntimeError(f"bad camera SHM shape={frame378.shape}")
        if frame378.dtype != np.uint8:
            raise RuntimeError(f"bad camera SHM dtype={frame378.dtype}")
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


class DetectorOnlyShmService:
    """Detector-only ML process consuming Camera Service latest-frame mmap files.

    No RTSP, DeepStream, tracker, API or UI code runs here. The consumer never
    queues historical frames: before each inference it re-reads the latest slot and
    advances directly to the newest sequence number.
    """

    def __init__(self) -> None:
        self.root = Path(
            os.environ.get("ML_DETECTOR_SHM_DIR", "/dev/shm/ai_surveillance")
        )
        self.camera_ids = [
            row.strip()
            for row in os.environ.get("ML_DETECTOR_CAMERAS", DEFAULT_CAMERAS).split(",")
            if row.strip()
        ]
        self.conf = min(1.0, max(0.01, float(os.environ.get("ML_DETECTOR_CONF", "0.18"))))
        self.max_det = max(1, min(100, int(os.environ.get("ML_DETECTOR_MAX_DET", "20"))))
        self.max_input_age_ms = max(
            50.0, float(os.environ.get("ML_DETECTOR_MAX_INPUT_AGE_MS", "300"))
        )
        self.attach_timeout = max(
            1.0, float(os.environ.get("ML_DETECTOR_ATTACH_TIMEOUT_SEC", "30"))
        )
        self.readers: dict[str, LatestFrameMmapReader] = {}
        self.last_seq: dict[str, int] = {cid: 0 for cid in self.camera_ids}
        self.processed: dict[str, int] = {cid: 0 for cid in self.camera_ids}
        self.processed_last: dict[str, int] = {cid: 0 for cid in self.camera_ids}
        self.stale_skips: dict[str, int] = {cid: 0 for cid in self.camera_ids}
        self.box_counts: dict[str, int] = {cid: 0 for cid in self.camera_ids}
        self.infer_ms: deque[float] = deque(maxlen=240)
        self.result_age_ms: deque[float] = deque(maxlen=240)
        self.input_age_ms: deque[float] = deque(maxlen=240)
        self.started = time.monotonic()
        self.stats_at = self.started
        self.stop_requested = False
        self.detector: TRT86DetectorClient | None = None

    def _path(self, cid: str) -> Path:
        return self.root / f"{cid.lower().replace('-', '_')}.frame"

    def _attach(self) -> None:
        deadline = time.monotonic() + self.attach_timeout
        pending = set(self.camera_ids)
        while pending and time.monotonic() < deadline and not self.stop_requested:
            for cid in list(pending):
                path = self._path(cid)
                if not path.exists():
                    continue
                reader = None
                try:
                    reader = LatestFrameMmapReader(path)
                    snap = reader.latest()
                except Exception:
                    if reader is not None:
                        try:
                            reader.close()
                        except Exception:
                            pass
                    continue
                self.readers[cid] = reader
                self.last_seq[cid] = int(snap.seq) if snap is not None else 0
                pending.remove(cid)
                print(
                    f"ML_DETECTOR_ATTACH camera={cid} path={path} baseline_seq={self.last_seq[cid]}",
                    flush=True,
                )
            if pending:
                time.sleep(0.05)
        if pending:
            raise RuntimeError("camera SHM attach timeout: " + ",".join(sorted(pending)))

    def _peek_seq(self, cid: str) -> int:
        return int(self.readers[cid]._header()[0])

    def _collect_candidates(self):
        rows = []
        for cid in self.camera_ids:
            seq = self._peek_seq(cid)
            if seq < self.last_seq[cid]:
                self.last_seq[cid] = seq
                print(f"ML_DETECTOR_SOURCE_RESET camera={cid} seq={seq}", flush=True)
                continue
            if seq <= self.last_seq[cid]:
                continue
            snap = self.readers[cid].latest()
            if snap is None or snap.seq <= self.last_seq[cid]:
                continue
            rows.append((snap.captured_ns, cid, snap))
        rows.sort(key=lambda item: item[0])
        return rows

    def _latest_for_infer(self, cid: str, candidate):
        newest_seq = self._peek_seq(cid)
        if newest_seq > candidate.seq:
            newest = self.readers[cid].latest()
            if newest is not None and newest.seq >= candidate.seq:
                return newest
        return candidate

    def _print_stats(self) -> None:
        now = time.monotonic()
        elapsed = max(0.001, now - self.stats_at)
        rates = []
        for cid in self.camera_ids:
            total = self.processed[cid]
            previous = self.processed_last[cid]
            rates.append(f"{cid}:{(total - previous) / elapsed:.2f}Hz")
            self.processed_last[cid] = total
        self.stats_at = now
        print(
            "ML_DETECTOR_STATS "
            f"actual=[{' '.join(rates)}] "
            f"infer_avg={sum(self.infer_ms) / len(self.infer_ms) if self.infer_ms else 0.0:.1f}ms "
            f"infer_p95={_percentile(self.infer_ms, 0.95):.1f}ms "
            f"input_age_p95={_percentile(self.input_age_ms, 0.95):.1f}ms "
            f"result_age_p95={_percentile(self.result_age_ms, 0.95):.1f}ms "
            f"stale={sum(self.stale_skips.values())} boxes={sum(self.box_counts.values())}",
            flush=True,
        )

    def run(self) -> int:
        print(
            "ML_DETECTOR_PROFILE "
            f"source=camera-service-shm cameras={len(self.camera_ids)} "
            f"input={INPUT_W}x{CONTENT_H}x3 expected_producer=2Hz "
            f"conf={self.conf:.2f} max_det={self.max_det} "
            f"max_input_age={self.max_input_age_ms:.0f}ms",
            flush=True,
        )
        print(
            "ML_DETECTOR_BOUNDARY rtsp=0 nvdec=0 deepstream=0 tracker=0 api=0 ui=0 "
            "policy=latest-only-no-backlog",
            flush=True,
        )
        self._attach()
        self.detector = TRT86DetectorClient()

        while not self.stop_requested:
            candidates = self._collect_candidates()
            if not candidates:
                if time.monotonic() - self.stats_at >= 5.0:
                    self._print_stats()
                time.sleep(0.002)
                continue

            for _captured_ns, cid, candidate in candidates:
                if self.stop_requested:
                    break
                snap = self._latest_for_infer(cid, candidate)
                self.last_seq[cid] = int(snap.seq)
                input_age = max(
                    0.0, (time.monotonic_ns() - snap.captured_ns) / 1_000_000.0
                )
                self.input_age_ms.append(input_age)
                if input_age > self.max_input_age_ms:
                    self.stale_skips[cid] += 1
                    continue
                if snap.width != INPUT_W or snap.height != CONTENT_H or snap.stride != INPUT_W * 3:
                    raise RuntimeError(
                        f"{cid}: bad camera SHM geometry "
                        f"{snap.width}x{snap.height} stride={snap.stride}"
                    )
                frame = np.frombuffer(snap.data, dtype=np.uint8).reshape(
                    (CONTENT_H, INPUT_W, 3)
                )
                result = self.detector.infer(frame, self.conf, self.max_det)
                result_age = max(
                    0.0, (time.monotonic_ns() - snap.captured_ns) / 1_000_000.0
                )
                self.processed[cid] += 1
                self.box_counts[cid] += len(result.boxes)
                self.infer_ms.append(result.roundtrip_ms)
                self.result_age_ms.append(result_age)
                n = sum(self.processed.values())
                if n <= 3 or n % 20 == 0:
                    best = max((row[4] for row in result.boxes), default=0.0)
                    print(
                        "ML_DETECTOR_TRT "
                        f"n={n} camera={cid} seq={snap.seq} "
                        f"input_age={input_age:.1f}ms roundtrip={result.roundtrip_ms:.1f}ms "
                        f"prep={result.prep_ms:.1f}ms trt={result.trt_ms:.1f}ms "
                        f"result_age={result_age:.1f}ms boxes={len(result.boxes)} best={best:.3f}",
                        flush=True,
                    )

            if time.monotonic() - self.stats_at >= 5.0:
                self._print_stats()

        return 0

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        for reader in self.readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self.readers.clear()


def main() -> int:
    service = DetectorOnlyShmService()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
