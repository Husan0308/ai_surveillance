#!/usr/bin/env python3
"""Run the verified r11 loopback soak with asynchronous 1 Hz GPU sampling.

The underlying candidate and acceptance checks are unchanged. This wrapper
keeps slow NVML CLI calls off the source-monitoring thread and records the
sampling cadence/results separately.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_verified_candidate_synthetic_six as base


class GpuSampler:
    def __init__(self, period_sec: float = 1.0):
        self.period_sec = period_sec
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._value: int | None = None
        self.samples: list[dict] = []
        self.thread = threading.Thread(target=self._run, name="gpu-memory-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3.0, check=True,
                )
                value = int(completed.stdout.strip().splitlines()[0])
                with self._lock:
                    self._value = value
                    self.samples.append({"monotonic_sec": started, "memory_used_mib": value,
                                         "query_sec": round(time.monotonic() - started, 4),
                                         "status": "PASS"})
            except Exception as exc:  # retain the last known sample, never block the pipeline monitor
                with self._lock:
                    self.samples.append({"monotonic_sec": started,
                                         "query_sec": round(time.monotonic() - started, 4),
                                         "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            self._stop.wait(self.period_sec)

    def start_and_wait(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._value is not None:
                    return
            time.sleep(0.05)
        raise TimeoutError("GPU memory sampler did not obtain its initial reading")

    def current_mib(self) -> int:
        with self._lock:
            if self._value is None:
                raise RuntimeError("GPU memory sample is unavailable")
            return self._value

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=4.0)


def _process_tree(root_pid: int) -> list[int]:
    pending = [root_pid]
    found: list[int] = []
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            children = []
        found.append(pid)
        pending.extend(int(child) for child in children if child.isdigit())
    return found


def _read_proc_sample(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        tail = raw[raw.rfind(")") + 2:].split()
        ppid = int(tail[1])
        cpu_ticks = int(tail[11]) + int(tail[12])
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        values = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                  for line in status if ":" in line}
        rss_kib = int(values.get("VmRSS", "0 kB").split()[0])
        threads = int(values.get("Threads", "0"))
        fd_count = len(list(Path(f"/proc/{pid}/fd").iterdir()))
        return {"pid": pid, "ppid": ppid, "cpu_ticks": cpu_ticks,
                "rss_kib": rss_kib, "threads": threads, "fd_count": fd_count}
    except (OSError, ValueError, IndexError):
        return None


class ContainerProcessSampler:
    """Sample only the isolated candidate container; never inspect camera processes."""
    def __init__(self, container_name: str, output: Path, period_sec: float = 2.0):
        self.container_name = container_name
        self.output = output
        self.period_sec = period_sec
        self._stop = threading.Event()
        self.samples: list[dict] = []
        self._previous_ticks: int | None = None
        self._previous_time: float | None = None
        self._hz = os.sysconf("SC_CLK_TCK")
        self._cpu_count = os.cpu_count() or 1
        self.thread = threading.Thread(target=self._run, name="container-proc-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            row: dict = {"monotonic_sec": now, "status": "WAITING_FOR_CONTAINER"}
            try:
                inspected = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Pid}}", self.container_name],
                    capture_output=True, text=True, timeout=2.0,
                )
                root_pid = int(inspected.stdout.strip()) if inspected.returncode == 0 else 0
                if root_pid > 0:
                    processes = [item for pid in _process_tree(root_pid)
                                 if (item := _read_proc_sample(pid)) is not None]
                    ticks = sum(item["cpu_ticks"] for item in processes)
                    if self._previous_ticks is None or self._previous_time is None:
                        core_pct = machine_pct = None
                    else:
                        elapsed = max(0.001, now - self._previous_time)
                        delta_ticks = max(0, ticks - self._previous_ticks)
                        core_pct = delta_ticks / self._hz / elapsed * 100.0
                        machine_pct = core_pct / self._cpu_count
                    self._previous_ticks, self._previous_time = ticks, now
                    gpu: dict = {"status": "ERROR"}
                    gpu_result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,utilization.decoder,utilization.encoder",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2.0,
                    )
                    if gpu_result.returncode == 0:
                        fields = [value.strip() for value in gpu_result.stdout.strip().split(",")]
                        if len(fields) == 7:
                            gpu = {"status": "PASS", "memory_used_mib": int(fields[0]),
                                   "gpu_utilization_pct": int(fields[1]),
                                   "memory_utilization_pct": int(fields[2]),
                                   "power_draw_w": float(fields[3]),
                                   "temperature_c": int(fields[4]),
                                   "decoder_utilization_pct": int(fields[5]),
                                   "encoder_utilization_pct": int(fields[6])}
                    row = {
                        "monotonic_sec": now, "status": "PASS", "container_pid": root_pid,
                        "process_count": len(processes), "cpu_core_pct": core_pct,
                        "cpu_machine_pct": machine_pct,
                        "rss_kib": sum(item["rss_kib"] for item in processes),
                        "thread_count": sum(item["threads"] for item in processes),
                        "fd_count": sum(item["fd_count"] for item in processes),
                        "gpu": gpu,
                        "processes": processes,
                    }
            except Exception as exc:
                row = {"monotonic_sec": now, "status": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}"}
            self.samples.append(row)
            self._stop.wait(self.period_sec)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=5.0)
        if not self.output.parent.is_dir():
            return
        self.output.write_text(json.dumps({
            "scope": "isolated loopback candidate container process tree",
            "period_sec": self.period_sec,
            "sample_count": len(self.samples),
            "status": "PASS" if self.samples and all(
                row["status"] in {"PASS", "WAITING_FOR_CONTAINER"} for row in self.samples) else "REVIEW_REQUIRED",
            "samples": self.samples,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=330)
    parser.add_argument("--plan", default="")
    parser.add_argument("--fixture-outage-plan", required=True)
    args = parser.parse_args()

    sampler = GpuSampler()
    sampler.start_and_wait()
    base.used_gpu_memory_mib = sampler.current_mib
    container_name = f"surv-synthetic-{args.output.name}"[:120]
    process_sampler = ContainerProcessSampler(
        container_name, args.output / "container-process-samples.json")
    process_sampler.start()
    code = 1
    try:
        code = base.run(args.candidate, args.output, args.duration, args.plan,
                        args.fixture_outage_plan)
        return code
    finally:
        sampler.stop()
        process_sampler.stop()
        if args.output.exists():
            (args.output / "gpu-monitor.json").write_text(
                json.dumps({"period_sec": sampler.period_sec,
                            "sample_count": len(sampler.samples),
                            "status": "PASS" if sampler.samples and all(
                                row["status"] == "PASS" for row in sampler.samples) else "REVIEW_REQUIRED",
                            "samples": sampler.samples}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    raise SystemExit(main())
