#!/usr/bin/env python3
"""Run immutable R13 against localhost RTSP fixtures with bounded process metrics.

This diagnostic wrapper changes only the per-frame evidence environment flag
passed to the container. It does not patch, rebuild, or alter the candidate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_six_camera_v13_wall_candidate as launcher
from scripts import run_verified_candidate_synthetic_six as soak
from scripts.run_r11_reliability_soak_throttled import ContainerProcessSampler, GpuSampler
from scripts.audit_r12_finite_source_eos import audit as audit_finite_source_eos


class GpuTelemetrySampler:
    def __init__(self, period_sec: float = 1.0):
        self.period_sec = period_sec
        self.stop_event = threading.Event()
        self.samples: list[dict] = []
        self.thread = threading.Thread(target=self._run, name="gpu-telemetry-sampler", daemon=True)

    def _run(self) -> None:
        query = "memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,utilization.decoder,utilization.encoder"
        while not self.stop_event.is_set():
            now = time.monotonic()
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0,
                )
                fields = [value.strip() for value in result.stdout.strip().split(",")]
                row = {"monotonic_sec": now, "status": "PASS" if result.returncode == 0 and len(fields) == 7 else "ERROR"}
                if row["status"] == "PASS":
                    row.update({
                        "memory_used_mib": int(fields[0]), "gpu_utilization_pct": int(fields[1]),
                        "memory_utilization_pct": int(fields[2]), "power_draw_w": float(fields[3]),
                        "temperature_c": int(fields[4]), "decoder_utilization_pct": int(fields[5]),
                        "encoder_utilization_pct": int(fields[6]),
                    })
                else:
                    row["error"] = result.stderr.strip() or result.stdout.strip() or "unparseable nvidia-smi output"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row = {"monotonic_sec": now, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
            self.samples.append(row)
            self.stop_event.wait(self.period_sec)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=320)
    parser.add_argument("--fixture-outage-plan", required=True)
    parser.add_argument("--finite-eos-camera", choices=[f"CAM-{i:02d}" for i in range(1, 7)])
    parser.add_argument("--finite-eos-frames", type=int, default=100)
    args = parser.parse_args()

    if args.finite_eos_camera and args.fixture_outage_plan:
        parser.error("finite EOS and outage plans are mutually exclusive")
    if not 1 <= args.finite_eos_frames <= 10000:
        parser.error("--finite-eos-frames must be in 1..10000")

    gpu = GpuSampler()
    gpu.start_and_wait()
    soak.used_gpu_memory_mib = gpu.current_mib
    container_name = f"surv-synthetic-{args.output.name}"[:120]
    process = ContainerProcessSampler(container_name, args.output / "container-process-samples.json")
    process.start()
    gpu_metrics = GpuTelemetrySampler()
    gpu_metrics.start()

    original_command = launcher.docker_command
    original_popen = soak.subprocess.Popen

    def quiet_evidence_command(*positional, **keywords):
        previous = os.environ.get("V11_VERBOSE_EVIDENCE")
        os.environ["V11_VERBOSE_EVIDENCE"] = "0"
        try:
            return original_command(*positional, **keywords)
        finally:
            if previous is None:
                os.environ.pop("V11_VERBOSE_EVIDENCE", None)
            else:
                os.environ["V11_VERBOSE_EVIDENCE"] = previous

    launcher.docker_command = quiet_evidence_command
    finite_port = soak.PORTS[launcher.CAMERAS.index(args.finite_eos_camera)] if args.finite_eos_camera else None

    def start_fixture(*positional, **keywords):
        argv = positional[0] if positional else keywords.get("args")
        if (finite_port is not None and isinstance(argv, (list, tuple))
                and "local_rtsp_test_fixture.py" in str(argv[1])
                and str(finite_port) in [str(value) for value in argv]):
            replacement = list(argv)
            replacement[1] = str(ROOT / "scripts/local_rtsp_eos_fixture.py")
            replacement[2:] = ["--port", str(finite_port), "--frames", str(args.finite_eos_frames)]
            argv = replacement
            if positional:
                positional = (argv, *positional[1:])
            else:
                keywords["args"] = argv
        return original_popen(*positional, **keywords)

    soak.subprocess.Popen = start_fixture
    result = 1
    try:
        result = soak.run(args.candidate, args.output, args.duration, "",
                          args.fixture_outage_plan)
        if args.finite_eos_camera:
            finite_report = audit_finite_source_eos(
                args.output.resolve(), args.finite_eos_camera,
                args.finite_eos_frames / 20.0,
            )
            (args.output / "finite-eos-audit.json").write_text(
                json.dumps(finite_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0 if finite_report["scenario_status"] == "PASS" else 1
        return result
    finally:
        launcher.docker_command = original_command
        soak.subprocess.Popen = original_popen
        gpu.stop()
        process.stop()
        if args.output.exists():
            (args.output / "gpu-telemetry.json").write_text(json.dumps({
                "period_sec": gpu_metrics.period_sec,
                "sample_count": len(gpu_metrics.samples),
                "status": "PASS" if gpu_metrics.samples and all(
                    row["status"] == "PASS" for row in gpu_metrics.samples
                ) else "REVIEW_REQUIRED",
                "samples": gpu_metrics.samples,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (args.output / "gpu-monitor.json").write_text(json.dumps({
                "period_sec": gpu.period_sec,
                "sample_count": len(gpu.samples),
                "status": "PASS" if gpu.samples and all(
                    row["status"] == "PASS" for row in gpu.samples
                ) else "REVIEW_REQUIRED",
                "samples": gpu.samples,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
