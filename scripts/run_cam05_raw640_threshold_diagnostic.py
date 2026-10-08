#!/usr/bin/env python3
"""Inspect the CAM-05 missed interval with the frozen 640 YOLO engine at a diagnostic floor."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import argparse
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
ENGINE = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003/models/640/yolo26m_raw_otm_b6_fp16.engine"
PARSER = ROOT / ".runtime/detection-ab-peoplenet-yolo26m-20261002-pI5KEcrC/models/yolo26m/parser/libyolo26rawperson.so"
SOURCE_CONFIG = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
BINARY = ROOT / ".runtime/f5-cam05-safety-20261007-r1/secondary-context-eval/pgie-isolation-context020-r2/bin/pgie-isolation-context020-r2"
VIDEO = ROOT / ".runtime/cam05-supplemental-diagnostic-20261007-r1/cam05-miss-clip-h264420.mp4"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--threshold", type=float, default=.20)
    cli.add_argument("--output-name", default="raw640-threshold020-r1")
    args = cli.parse_args()
    if not 0.0 < args.threshold < .25 or not args.output_name.startswith("raw640-threshold"):
        raise ValueError("threshold must be in (0,.25) and output name must be an isolated raw640 run")
    output = ROOT / ".runtime/f5-cam05-safety-20261007-r1" / args.output_name
    if output.exists():
        raise FileExistsError(output)
    if not all(path.is_file() for path in (ENGINE, PARSER, SOURCE_CONFIG, BINARY, VIDEO)):
        raise FileNotFoundError("frozen model or exact CAM-05 diagnostic input is missing")
    output.mkdir(parents=True)
    configs = output / "config"
    runs = output / "run"
    configs.mkdir()
    runs.mkdir()
    config_text = SOURCE_CONFIG.read_text()
    if config_text.count("batch-size=6") != 1 or config_text.count("pre-cluster-threshold=0.25") != 1:
        raise RuntimeError("frozen PGIE config no longer matches the expected detector contract")
    diagnostic_config = configs / "raw640-threshold020.txt"
    threshold = f"{args.threshold:.2f}"
    diagnostic_config.write_text(config_text.replace("batch-size=6", "batch-size=2").replace(
        "pre-cluster-threshold=0.25", f"pre-cluster-threshold={threshold}"))
    uri = base64.b64encode(b"file:///workspace/input/cam05-miss-clip-h264420.mp4").decode()
    source_rows = []
    for name in ("CAM-01", "CAM-04"):
        source_rows.extend((f"[{name}]", f"uri={uri}", "username=", "password=",
            "latency_ms=0", "drop_on_latency=false", "decoder_low_latency_mode=false",
            "decoder_extra_surfaces=4", ""))
    source_rows.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
        "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    sources = runs / "sources.ini"
    sources.write_text("\n".join(source_rows))
    command = ["docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all",
        "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", "GST_REGISTRY=/tmp/cam05-raw640-gst.bin",
        "-e", "PGIE_ISOLATION_SOURCE_COUNT=2", "-e", f"PGIE_ISOLATION_MIN_CONFIDENCE={threshold}",
        "-e", "PGIE_ISOLATION_MUX_BATCH=2", "-e", "PGIE_ISOLATION_MUX_WIDTH=1920",
        "-e", "PGIE_ISOLATION_MUX_HEIGHT=1080", "-e", "PGIE_ISOLATION_LIVE_SOURCE=0",
        "-e", "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US=-1", "-e", "PGIE_ISOLATION_MUX_MEMORY_TYPE=0",
        "-e", "PGIE_ISOLATION_CONFIG=/config/raw640-threshold020.txt",
        "--mount", f"type=bind,src={BINARY.parent},dst=/candidate,readonly",
        "--mount", f"type=bind,src={configs},dst=/config,readonly",
        "--mount", f"type=bind,src={ENGINE.parent},dst=/models,readonly",
        "--mount", f"type=bind,src={PARSER},dst=/models/libyolo26rawperson.so,readonly",
        "--mount", f"type=bind,src={VIDEO.parent},dst=/workspace/input,readonly",
        "--mount", f"type=bind,src={runs},dst=/work", IMAGE,
        f"/candidate/{BINARY.name}", "/work/sources.ini", "30"]
    report = {"status": "RUNNING", "input_sha256": sha(VIDEO), "engine_sha256": sha(ENGINE),
        "parser_sha256": sha(PARSER), "frozen_pgie_config_sha256": sha(SOURCE_CONFIG),
        "diagnostic_config_sha256": sha(diagnostic_config), "binary_sha256": sha(BINARY),
        "mux": [1920, 1080], "batch": 2, "threshold": float(args.threshold),
        "sources": "same 41-frame CAM-05 clip duplicated as two direct MP4 sources; no RTSP"}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    start = time.time()
    with (runs / "native.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    log_text = (runs / "native.log").read_text(errors="replace")
    report.update({"return_code": result.returncode, "elapsed_sec": time.time() - start,
        "eos": "PGIE_ISOLATION EOS all file sources drained" in log_text,
        "fatal": "fatal=0" not in log_text})
    if result.returncode or not report["eos"] or report["fatal"]:
        report["status"] = "FAIL"
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(f"DeepStream diagnostic failed; inspect {runs / 'native.log'}")
    detections = [json.loads(line) for line in (runs / "detections.jsonl").read_text().splitlines() if line]
    frames = [json.loads(line) for line in (runs / "frames.jsonl").read_text().splitlines() if line]
    report.update({"status": "DIAGNOSTIC_ONLY", "unique_frames_per_source": len(frames) // 2,
        "detections_total": len(detections), "class0_confidence_ge_020": sum(
            int(row["class_id"]) == 0 and float(row["confidence"]) >= .20 for row in detections),
        "detections_per_frame_source0": {str(i): [row for row in detections
            if int(row["source_id"]) == 0 and int(row["frame"]) == i] for i in range(41)}})
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "detections_per_frame_source0"}, indent=2))


if __name__ == "__main__":
    main()
