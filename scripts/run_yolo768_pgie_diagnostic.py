#!/usr/bin/env python3
"""Run the isolated 768 YOLO26m candidate through DeepStream PGIE on direct MP4 files."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
OUT_ROOT = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r2"
MODELS = OUT_ROOT / "models"
CANDIDATE_CONFIG = OUT_ROOT / "config/config_infer_yolo26m_768_b6.txt"
FROZEN_CONFIG = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
PGIE_BINARY = ROOT / ".runtime/f5-cam05-safety-20261007-r1/secondary-context-eval/pgie-isolation-context020-r2/bin/pgie-isolation-context020-r2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_config(names_and_paths: list[tuple[str, str]]) -> str:
    rows: list[str] = []
    for name, mounted_name in names_and_paths:
        uri = base64.b64encode(f"file:///workspace/videos/{mounted_name}".encode()).decode()
        rows.extend((f"[{name}]", f"uri={uri}", "username=", "password=", "latency_ms=0",
            "drop_on_latency=false", "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    rows.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20", "interrupt_seconds=12",
        "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-a", type=Path, required=True)
    ap.add_argument("--input-b", type=Path, required=True,
        help="second source; may be the same video for a two-slot focused diagnostic")
    ap.add_argument("--output-name", required=True)
    args = ap.parse_args()
    input_a, input_b = args.input_a.resolve(), args.input_b.resolve()
    output = OUT_ROOT / args.output_name
    if not output.is_relative_to(ROOT / ".runtime") or output.exists():
        raise ValueError("output must be a new isolated directory under .runtime")
    if not all(path.is_file() for path in (input_a, input_b, MODELS / "yolo26m_raw_otm_768_b6_fp16.engine",
            MODELS / "libyolo26rawperson_768.so", CANDIDATE_CONFIG, PGIE_BINARY)):
        raise FileNotFoundError("candidate engine/parser/config or PGIE-isolation binary missing")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    configs = output / "config"
    runs = output / "run"
    inputs.mkdir(); configs.mkdir(); runs.mkdir()
    staged_inputs: list[tuple[str, str]] = []
    for idx, (name, source) in enumerate((("CAM-01", input_a), ("CAM-04", input_b))):
        mounted_name = f"source-{idx}.mp4"
        shutil.copy2(source, inputs / mounted_name)
        staged_inputs.append((name, mounted_name))
    sources = runs / "sources.ini"
    sources.write_text(camera_config(staged_inputs))
    config_text = CANDIDATE_CONFIG.read_text()
    if "infer-dims=3;768;768" not in config_text or "pre-cluster-threshold=0.25" not in config_text:
        raise RuntimeError("candidate config no longer matches the isolated 768/.25 contract")
    staged_config = configs / "config_infer_yolo26m_768_b6.txt"
    staged_config.write_text(config_text.replace("batch-size=6", "batch-size=2"))
    command = ["docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", "GST_REGISTRY=/tmp/yolo768-pgie-gst.bin", "-e", "PGIE_ISOLATION_SOURCE_COUNT=2",
        "-e", "PGIE_ISOLATION_MIN_CONFIDENCE=0.23", "-e", "PGIE_ISOLATION_MUX_BATCH=2",
        "-e", "PGIE_ISOLATION_MUX_WIDTH=1920", "-e", "PGIE_ISOLATION_MUX_HEIGHT=1080",
        "-e", "PGIE_ISOLATION_LIVE_SOURCE=0", "-e", "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US=-1",
        "-e", "PGIE_ISOLATION_MUX_MEMORY_TYPE=0", "-e", "PGIE_ISOLATION_CONFIG=/config/config_infer_yolo26m_768_b6.txt",
        "--mount", f"type=bind,src={PGIE_BINARY.parent},dst=/candidate,readonly",
        "--mount", f"type=bind,src={configs},dst=/config,readonly",
        "--mount", f"type=bind,src={MODELS},dst=/models,readonly",
        "--mount", f"type=bind,src={inputs},dst=/workspace/videos,readonly",
        "--mount", f"type=bind,src={runs},dst=/work", IMAGE,
        f"/candidate/{PGIE_BINARY.name}", "/work/sources.ini", "600"]
    provenance = {"status": "RUNNING", "inputs": [str(input_a.relative_to(ROOT)), str(input_b.relative_to(ROOT))],
        "input_sha256": [sha(input_a), sha(input_b)], "engine_sha256": sha(MODELS / "yolo26m_raw_otm_768_b6_fp16.engine"),
        "onnx_sha256": sha(MODELS / "yolo26m_raw_otm_768.onnx"),
        "parser_sha256": sha(MODELS / "libyolo26rawperson_768.so"),
        "candidate_config_sha256": sha(CANDIDATE_CONFIG), "staged_config_sha256": sha(staged_config),
        "binary_sha256": sha(PGIE_BINARY), "mux": [1920,1080], "pgie_batch": 2,
        "candidate_input": [768,768], "confidence": .25, "nms_iou": .45,
        "source_mode": "two direct MP4 files; no RTSP, tracker, identity, verifier, or external inference"}
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    start = time.time()
    with (runs / "native.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    log_text = (runs / "native.log").read_text(errors="replace")
    provenance.update({"return_code": result.returncode, "elapsed_sec": time.time() - start,
        "eos": "PGIE_ISOLATION EOS all file sources drained" in log_text, "fatal": "fatal=0" not in log_text})
    if result.returncode or not provenance["eos"] or provenance["fatal"]:
        provenance["status"] = "FAIL"
        (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        raise RuntimeError(f"DeepStream 768 PGIE diagnostic failed; inspect {runs / 'native.log'}")
    provenance["status"] = "DIAGNOSTIC_ONLY"
    provenance["frames_per_source"] = sum(1 for line in (runs / "source-frames.jsonl").read_text().splitlines() if line) // 2
    provenance["detection_rows"] = sum(1 for line in (runs / "detections.jsonl").read_text().splitlines() if line)
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    import shutil
    main()
