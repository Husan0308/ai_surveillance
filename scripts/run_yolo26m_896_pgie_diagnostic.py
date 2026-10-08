#!/usr/bin/env python3
"""Run isolated 896x896 YOLO26m PGIE-only inference on two direct MP4 sources."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
ROOT_CANDIDATE = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo896-candidate-r1"
PGIE_BINARY = ROOT / ".runtime/f5-cam05-safety-20261007-r1/secondary-context-eval/pgie-isolation-context020-r2/bin/pgie-isolation-context020-r2"
SIDE = 896


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_config(name_paths: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for name, mounted in name_paths:
        uri = base64.b64encode(f"file:///workspace/videos/{mounted}".encode()).decode()
        lines.extend((f"[{name}]", f"uri={uri}", "username=", "password=", "latency_ms=0",
            "drop_on_latency=false", "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20", "interrupt_seconds=12",
        "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-a", type=Path, required=True)
    ap.add_argument("--input-b", type=Path, required=True)
    ap.add_argument("--output-name", required=True)
    args = ap.parse_args()
    input_a, input_b = args.input_a.resolve(), args.input_b.resolve()
    if not ROOT_CANDIDATE.is_dir() or not all(p.is_file() for p in (input_a, input_b, PGIE_BINARY)):
        raise FileNotFoundError("candidate, source files, or PGIE isolation binary missing")
    model_dir, config_path = ROOT_CANDIDATE / "models", ROOT_CANDIDATE / "config/config_infer_yolo26m_896_b6.txt"
    required = (model_dir / "yolo26m_raw_otm_896_b6_fp16.engine", model_dir / "libyolo26rawperson_896.so",
        model_dir / "libyolo26rawperson.so", config_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("896 candidate engine/parser/config is incomplete")
    out = ROOT_CANDIDATE / args.output_name
    if not out.is_relative_to(ROOT / ".runtime") or out.exists():
        raise ValueError("output must be a new isolated directory under .runtime")
    out.mkdir()
    inputs, cfgdir, run = out / "inputs", out / "config", out / "run"
    inputs.mkdir(); cfgdir.mkdir(); run.mkdir()
    staged: list[tuple[str, str]] = []
    for idx, (name, source) in enumerate((("CAM-01", input_a), ("CAM-04", input_b))):
        mounted = f"source-{idx}.mp4"
        shutil.copy2(source, inputs / mounted)
        staged.append((name, mounted))
    source_ini = run / "sources.ini"
    source_ini.write_text(source_config(staged))
    cfg = config_path.read_text()
    if "infer-dims=3;896;896" not in cfg or "pre-cluster-threshold=0.25" not in cfg:
        raise RuntimeError("candidate config changed from the 896/.25 contract")
    staged_config = cfgdir / config_path.name
    staged_config.write_text(cfg.replace("batch-size=6", "batch-size=2"))
    command = ["docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", "GST_REGISTRY=/tmp/yolo896-pgie-gst.bin", "-e", "PGIE_ISOLATION_SOURCE_COUNT=2",
        "-e", "PGIE_ISOLATION_MIN_CONFIDENCE=0.23", "-e", "PGIE_ISOLATION_MUX_BATCH=2",
        "-e", "PGIE_ISOLATION_MUX_WIDTH=1920", "-e", "PGIE_ISOLATION_MUX_HEIGHT=1080",
        "-e", "PGIE_ISOLATION_LIVE_SOURCE=0", "-e", "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US=-1",
        "-e", "PGIE_ISOLATION_MUX_MEMORY_TYPE=0", "-e", "PGIE_ISOLATION_CONFIG=/config/config_infer_yolo26m_896_b6.txt",
        "--mount", f"type=bind,src={PGIE_BINARY.parent},dst=/candidate,readonly",
        "--mount", f"type=bind,src={cfgdir},dst=/config,readonly",
        "--mount", f"type=bind,src={model_dir},dst=/models,readonly",
        "--mount", f"type=bind,src={inputs},dst=/workspace/videos,readonly",
        "--mount", f"type=bind,src={run},dst=/work", IMAGE,
        f"/candidate/{PGIE_BINARY.name}", "/work/sources.ini", "600"]
    manifest = {"status": "RUNNING", "inputs": [str(input_a.relative_to(ROOT)), str(input_b.relative_to(ROOT))],
        "input_sha256": [sha(input_a), sha(input_b)], "engine_sha256": sha(required[0]),
        "onnx_sha256": sha(model_dir / "yolo26m_raw_otm_896.onnx"), "parser_sha256": sha(required[1]),
        "candidate_config_sha256": sha(config_path), "staged_config_sha256": sha(staged_config),
        "pgie_binary_sha256": sha(PGIE_BINARY), "mux": [1920, 1080], "pgie_batch": 2,
        "input_resolution": [SIDE, SIDE], "confidence": 0.25, "nms_iou": 0.45,
        "source_mode": "two direct MP4 files; no RTSP, tracker, identity, verifier, or external inference"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.time()
    with (run / "native.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    log_text = (run / "native.log").read_text(errors="replace")
    manifest.update({"return_code": result.returncode, "elapsed_sec": time.time() - started,
        "eos": "PGIE_ISOLATION EOS all file sources drained" in log_text, "fatal": "fatal=0" not in log_text})
    if result.returncode or not manifest["eos"] or manifest["fatal"]:
        manifest["status"] = "FAIL"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise RuntimeError(f"DeepStream 896 diagnostic failed; inspect {run / 'native.log'}")
    manifest["status"] = "DIAGNOSTIC_ONLY"
    frame_lines = [line for line in (run / "source-frames.jsonl").read_text().splitlines() if line]
    manifest["frames_per_source"] = len(frame_lines) // 2
    manifest["detection_rows"] = sum(1 for line in (run / "detections.jsonl").read_text().splitlines() if line)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
