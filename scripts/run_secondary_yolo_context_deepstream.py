#!/usr/bin/env python3
"""Run prepared crop candidates through the pinned DeepStream YOLO nvinfer stack."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
PGIE_ROOT = ROOT / ".runtime/pgie-input-isolation-20261005"
BINARY = PGIE_ROOT / "bin/pgie-isolation"
MODEL_DIR = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003/models/640"
PARSER = ROOT / ".runtime/detection-ab-peoplenet-yolo26m-20261002-pI5KEcrC/models/yolo26m/parser/libyolo26rawperson.so"
FROZEN_CONFIG = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def camera_config(video_uri: str | list[str]) -> str:
    rows = []
    uris = [video_uri, video_uri] if isinstance(video_uri, str) else video_uri
    if len(uris) not in {2, 6}:
        raise ValueError("diagnostic input must contain two or six streams")
    for index, item_uri in enumerate(uris, 1):
        name = f"CAM-{index:02d}"
        uri = base64.b64encode(item_uri.encode()).decode()
        rows.extend((f"[{name}]", f"uri={uri}", "username=", "password=",
            "latency_ms=0", "drop_on_latency=false", "decoder_low_latency_mode=false",
            "decoder_extra_surfaces=4", ""))
    rows.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
        "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="isolated model directory containing engine, labels, and parser")
    parser.add_argument("--config-template", type=Path, default=FROZEN_CONFIG,
                        help="nvinfer config template for this verifier-only diagnostic")
    parser.add_argument("--input-side", type=int, default=640,
                        help="square verifier network input dimension")
    parser.add_argument("--stream-pair", type=str, default="",
                        help="replay two exact streams from a six-stream candidate map, e.g. 0,1")
    parser.add_argument("--stream-group", type=str, default="",
                        help="replay all six exact streams as one batch-6 crop diagnostic, e.g. 0,1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, choices=(2, 6), default=2,
                        help="verifier and mux batch size; six requires all six mapped streams")
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    binary = args.binary.resolve()
    model_dir = args.model_dir.resolve()
    config_template = args.config_template.resolve()
    if (not dataset.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime")
            or not model_dir.is_relative_to(ROOT / ".runtime")):
        raise ValueError("dataset/output must remain under .runtime")
    if args.input_side < 32 or args.input_side % 32:
        raise ValueError("input-side must be a positive multiple of 32")
    if output.exists():
        raise FileExistsError(output)
    if not binary.is_file() or not config_template.is_file():
        raise FileNotFoundError("pinned pgie-isolation binary or verifier config is missing")
    if args.stream_pair and args.stream_group:
        raise ValueError("choose stream-pair or stream-group, not both")
    source_count = args.batch_size
    selected_streams = args.stream_group or args.stream_pair
    if selected_streams:
        try:
            stream_ids = [int(value) for value in selected_streams.split(",")]
        except ValueError as exc:
            raise ValueError("stream selection must contain numeric indices") from exc
        if (len(stream_ids) != source_count or len(set(stream_ids)) != source_count
                or any(value not in range(6) for value in stream_ids)):
            raise ValueError(f"batch-{source_count} diagnostic requires {source_count} distinct stream indices in 0..5")
        sequence_path = dataset / "sequence-map.json"
        if not sequence_path.is_file():
            raise FileNotFoundError("six-stream dataset sequence-map.json is missing")
        original_maps = json.loads(sequence_path.read_text())
        stream_sources = [dataset / "videos" / f"stream-{value:02d}.mp4" for value in stream_ids]
        if any(not source.is_file() for source in stream_sources):
            raise FileNotFoundError("a selected exact candidate crop stream is missing")
        maps = []
        for item in original_maps:
            if item.get("filler") or int(item["stream_id"]) not in stream_ids:
                continue
            expected = [float(value) for value in item["crop_expected_bbox"]]
            maps.append({**item, "source_id": stream_ids.index(int(item["stream_id"])),
                "crop_index": int(item["stream_frame"]), "expected_bbox_in_crop": expected,
                "expected_center_in_crop": [expected[0] + expected[2] / 2, expected[1] + expected[3] / 2],
                "kind": item.get("candidate_kind", "LOWCONF")})
        video_uris = [f"file:///workspace/inputs/videos/{source.name}" for source in stream_sources]
        video_mount = dataset / "videos"
        probe_paths = stream_sources
        source_count = len(stream_ids)
    else:
        video = dataset / "context-crops.mp4"
        map_file = dataset / "candidate-map.jsonl"
        if not video.is_file() or not map_file.is_file():
            raise FileNotFoundError("dataset video/map is missing")
        maps = read_rows(map_file)
        video_uris = "file:///workspace/inputs/videos/context-crops.mp4"
        video_mount = dataset
        probe_paths = [video]
    if not maps:
        raise RuntimeError("candidate map is empty")
    probes = []
    for source in probe_paths:
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(source)], text=True))
        probes.append(probe["streams"][0])
    for stream in probes:
        if (stream["codec_name"], int(stream["width"]), int(stream["height"]), stream["avg_frame_rate"]) != ("h264", 640, 640, "20/1"):
            raise RuntimeError(f"crop video contract mismatch: {stream}")
    frames_per_source = int(probes[0]["nb_frames"])
    if any(int(stream["nb_frames"]) != frames_per_source for stream in probes):
        raise RuntimeError("paired crop streams must have equal frame counts")
    if any(int(row["crop_index"]) >= frames_per_source for row in maps):
        raise RuntimeError("candidate map references a frame outside a crop stream")
    output.mkdir(parents=True)
    config_text = config_template.read_text()
    if config_template == FROZEN_CONFIG:
        config_text = config_text.replace("batch-size=6", "batch-size=2").replace(
            "pre-cluster-threshold=0.25", "pre-cluster-threshold=0.20")
        if FROZEN_CONFIG.read_text().count("pre-cluster-threshold=0.25") != 1:
            raise RuntimeError("frozen config threshold contract changed")
    required = (f"batch-size={source_count}", f"infer-dims=3;{args.input_side};{args.input_side}",
        "pre-cluster-threshold=0.20", "nms-iou-threshold=0.45", "maintain-aspect-ratio=1",
        "symmetric-padding=1", "model-engine-file=/models/", "custom-lib-path=/models/libyolo26rawperson.so")
    if any(field not in config_text for field in required):
        raise RuntimeError("verifier config violates batch/input/threshold/preprocessing contract")
    engine_rel = config_text.split("model-engine-file=/models/", 1)[1].splitlines()[0]
    engine_path = model_dir / engine_rel
    parser_path = model_dir / "libyolo26rawperson.so"
    if not engine_path.is_file() or not parser_path.is_file():
        raise FileNotFoundError("verifier engine or parser missing from isolated model directory")
    config_dir = output / "config"
    config_dir.mkdir()
    config = config_dir / f"context-{args.input_side}-batch{source_count}.txt"
    config.write_text(config_text)
    run = output / "run"
    run.mkdir()
    sources = run / "sources.ini"
    sources.write_text(camera_config(video_uris))
    cmd = ["docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", "GST_REGISTRY=/tmp/secondary-context-gst.bin", "-e", f"PGIE_ISOLATION_SOURCE_COUNT={source_count}",
        # nvinfer's configured pre-cluster threshold remains .20. This audit
        # floor is only an output-validity guard; setting it below the model
        # cutoff avoids rejecting a float16-rounded .199x confidence emitted
        # at the exact configured boundary.
        "-e", "PGIE_ISOLATION_MIN_CONFIDENCE=0.0",
        "-e", f"PGIE_ISOLATION_MUX_BATCH={source_count}", "-e", "PGIE_ISOLATION_MUX_WIDTH=640",
        "-e", "PGIE_ISOLATION_MUX_HEIGHT=640", "-e", "PGIE_ISOLATION_LIVE_SOURCE=0",
        "-e", "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US=-1", "-e", "PGIE_ISOLATION_MUX_MEMORY_TYPE=0",
        "-e", f"PGIE_ISOLATION_CONFIG=/config/{config.name}",
        "--mount", f"type=bind,src={binary.parent},dst=/candidate,readonly",
        "--mount", f"type=bind,src={config_dir},dst=/config,readonly",
        "--mount", f"type=bind,src={model_dir},dst=/models,readonly",
        "--mount", f"type=bind,src={video_mount},dst=/workspace/inputs/videos,readonly",
        "--mount", f"type=bind,src={run},dst=/work",
        IMAGE, f"/candidate/{binary.name}", "/work/sources.ini", "180"]
    launch = {"argv_redacted": cmd[:cmd.index(IMAGE)] + [IMAGE, "<pgie-isolation>", "<sources.ini>", "180"],
        "binary": str(binary.relative_to(ROOT)), "binary_sha256": sha(binary), "engine_sha256": sha(engine_path),
        "parser_sha256": sha(parser_path), "config_template_sha256": sha(config_template),
        "diagnostic_config_sha256": sha(config), "nvinfer_input": [args.input_side, args.input_side], "mux": [640, 640, source_count],
        "nvinfer_threshold": 0.20, "audit_output_floor": 0.0, "nms_iou": 0.45,
        "source_mode": "direct MP4 crop readers; no RTSP",
        "frames_expected_per_source": frames_per_source, "crop_candidates": len(maps),
        "source_count": source_count,
        "inference": "DeepStream 9.1 nvinfer TensorRT YOLO26m only; crop pixels were prepared offline, not inferred by Python/OpenCV"}
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    started = time.time()
    with (run / "native.log").open("x") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                rc = proc.wait(timeout=10)
            if rc != 0:
                raise RuntimeError(f"crop inference timed out/failed: {run / 'native.log'}")
    log_text = (run / "native.log").read_text(errors="replace")
    if rc != 0 or "PGIE_ISOLATION EOS all file sources drained" not in log_text or "fatal=0" not in log_text:
        raise RuntimeError(f"DeepStream crop inference failed rc={rc}; see {run / 'native.log'}")
    detections = read_rows(run / "detections.jsonl")
    by_frame: dict[tuple[int, int], list[dict]] = {}
    for row in detections:
        if int(row["class_id"]) == 0 and float(row["confidence"]) >= .20:
            by_frame.setdefault((int(row["source_id"]), int(row["frame"])), []).append(row)
    evaluated = []
    for item in maps:
        frame_num = int(item["crop_index"])
        source_id = int(item.get("source_id", 0))
        ex, ey = map(float, item["expected_center_in_crop"])
        containing = []
        for detection in by_frame.get((source_id, frame_num), []):
            x, y, w, h = map(float, detection["box"])
            expected = item.get("expected_bbox_in_crop")
            if expected:
                ax, ay, aw, ah = map(float, expected)
                iw = max(0.0, min(ax + aw, x + w) - max(ax, x))
                ih = max(0.0, min(ay + ah, y + h) - max(ay, y))
                intersection = iw * ih
                union = aw * ah + w * h - intersection
                overlap = intersection / union if union > 0 else 0.0
                distance = ((ax + aw / 2 - x - w / 2) ** 2 + (ay + ah / 2 - y - h / 2) ** 2) ** .5
                width_ratio = max(aw, w) / max(1.0, min(aw, w))
                height_ratio = max(ah, h) / max(1.0, min(ah, h))
                if overlap >= .30 and distance <= 50 and width_ratio <= 2 and height_ratio <= 2:
                    containing.append({**detection, "strict_iou": overlap, "strict_center_distance": distance,
                        "strict_width_ratio": width_ratio, "strict_height_ratio": height_ratio})
            elif x <= ex <= x + w and y <= ey <= y + h:
                containing.append(detection)
        status = "CONFIRMED" if len(containing) == 1 else ("AMBIGUOUS" if containing else "NO_LOCAL_PERSON")
        evaluated.append({**item, "verifier_status": status,
            "matching_proposals": [{"confidence": float(x["confidence"]), "bbox": x["box"],
                "iou": x.get("strict_iou"), "center_distance_px": x.get("strict_center_distance"),
                "width_ratio": x.get("strict_width_ratio"), "height_ratio": x.get("strict_height_ratio")} for x in containing],
            "all_context_proposals": by_frame.get((source_id, frame_num), [])})
    positives = [row for row in evaluated if row["kind"].startswith("CAM05_")
                  and row["kind"].endswith("_SUPPLEMENTAL")]
    negatives = [row for row in evaluated if row["kind"] == "REVIEWED_CARDBOARD_640_020"]
    report = {"status": "DIAGNOSTIC_ONLY", "return_code": rc, "elapsed_sec": time.time() - started,
        "source_eos": True, "frames_per_source": frames_per_source, "source_count": source_count,
        "detections": len(detections),
        "candidate_status_by_camera": {
            camera: {status: sum(1 for row in evaluated if row.get("camera_id") == camera and row["verifier_status"] == status)
                     for status in ("CONFIRMED", "AMBIGUOUS", "NO_LOCAL_PERSON")}
            for camera in sorted({str(row.get("camera_id", "unknown")) for row in evaluated})},
        "cam05_seated_positive": {"count": len(positives),
            "confirmed": sum(row["verifier_status"] == "CONFIRMED" for row in positives),
            "ambiguous": sum(row["verifier_status"] == "AMBIGUOUS" for row in positives),
            "rejected": sum(row["verifier_status"] == "NO_LOCAL_PERSON" for row in positives)},
        "reviewed_cardboard_negative": {"count": len(negatives),
            "confirmed": sum(row["verifier_status"] == "CONFIRMED" for row in negatives),
            "ambiguous": sum(row["verifier_status"] == "AMBIGUOUS" for row in negatives),
            "rejected": sum(row["verifier_status"] == "NO_LOCAL_PERSON" for row in negatives)},
        "results": evaluated}
    (output / "context-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
