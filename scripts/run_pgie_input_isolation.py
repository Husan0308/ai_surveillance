"""Run PGIE-only input/mux A/B/C/D variants against the frozen room MP4 pair."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime/pgie-input-isolation-20261005"
DIAGNOSTIC_BINARY = os.getenv("PGIE_ISOLATION_BINARY", "pgie-isolation")
VIDEO_DIR = ROOT / ".runtime/amc-transformer-personpresent-MtbZPbuh/amc_inputs/videos"
VIDEOS = {"CAM-01": VIDEO_DIR / "cam_00.mp4", "CAM-04": VIDEO_DIR / "cam_01.mp4"}
ORACLE = ROOT / ".runtime/sixcam-local6-20261004-b/replay-local6-type2-shadow-age/run/logs/probe/pretracker_geometry.csv"
PGIE_SOURCE = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
MODEL_DIR = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003/models/640"
PARSER = ROOT / ".runtime/detection-ab-peoplenet-yolo26m-20261002-pI5KEcrC/models/yolo26m/parser/libyolo26rawperson.so"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
EXPECTED = {
    "engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "parser": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "pgie": "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113",
}
CAPTURE_FRAMES = "1045,1046,1050,1060,1086,1088,1159,1332"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_config(source_count: int = 2) -> str:
    rows = []
    order = list(VIDEOS)
    if source_count == 6:
        order = ["CAM-01", "CAM-01", "CAM-01", "CAM-04", "CAM-04", "CAM-04"]
    for index, camera in enumerate(order):
        name = f"CAM-{index + 1:02d}" if source_count == 6 else camera
        uri = f"file:///workspace/inputs/videos/{VIDEOS[camera].name}"
        rows.extend((f"[{name}]", f"uri={base64.b64encode(uri.encode()).decode()}",
                     "username=", "password=", "latency_ms=0", "drop_on_latency=false",
                     "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    rows.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                 "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(rows)


def stage_configs() -> None:
    config_dir = OUT / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    original = PGIE_SOURCE.read_text()
    if sha(PGIE_SOURCE) != EXPECTED["pgie"]:
        raise RuntimeError("frozen PGIE configuration SHA mismatch")
    for batch in (2, 6):
        changed, count = __import__("re").subn(r"(?m)^batch-size=\d+$", f"batch-size={batch}", original, count=1)
        if count != 1:
            raise RuntimeError("expected exactly one PGIE batch-size field")
        (config_dir / f"pgie-batch{batch}.txt").write_text(changed)
    (config_dir / "sources-pair.ini").write_text(camera_config(2))
    (config_dir / "sources-six-duplicated.ini").write_text(camera_config(6))


def docker_run(case: str, *, width: int, height: int, source_count: int = 2,
               mux_batch: int = 2, pgie_batch: int = 2, live_source: int = 0,
               timeout_us: int = -1, memory_type: int = 0) -> dict:
    run_dir = OUT / "runs" / case
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence: {run_dir}")
    (run_dir / "pixels").mkdir(parents=True)
    cfg = "sources-pair.ini" if source_count == 2 else "sources-six-duplicated.ini"
    camera_path = run_dir / "sources.ini"
    camera_path.write_text((OUT / "config" / cfg).read_text())
    pgie_path = f"/config/pgie-batch{pgie_batch}.txt"
    args = [
        "docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all", "--network=host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", f"PGIE_ISOLATION_SOURCE_COUNT={source_count}",
        "-e", f"PGIE_ISOLATION_MUX_BATCH={mux_batch}",
        "-e", f"PGIE_ISOLATION_MUX_WIDTH={width}",
        "-e", f"PGIE_ISOLATION_MUX_HEIGHT={height}",
        "-e", f"PGIE_ISOLATION_LIVE_SOURCE={live_source}",
        "-e", f"PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US={timeout_us}",
        "-e", f"PGIE_ISOLATION_MUX_MEMORY_TYPE={memory_type}",
        "-e", f"PGIE_ISOLATION_CONFIG={pgie_path}",
        "-e", f"PGIE_ISOLATION_CAPTURE_FRAMES={CAPTURE_FRAMES}",
        "-v", f"{OUT / 'bin'}:/candidate:ro",
        "-v", f"{OUT / 'config'}:/config:ro",
        "-v", f"{MODEL_DIR}:/models:ro",
        "-v", f"{PARSER}:/models/libyolo26rawperson.so:ro",
        "-v", f"{VIDEO_DIR}:/workspace/inputs/videos:ro",
        "-v", f"{run_dir}:/work:rw",
        IMAGE, f"/candidate/{DIAGNOSTIC_BINARY}", "/work/sources.ini", "600",
    ]
    metadata = {
        "case": case, "argv_redacted": args[:args.index(IMAGE)] + [IMAGE, "<candidate>", "<sources.ini>", "600"],
        "source_count": source_count, "mux_batch": mux_batch, "pgie_batch": pgie_batch,
        "mux_width": width, "mux_height": height, "live_source": live_source,
        "batched_push_timeout_us": timeout_us, "mux_memory_type": memory_type,
        "source_mapping": ["CAM-01", "CAM-04"] if source_count == 2 else
                          ["CAM-01 (repeat)"] * 3 + ["CAM-04 (repeat)"] * 3,
        "started_epoch": time.time(), "input_sha256": {c: sha(p) for c, p in VIDEOS.items()},
        "engine_sha256": sha(MODEL_DIR / "yolo26m_raw_otm_b6_fp16.engine"),
        "parser_sha256": sha(PARSER), "pgie_source_sha256": sha(PGIE_SOURCE),
        "pgie_staged_sha256": sha(OUT / "config" / f"pgie-batch{pgie_batch}.txt"),
    }
    (run_dir / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (run_dir / "native.log").open("x") as log:
        result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    metadata["return_code"] = result.returncode
    metadata["finished_epoch"] = time.time()
    (run_dir / "exit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"{case} failed; inspect {run_dir / 'native.log'}")
    return metadata


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def read_oracle() -> dict[tuple[str, int], list[dict]]:
    result: dict[tuple[str, int], list[dict]] = {}
    with ORACLE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["camera_id"], int(row["frame"]))
            result.setdefault(key, []).append({
                "bbox": [float(row["left"]), float(row["top"]), float(row["width"]), float(row["height"])],
                "confidence": float(row["confidence"]), "pts_ns": int(row["pts"]),
            })
    return result


def read_candidate(run: Path, source_ids: dict[int, str]) -> tuple[dict, dict]:
    frames = {}
    for line in (run / "frames.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["source_id"] in source_ids:
            frames[(source_ids[row["source_id"]], int(row["frame"]))] = row
    boxes: dict[tuple[str, int], list[dict]] = {}
    for line in (run / "detections.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["source_id"] in source_ids:
            key = (source_ids[row["source_id"]], int(row["frame"]))
            meta = frames.get(key, {})
            px = meta.get("pipeline_width", 0) or int(run.joinpath("launch.json").read_text() and json.loads((run / "launch.json").read_text())["mux_width"])
            py = meta.get("pipeline_height", 0) or json.loads((run / "launch.json").read_text())["mux_height"]
            sx = (meta.get("source_width") or 1920) / px
            sy = (meta.get("source_height") or 1080) / py
            x, y, w, h = row["box"]
            boxes.setdefault(key, []).append({"bbox": [x * sx, y * sy, w * sx, h * sy],
                                               "confidence": row["confidence"], "pts_ns": row["pts_ns"]})
    return frames, boxes


def compare(candidate: Path, name: str, source_ids: dict[int, str]) -> dict:
    oracle = read_oracle()
    frames, boxes = read_candidate(candidate, source_ids)
    report = {"variant": name, "cameras": {}}
    for camera in VIDEOS:
        keys = sorted(key for key in oracle if key[0] == camera)
        ious, conf_delta, counts_mismatch = [], [], []
        first_diff = None
        matched = missing = extra = 0
        for key in keys:
            ref = oracle[key]
            got = boxes.get(key, [])
            if len(ref) != len(got) and first_diff is None:
                first_diff = {"frame": key[1], "oracle_count": len(ref), "candidate_count": len(got)}
            counts_mismatch.append(len(ref) - len(got))
            remaining = set(range(len(got)))
            pairs = []
            for r in ref:
                if not remaining:
                    break
                j = max(remaining, key=lambda idx: iou(r["bbox"], got[idx]["bbox"]))
                remaining.remove(j)
                pairs.append((r, got[j]))
            for r, g in pairs:
                matched += 1
                ious.append(iou(r["bbox"], g["bbox"]))
                conf_delta.append(abs(r["confidence"] - g["confidence"]))
            missing += max(0, len(ref) - len(pairs))
            extra += max(0, len(got) - len(pairs))
        report["cameras"][camera] = {
            "oracle_frames": len(keys), "candidate_frames": sum(1 for k in frames if k[0] == camera),
            "oracle_boxes": sum(len(oracle[k]) for k in keys),
            "candidate_boxes": sum(len(boxes.get(k, [])) for k in keys),
            "matched": matched, "missing": missing, "extra": extra,
            "first_count_divergence": first_diff,
            "iou_p05": sorted(ious)[max(0, int(len(ious) * .05) - 1)] if ious else None,
            "iou_median": statistics.median(ious) if ious else None,
            "iou_p95": sorted(ious)[min(len(ious)-1, int(len(ious) * .95))] if ious else None,
            "confidence_abs_delta_median": statistics.median(conf_delta) if conf_delta else None,
            "confidence_abs_delta_p95": sorted(conf_delta)[min(len(conf_delta)-1, int(len(conf_delta) * .95))] if conf_delta else None,
            "pts_offset_p50_ns": statistics.median([boxes[k][0]["pts_ns"] - oracle[k][0]["pts_ns"]
                                                       for k in keys if boxes.get(k) and oracle.get(k)]) if keys else None,
            "pixel_capture_records": [line for line in (candidate / "mux-pixels.csv").read_text().splitlines()[1:]
                                      if line.startswith(str(list(source_ids).index(next(i for i, c in source_ids.items() if c == camera))) + ",")]
                                      if (candidate / "mux-pixels.csv").exists() else [],
        }
    return report


def main() -> None:
    if not OUT.exists():
        raise RuntimeError(f"expected prepared candidate directory: {OUT}")
    if sha(MODEL_DIR / "yolo26m_raw_otm_b6_fp16.engine") != EXPECTED["engine"] or sha(PARSER) != EXPECTED["parser"]:
        raise RuntimeError("frozen YOLO engine/parser hash mismatch")
    if any(not path.is_file() for path in VIDEOS.values()):
        raise RuntimeError("authorized CAM-01/CAM-04 inputs missing")
    stage_configs()
    if sha(OUT / "bin/pgie-isolation") == "":
        raise RuntimeError("diagnostic binary missing")
    runs = []
    runs.append(docker_run("B-integrated-oracle-geometry", width=1920, height=1080,
                           source_count=2, mux_batch=2, pgie_batch=2,
                           live_source=0, timeout_us=-1, memory_type=0))
    runs.append(docker_run("C-mux-resolution-only", width=2560, height=1440,
                           source_count=2, mux_batch=2, pgie_batch=2,
                           live_source=0, timeout_us=-1, memory_type=0))
    reports = [compare(OUT / "runs" / item["case"], item["case"], {0: "CAM-01", 1: "CAM-04"})
               for item in runs]
    (OUT / "B-C-pgie-comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    (OUT / "run-manifest.json").write_text(json.dumps({
        "source_mp4_sha256": {c: sha(p) for c, p in VIDEOS.items()},
        "approved_oracle_audit": str(ORACLE.relative_to(ROOT)),
        "approved_oracle_configs": {
            "mux": {"sources": 2, "width": 1920, "height": 1080, "batch": 2,
                    "live_source": 0, "batched_push_timeout_us": -1},
            "pgie_batch": 2,
        },
        "integrated_diagnostic_binary_sha256": sha(OUT / "bin/pgie-isolation"),
        "integrated_source_sha256": {
            path.name: sha(path) for path in sorted((ROOT / "services/camera_v11/native/six_camera_v13").glob("*.cpp"))
        } | {"main.cpp": sha(ROOT / "services/camera_v11/native/six_camera_v13/main.cpp"),
             "detection.hpp": sha(ROOT / "services/camera_v11/native/six_camera_v13/detection.hpp")},
        "engine_sha256": EXPECTED["engine"], "parser_sha256": EXPECTED["parser"],
        "thresholds_parser_preprocess_unchanged": True,
        "tracker_in_graph": False, "identity_in_graph": False, "mv3dt_in_graph": False,
        "notes": "B/C are detector-only; C changes mux dimensions only. Oracle A PGIE records are the saved 2-source approved replay audit for the exact hashed input pair.",
    }, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
