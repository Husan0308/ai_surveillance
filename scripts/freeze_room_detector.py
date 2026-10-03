"""F4 evidence derived from raw diagnostics; no acceptance rebaselining."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import os

import numpy as np
from PIL import Image, ImageDraw

from scripts.build_room_candidate import ROOT, verify_build
from scripts.run_room_candidate import save, protected


def inspect_live(run: Path) -> dict:
    result = json.loads((run / "result.json").read_text())
    rows = [json.loads(l) for l in (run / "run/logs/probe/frame_path_audit.jsonl").read_text().splitlines()]
    detections = {camera: [r for r in rows if r.get("stage") == "pgie" and r["mapped_camera_id"] == camera]
                  for camera in ("CAM-01", "CAM-04")}
    diagnostics = defaultdict(list)
    for line in (run / "run/logs/probe/preview_diagnostics.jsonl").read_text().splitlines():
        row = json.loads(line)
        diagnostics[row["camera_id"]].append(row)
    samples = json.loads((run / "samples.json").read_text())
    gpu = [list(map(float, row["gpu"].split(","))) for row in samples]
    snapshot_manifest = json.loads((run / "visual/snapshots.json").read_text())
    overlays = []
    for snap in snapshot_manifest:
        nearest = min(detections[snap["camera_id"]], key=lambda r: abs(int(r["pts"]) - snap["pts"]))
        image = Image.open(snap["path"])
        draw = ImageDraw.Draw(image)
        pts_delta = (int(nearest["pts"]) - snap["pts"]) / 1e6
        draw.text((15, 15), f'{snap["camera_id"]} YOLO26m DET frame={nearest["frame_num"]} PTS delta={pts_delta:.1f}ms', fill="yellow", stroke_width=2, stroke_fill="black")
        for obj in nearest["objects"]:
            x, y, w, h = obj["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="lime", width=3)
            draw.text((x, max(30, y - 16)), f'DET {obj["confidence"]:.3f}', fill="lime", stroke_width=1, stroke_fill="black")
        destination = Path(snap["path"]).with_name(Path(snap["path"]).stem + "-DET.jpg")
        image.save(destination)
        overlays.append({"path": str(destination), "pts_delta_ms": pts_delta, "frame": nearest["frame_num"]})
    result["overlay_evidence"] = overlays
    result["gpu"] = {"mean_utilization": float(np.mean([g[0] for g in gpu])),
                     "peak_utilization": max(g[0] for g in gpu), "peak_memory_mib": max(g[1] for g in gpu)}
    checks = {"runtime_pass": result["status"] == "PASS", "successful_shutdown": result["exit_code"] == 0,
        "two_rtsp_owners": all(r["rtsp_count"] == 2 for r in samples if len(r["preview"]) == 2),
        "tracker_disabled": "tracker" not in result["stages"], "calibration_unchanged": result["protected_unchanged"]}
    log = (run / "deepstream.log").read_text(errors="replace")
    for camera, index in (("CAM-01", 0), ("CAM-04", 1)):
        diag = diagnostics[camera]
        output = [r["decoder_output"] for r in diag]
        sampled = [(s["mono_ns"], d) for s in samples for d in (s.get("decoder") or []) if d["camera_id"] == camera]
        if len(sampled) < 2:
            raise ValueError("wall-clock decoder counters required for acceptance FPS")
        counters = {"queue_high_water": max(r["analytics_queue_level"] for r in diag),
            "queue_limit": diag[-1]["analytics_queue_limit"], "queue_full_events": diag[-1]["analytics_queue_overruns"],
            "decoder_input": diag[-1]["decoder_input"], "decoder_output": output[-1],
            "decode_fps": (sampled[-1][1]["decoder_output"] - sampled[0][1]["decoder_output"]) / ((sampled[-1][0] - sampled[0][0]) / 1e9),
            "pts_misses": diag[-1]["preview_pts_misses"], "input_backsteps": diag[-1]["input_pts_backsteps"],
            "output_backsteps": diag[-1]["output_pts_backsteps"],
            "instrumented_decoders": log.count(f"exact decoder pads instrumented camera_index={index}")}
        confidence = [o["confidence"] for r in detections[camera] for o in r["objects"]]
        timings = [s["preview"][camera] for s in samples if camera in s["preview"]]
        counters["preview_timing_invalid"] = sum(not (0 < t["t0"] <= t["t1"] <= t["t6"]) for t in timings)
        counters["decoder_out_to_publish_p95_ms"] = float(np.percentile([(t["t6"] - t["t1"]) / 1e6 for t in timings], 95))
        counters["decoder_ref_to_publish_p95_ms"] = float(np.percentile([(t["t6"] - t["t0"]) / 1e6 for t in timings], 95))
        counters["confidence_p5_p50_p95"] = np.percentile(confidence, [5, 50, 95]).tolist()
        checks[f"{camera}_bounded_queues"] = counters["queue_high_water"] <= counters["queue_limit"] and counters["queue_full_events"] == 0
        checks[f"{camera}_timing_integrity"] = counters["pts_misses"] == counters["input_backsteps"] == counters["output_backsteps"] == counters["preview_timing_invalid"] == 0
        checks[f"{camera}_one_decoder"] = counters["instrumented_decoders"] == 1
        checks[f"{camera}_decode_fps"] = 19 <= counters["decode_fps"] <= 21
        result["cameras"][camera].update(counters)
    result["checks"] = checks
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    save(run / "detector-evidence.json", result)
    return result


def finalize(output: Path, live: Path, app_regression: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    live_result = inspect_live(live)
    app = json.loads((app_regression / "result.json").read_text())
    if live_result["status"] != "PASS" or app["status"] != "PASS":
        raise RuntimeError("F4 or previous app freeze failed")
    build = verify_build(ROOT / ".runtime/freeze/F4-yolo26m-live/candidate-audit-build/build.json")
    python = ROOT / ".runtime/full-stack-venv/bin/python"
    commands = [[str(python), "-B", "-m", "pytest", "-q", "tests", f"--junitxml={output / 'unit.xml'}"],
        ["bash", "tests/native/run_bbox_history_lifecycle_tests.sh"],
        ["bash", "scripts/start_full_live_stack.sh", "--preflight"],
        [str(python), "-I", "-B", "-m", "pip", "check"],
        [str(python), "-B", "scripts/freeze_repo_baseline.py", "--output", ".runtime/freeze/F0-repo-baseline", "--verify"],
        ["git", "diff", "--check"]]
    checks = []
    with (output / "test-results.txt").open("x") as log:
        for cmd in commands:
            test = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                env=dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen"))
            log.write(json.dumps(cmd) + "\n" + test.stdout + test.stderr + "\n")
            checks.append({"command": cmd, "exit": test.returncode})
    current = protected()
    original = json.loads((live / "protected-before.json").read_text())
    save(output / "protected-before.json", original)
    save(output / "protected-after.json", current)
    result = {"status": "PASS" if all(c["exit"] == 0 for c in checks) and current == original else "FAIL",
        "checks": checks, "protected_unchanged": current == original, "live": live_result,
        "previous_freeze_runtime": str(app_regression), "previous_freeze_result": app["status"], "build": build}
    save(output / "final-acceptance.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--app-regression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.output, args.live, args.app_regression)
    print(json.dumps({"status": result["status"], "protected_unchanged": result["protected_unchanged"]}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
