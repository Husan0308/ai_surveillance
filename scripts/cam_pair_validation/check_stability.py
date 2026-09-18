#!/usr/bin/env python3
"""Fail closed on weak CAM-01 + CAM-02 DeepStream 9.1 soak evidence."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_rows(text: str, prefix: str):
    return [
        {k: float(v) for k, v in re.findall(rf"(\w+)=({NUMBER_RE})", line)}
        for line in text.splitlines()
        if line.startswith(prefix)
    ]


def assess(source_rows, pair_rows, text, gpu):
    failures = []
    if set(source_rows) != {"CAM-01", "CAM-02"}:
        return {"status": "BLOCKED", "failures": ["Missing source telemetry"]}

    for cid, rows in source_rows.items():
        stable = [r for r in rows if r.get("elapsed", 0) >= 30]
        if len(stable) < 2:
            failures.append(f"{cid}: insufficient telemetry")
            continue
        if stable[-1]["elapsed"] - stable[0]["elapsed"] < 600:
            failures.append(f"{cid}: less than 600 seconds after warmup")
        for a, b in zip(stable, stable[1:]):
            dt = b["elapsed"] - a["elapsed"]
            if not 0 < dt <= 7:
                failures.append(f"{cid}: telemetry gap")
                break
            if b["input"] <= a["input"] or b["pts_ns"] <= a["pts_ns"]:
                failures.append(f"{cid}: frozen frame/PTS counter")
                break
        if any(not 18 <= r["fps"] <= 22 or r["age"] > 1 or r["max_gap_ms"] > 1000 for r in stable):
            failures.append(f"{cid}: FPS/arrival threshold exceeded")
        if any(
            r.get("queue", 0) >= 12
            or r.get("queue_ms", 0) > 600
            or r.get("rtp_lost", 0) > 0
            or r.get("rtp_late", 0) > 0
            or r.get("errors", 0) > 0
            or r.get("warnings", 0) > 0
            or r.get("pts_backwards", 0) > 0
            or r.get("pts_duplicates", 0) > 0
            or r.get("decoder", 0) != 1
            for r in stable
        ):
            failures.append(f"{cid}: queue/loss/error/timestamp/decode failure")

    stable_pair = [r for r in pair_rows if r.get("elapsed", 0) >= 30]
    if len(stable_pair) < 2:
        failures.append("PAIR: insufficient telemetry")
    else:
        if stable_pair[-1]["elapsed"] - stable_pair[0]["elapsed"] < 600:
            failures.append("PAIR: less than 600 seconds after warmup")
        for a, b in zip(stable_pair, stable_pair[1:]):
            if b["output"] <= a["output"] or b["pts_ns"] <= a["pts_ns"]:
                failures.append("PAIR: frozen output/PTS")
                break
        # nvstreammux pushes when a batch fills OR batched-push-timeout expires.
        # With two asynchronous live sources and sync-inputs=false, downstream
        # buffer cadence is not required to equal the per-camera frame rate.
        # Per-camera FPS above is the authoritative rate gate.
        missing_pair_fields = [
            (i, key)
            for i, r in enumerate(stable_pair)
            for key in ("age", "max_gap_ms", "rss_mib", "cpu_pct")
            if key not in r
        ]
        if missing_pair_fields:
            preview = ", ".join(f"row{i}:{key}" for i, key in missing_pair_fields[:5])
            failures.append(f"PAIR: malformed telemetry missing required field(s): {preview}")

        if any(
            r.get("age", float("inf")) > 1
            or r.get("max_gap_ms", float("inf")) > 1000
            or r.get("dropped", 0) > 0
            or r.get("shared_errors", 0) > 0
            or r.get("shared_warnings", 0) > 0
            or r.get("pts_backwards", 0) > 0
            or r.get("pts_duplicates", 0) > 0
            for r in stable_pair
        ):
            failures.append("PAIR: output/error/timestamp threshold exceeded")

        resource_rows = [r for r in stable_pair if "rss_mib" in r and "cpu_pct" in r]
        if len(resource_rows) < len(stable_pair):
            failures.append("PAIR: incomplete RSS/CPU telemetry")
        elif resource_rows:
            rss_growth = statistics.median(r["rss_mib"] for r in resource_rows[-12:]) - statistics.median(
                r["rss_mib"] for r in resource_rows[:12]
            )
            cpu_mean = statistics.mean(r["cpu_pct"] for r in resource_rows)
            if rss_growth > 48:
                failures.append("PAIR: RSS grew more than 48 MiB")
            if cpu_mean > 50:
                failures.append("PAIR: CPU usage exceeded 50% of one logical core")
    if "nvstreammux(batch=2" not in text or "inference=0" not in text:
        failures.append("Missing two-source mux/no-inference evidence")
    for cid in ("CAM-01", "CAM-02"):
        if f"{cid} END " not in text or "hardware_decoder=1" not in text:
            failures.append(f"{cid}: missing clean hardware-decoder completion")
    if "PAIR END " not in text or "fatal=0" not in text:
        failures.append("Missing clean pair completion")
    if re.search(r"PAIR FATAL|CRITICAL|MESA: error|libEGL warning|\bERROR\s", text):
        failures.append("Fatal/critical runtime diagnostic in log")

    g = []
    if pair_rows:
        end = pair_rows[-1].get("elapsed", 0)
        g = [r for r in gpu if 30 <= r["elapsed"] <= end]
    if len(g) < 100:
        failures.append("Insufficient GPU telemetry")
    else:
        if max(r["memory_used_mib"] for r in g) - min(r["memory_used_mib"] for r in g) > 96:
            failures.append("VRAM spread exceeded 96 MiB")
        if max(r["decoder_pct"] for r in g) <= 0:
            failures.append("No NVDEC utilization evidence")
        if max(b["elapsed"] - a["elapsed"] for a, b in zip(g, g[1:])) > 10:
            failures.append("GPU telemetry gap")

    summary = {"status": "PASS" if not failures else "BLOCKED", "failures": failures}
    if stable_pair:
        summary.update(
            {
                "measured_seconds": pair_rows[-1]["elapsed"],
                "steady_seconds": stable_pair[-1]["elapsed"] - stable_pair[0]["elapsed"],
                "output_frames": int(pair_rows[-1]["output"]),
                "pair_fps_min": min(r["fps"] for r in stable_pair),
                "pair_fps_max": max(r["fps"] for r in stable_pair),
                "pair_fps_mean": statistics.mean(r["fps"] for r in stable_pair),
                "cpu_pct_mean_one_core": (
                    statistics.mean(r["cpu_pct"] for r in stable_pair if "cpu_pct" in r)
                    if any("cpu_pct" in r for r in stable_pair)
                    else None
                ),
                "rss_mib_start": next((r["rss_mib"] for r in stable_pair if "rss_mib" in r), None),
                "rss_mib_end": next((r["rss_mib"] for r in reversed(stable_pair) if "rss_mib" in r), None),
            }
        )
    for cid, rows in source_rows.items():
        stable = [r for r in rows if r.get("elapsed", 0) >= 30]
        if stable:
            summary[cid] = {
                "frames": int(rows[-1]["input"]),
                "fps_min": min(r["fps"] for r in stable),
                "fps_max": max(r["fps"] for r in stable),
                "fps_mean": statistics.mean(r["fps"] for r in stable),
                "max_gap_ms": max(r["max_gap_ms"] for r in stable),
                "queue_max": max(r["queue"] for r in stable),
            }
    for key in ("memory_used_mib", "gpu_pct", "decoder_pct", "encoder_pct"):
        if g:
            summary[key] = {
                "min": min(r[key] for r in g),
                "max": max(r[key] for r in g),
                "mean": statistics.mean(r[key] for r in g),
            }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    text = (args.directory / "pipeline.log").read_text()
    source_rows = {
        "CAM-01": parse_rows(text, "CAM-01 STATS "),
        "CAM-02": parse_rows(text, "CAM-02 STATS "),
    }
    pair_rows = parse_rows(text, "PAIR STATS ")

    gpu = []
    with (args.directory / "gpu.csv").open() as f:
        for row in csv.DictReader(f):
            gpu.append(
                {
                    k: float(row[k])
                    for k in (
                        "time",
                        "memory_used_mib",
                        "gpu_pct",
                        "decoder_pct",
                        "encoder_pct",
                    )
                }
            )
    if gpu:
        start = gpu[0]["time"]
        for row in gpu:
            row["elapsed"] = row["time"] - start

    report = assess(source_rows, pair_rows, text, gpu)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_packets",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(args.directory / "CAM-01_CAM-02.mkv"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        video = json.loads(probe.stdout)
        stream = video["streams"][0]
        if probe.returncode or float(video["format"]["duration"]) < 600:
            raise ValueError("recording too short")
        if stream["width"] != 2560 or stream["height"] != 720 or stream["codec_name"] != "h264":
            raise ValueError("unexpected output format")
        if int(stream["nb_read_packets"]) < 12000:
            raise ValueError("recording has too few frames")
        if pair_rows and abs(int(stream["nb_read_packets"]) - int(pair_rows[-1]["output"])) > 12:
            raise ValueError("encoded packet count does not match output counter")
        report["recorded_video"] = video
    except (ValueError, KeyError, IndexError, json.JSONDecodeError):
        report["status"] = "BLOCKED"
        report["failures"].append("Recorded tiled video validation failed")

    (args.directory / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
