"""Compare saved current-frame post-tracker bboxes between frozen V13 and local6."""
from __future__ import annotations

import json
import argparse
import math
from pathlib import Path
import statistics
import subprocess
import sys

from scripts.run_local_nvdcf_parity_replay import INPUTS, ROOT

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / ".runtime/visual-bbox-review-20261004"


def rows(path: Path) -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for line in path.open():
        row = json.loads(line)
        if row.get("stage") != "tracker":
            continue
        camera = row.get("mapped_camera_id")
        if camera not in {"CAM-01", "CAM-04"}:
            continue
        key = (camera, int(row["frame_num"]))
        if key in result:
            raise ValueError(f"duplicate tracker frame {key} in {path}")
        result[key] = row
    return result


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def compare(reference: dict, local: dict, camera: str) -> dict:
    expected = {(camera, frame) for frame in range(2400)}
    if not expected <= reference.keys() or not expected <= local.keys():
        raise ValueError(f"{camera}: missing frame audit rows; V13={sum(k[0] == camera for k in reference)} local={sum(k[0] == camera for k in local)}")
    all_ious: list[float] = []
    centers: list[float] = []
    width_delta: list[float] = []
    height_delta: list[float] = []
    count_delta: list[int] = []
    id_pairs: dict[int, dict[int, int]] = {}
    frames_with_count_mismatch = 0
    unmatched_reference = unmatched_local = 0
    for frame in range(2400):
        key = (camera, frame)
        a_objs = [o for o in reference[key].get("objects", []) if o.get("class_id") == 0 and len(o.get("bbox", [])) == 4 and int(o.get("object_id", -1)) >= 0]
        b_objs = [o for o in local[key].get("objects", []) if o.get("class_id") == 0 and len(o.get("bbox", [])) == 4 and int(o.get("object_id", -1)) >= 0]
        if str(reference[key].get("pts")) != str(local[key].get("pts")):
            raise ValueError(f"{camera} frame {frame}: PTS differs between V13/local6")
        count_delta.append(len(b_objs) - len(a_objs))
        if len(a_objs) != len(b_objs):
            frames_with_count_mismatch += 1
        candidates = sorted(
            ((iou(a["bbox"], b["bbox"]), ai, bi)
             for ai, a in enumerate(a_objs) for bi, b in enumerate(b_objs)),
            reverse=True,
        )
        used_a: set[int] = set()
        used_b: set[int] = set()
        for score, ai, bi in candidates:
            if ai in used_a or bi in used_b:
                continue
            used_a.add(ai)
            used_b.add(bi)
            a, b = a_objs[ai], b_objs[bi]
            abox, bbox = list(map(float, a["bbox"])), list(map(float, b["bbox"]))
            all_ious.append(score)
            centers.append(math.hypot((abox[0] + abox[2] / 2) - (bbox[0] + bbox[2] / 2),
                                      (abox[1] + abox[3] / 2) - (bbox[1] + bbox[3] / 2)))
            width_delta.append(abs(bbox[2] / abox[2] - 1.0) if abox[2] else 0.0)
            height_delta.append(abs(bbox[3] / abox[3] - 1.0) if abox[3] else 0.0)
            aid, bid = int(a["object_id"]), int(b["object_id"])
            id_pairs.setdefault(aid, {})[bid] = id_pairs.setdefault(aid, {}).get(bid, 0) + 1
        unmatched_reference += len(a_objs) - len(used_a)
        unmatched_local += len(b_objs) - len(used_b)
    id_splits = {str(aid): {"local_ids": sorted(map(str, by_id)),
                            "dominant_local_id_fraction": max(by_id.values()) / sum(by_id.values())}
                 for aid, by_id in id_pairs.items() if len(by_id) > 1}
    return {
        "camera_id": camera,
        "frames_compared": 2400,
        "v13_tracked_boxes": sum(len([o for o in reference[(camera, f)].get("objects", []) if o.get("class_id") == 0]) for f in range(2400)),
        "local6_tracked_boxes": sum(len([o for o in local[(camera, f)].get("objects", []) if o.get("class_id") == 0]) for f in range(2400)),
        "matched_current_boxes": len(all_ious),
        "unmatched_v13_boxes": unmatched_reference,
        "unmatched_local6_boxes": unmatched_local,
        "frames_object_count_mismatch": frames_with_count_mismatch,
        "bbox_iou": {"p5": quantile(all_ious, .05), "median": quantile(all_ious, .5),
                     "p95": quantile(all_ious, .95), "mean": statistics.fmean(all_ious) if all_ious else None},
        "center_delta_px": {"median": quantile(centers, .5), "p95": quantile(centers, .95),
                             "max": max(centers, default=None)},
        "relative_size_delta_abs": {"width_median": quantile(width_delta, .5),
                                    "height_median": quantile(height_delta, .5),
                                    "width_p95": quantile(width_delta, .95),
                                    "height_p95": quantile(height_delta, .95)},
        "reference_track_ids_split_across_local_ids": id_splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--local-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT / ".runtime")
    out.mkdir(parents=True, exist_ok=False)
    local = rows(args.local_audit)
    reference = rows(args.reference_audit)
    metrics = {camera: compare(reference, local, camera) for camera in ("CAM-01", "CAM-04")}
    output = {"reference": "frozen accepted V13 person-present audit; input hashes match the exact person-present pair",
              "reference_audit": str(args.reference_audit), "local6_audit": str(args.local_audit),
              "per_camera": metrics, "human_visual_approval": "PENDING"}

    visual = out / "visual/local6"
    visual.mkdir(parents=True, exist_ok=False)
    reference_visual = out / "visual/v13"
    reference_visual.mkdir(parents=True, exist_ok=False)
    videos = [INPUTS[c][0] for c in ("CAM-01", "CAM-04")]
    renderer = ROOT / ".runtime/visual-bbox-review-20261004/render_bbox_review.py"
    python = str(ROOT / ".runtime/full-stack-venv/bin/python")
    for audit_path, output_dir in ((args.reference_audit, reference_visual),
                                   (args.local_audit, visual)):
        subprocess.run([python, str(renderer), "--audit", str(audit_path),
                        "--cam01", str(videos[0]), "--cam04", str(videos[1]),
                        "--out", str(output_dir)], check=True)
    for prefix, reference_video in (("CAM01", REVIEW / "CAM01-tracker.mp4"),
                                    ("CAM04", REVIEW / "CAM04-tracker.mp4")):
        comparison = out / f"{prefix}-v13-vs-local6.mp4"
        reference_video = reference_visual / f"{prefix}-tracker.mp4"
        local_video = visual / f"{prefix}-tracker.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(reference_video), "-i", str(local_video),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison),
        ], check=True)
        output[f"{prefix}_comparison_video"] = str(comparison)
    output["parity_video_sha256"] = {
        prefix: __import__("hashlib").sha256((out / f"{prefix}-v13-vs-local6.mp4").read_bytes()).hexdigest()
        for prefix in ("CAM01", "CAM04")
    }
    (out / "parity-metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
