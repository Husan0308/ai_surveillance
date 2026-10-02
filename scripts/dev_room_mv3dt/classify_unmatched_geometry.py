#!/usr/bin/env python3
"""Classify detector proposals that do not survive NvDCF using geometry evidence.

Read-only diagnostic for the archived final-detector A/B replay. It combines:
  * frame_path_audit.jsonl (PGIE/tracker confidence metadata),
  * pretracker_geometry.csv,
  * posttracker_geometry.csv.

The script is deliberately schema-tolerant. It auto-detects common column names
and, if geometry cannot be parsed, prints the discovered headers instead of
guessing. It never edits production/runtime configs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AB_ROOT = ROOT / ".runtime/final-detector-ab-20261002-uSpPqjLY"

ALIASES = {
    "camera": ("camera_id", "mapped_camera_id", "camera", "cam", "cameraId"),
    "source": ("source_id", "source", "stream_id", "pad_index"),
    "frame": ("frame", "frame_num", "frame_id", "frameNumber"),
    "confidence": ("confidence", "detector_confidence", "det_conf", "score"),
    "left": ("left", "x", "bbox_left", "rect_left"),
    "top": ("top", "y", "bbox_top", "rect_top"),
    "width": ("width", "w", "bbox_width", "rect_width"),
    "height": ("height", "h", "bbox_height", "rect_height"),
    "object_id": ("object_id", "native_track_id", "track_id", "tracking_id"),
}


def pick(fieldnames: list[str], key: str) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for alias in ALIASES[key]:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def q(value: Any) -> float:
    return round(float(value), 6)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0.0 else 0.0


def resolve_run(label: str) -> Path:
    root = AB_ROOT / "runs" / label
    replays = sorted(p for p in root.glob("replay-*") if p.is_dir())
    if not replays:
        raise SystemExit(f"no replay found under {root}")
    return replays[-1]


def load_audit(path: Path, camera: str) -> dict[int, dict[str, list[dict]]]:
    out: dict[int, dict[str, list[dict]]] = defaultdict(dict)
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON {path}:{line_no}: {exc}") from exc
            if rec.get("record") != "frame":
                continue
            if rec.get("mapped_camera_id") != camera:
                continue
            stage = rec.get("stage")
            if stage not in {"pgie", "tracker"}:
                continue
            out[int(rec["frame_num"])][stage] = list(rec.get("objects") or [])
    return out


def unmatched_by_confidence(frame_map: dict[int, dict[str, list[dict]]]) -> list[dict]:
    misses: list[dict] = []
    for frame in sorted(frame_map):
        pgie = frame_map[frame].get("pgie", [])
        tracker = frame_map[frame].get("tracker", [])
        avail = Counter()
        for obj in tracker:
            try:
                conf = q(obj.get("confidence", -1))
            except (TypeError, ValueError):
                continue
            if conf >= 0:
                avail[conf] += 1
        for obj in pgie:
            try:
                conf = q(obj.get("confidence", -1))
            except (TypeError, ValueError):
                continue
            if avail[conf]:
                avail[conf] -= 1
            else:
                misses.append({"frame": frame, "confidence": conf})
    return misses


def load_geometry(path: Path) -> tuple[list[dict], dict]:
    if not path.is_file():
        return [], {"path": str(path), "present": False}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        mapping = {key: pick(fields, key) for key in ALIASES}
        needed = ("frame", "left", "top", "width", "height")
        supported = all(mapping[k] for k in needed)
        rows = list(reader)
    return rows, {
        "path": str(path),
        "present": True,
        "headers": fields,
        "mapping": mapping,
        "geometry_supported": supported,
    }


def row_camera(row: dict, mapping: dict) -> str | None:
    cam_col = mapping.get("camera")
    if cam_col and row.get(cam_col):
        return str(row[cam_col])
    src_col = mapping.get("source")
    if src_col and row.get(src_col) not in (None, ""):
        try:
            src = int(float(row[src_col]))
        except ValueError:
            return str(row[src_col])
        return {0: "CAM-01", 1: "CAM-04"}.get(src, f"source:{src}")
    return None


def parse_row(row: dict, mapping: dict) -> dict | None:
    try:
        frame = int(float(row[mapping["frame"]]))
        box = (
            float(row[mapping["left"]]),
            float(row[mapping["top"]]),
            float(row[mapping["width"]]),
            float(row[mapping["height"]]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    conf = None
    ccol = mapping.get("confidence")
    if ccol and row.get(ccol) not in (None, ""):
        try:
            conf = q(row[ccol])
        except ValueError:
            pass
    oid = None
    ocol = mapping.get("object_id")
    if ocol:
        oid = row.get(ocol)
    return {
        "frame": frame,
        "camera": row_camera(row, mapping),
        "confidence": conf,
        "bbox": box,
        "object_id": oid,
    }


def index_geometry(rows: list[dict], meta: dict, camera: str) -> dict[int, list[dict]]:
    if not meta.get("geometry_supported"):
        return {}
    mapping = meta["mapping"]
    out: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        parsed = parse_row(row, mapping)
        if parsed is None:
            continue
        if parsed["camera"] not in (None, camera):
            continue
        out[parsed["frame"]].append(parsed)
    return out


def band(conf: float) -> str:
    if conf < 0.50:
        return "below_0.50"
    if conf < 0.70:
        return "0.50_to_0.70"
    return "at_least_0.70"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="nvdcf-retention-v6-newtarget035")
    ap.add_argument("--camera", default="CAM-01")
    ap.add_argument("--duplicate-iou", type=float, default=0.50)
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    run = resolve_run(args.label)
    probe = run / "run/logs/probe"
    audit_path = probe / "frame_path_audit.jsonl"
    if not audit_path.is_file():
        raise SystemExit(f"missing audit: {audit_path}")

    frame_map = load_audit(audit_path, args.camera)
    misses = unmatched_by_confidence(frame_map)

    pre_rows, pre_meta = load_geometry(probe / "pretracker_geometry.csv")
    post_rows, post_meta = load_geometry(probe / "posttracker_geometry.csv")
    pre = index_geometry(pre_rows, pre_meta, args.camera)
    post = index_geometry(post_rows, post_meta, args.camera)

    classified = []
    unresolved_geometry = 0
    for miss in misses:
        frame = miss["frame"]
        conf = miss["confidence"]
        candidates = pre.get(frame, [])
        # Prefer exact confidence match when the CSV contains detector confidence.
        exact = [r for r in candidates if r["confidence"] is not None and r["confidence"] == conf]
        target = exact[0] if exact else None
        if target is None:
            unresolved_geometry += 1
            classified.append({
                **miss,
                "band": band(conf),
                "geometry_resolved": False,
            })
            continue

        others = [r for r in candidates if r is not target]
        post_rows_frame = post.get(frame, [])
        max_pre_iou = max((iou(target["bbox"], r["bbox"]) for r in others), default=0.0)
        max_post_iou = max((iou(target["bbox"], r["bbox"]) for r in post_rows_frame), default=0.0)
        duplicate_like = max(max_pre_iou, max_post_iou) >= args.duplicate_iou
        classified.append({
            **miss,
            "band": band(conf),
            "geometry_resolved": True,
            "bbox": [round(x, 3) for x in target["bbox"]],
            "max_iou_other_pretracker": round(max_pre_iou, 4),
            "max_iou_posttracker": round(max_post_iou, 4),
            "duplicate_like": duplicate_like,
        })

    summary = {}
    for name in ("below_0.50", "0.50_to_0.70", "at_least_0.70"):
        items = [x for x in classified if x["band"] == name]
        resolved = [x for x in items if x.get("geometry_resolved")]
        dups = [x for x in resolved if x.get("duplicate_like")]
        summary[name] = {
            "unmatched": len(items),
            "geometry_resolved": len(resolved),
            "duplicate_like": len(dups),
            "duplicate_like_percent_of_resolved": (100.0 * len(dups) / len(resolved)) if resolved else None,
        }

    ranked = sorted(
        classified,
        key=lambda x: (
            0 if x.get("duplicate_like") is False else 1,
            -x["confidence"],
            x["frame"],
        ),
    )

    report = {
        "run": str(run),
        "camera": args.camera,
        "duplicate_iou_threshold": args.duplicate_iou,
        "unmatched_total": len(misses),
        "geometry_unresolved": unresolved_geometry,
        "pretracker_schema": pre_meta,
        "posttracker_schema": post_meta,
        "summary_by_confidence_band": summary,
        "highest_priority_nonduplicate_like_or_unresolved": ranked[: args.top],
        "interpretation_guardrail": (
            "duplicate_like is geometric evidence only, not a production decision. "
            "Do not raise detector thresholds until unmatched boxes are shown to be "
            "spurious/duplicate rather than true people."
        ),
    }

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
