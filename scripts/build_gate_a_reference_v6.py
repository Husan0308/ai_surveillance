#!/usr/bin/env python3
"""Create an auditable human-truth correction to the staged Gate A reference.

This does not modify the historical oracle or v5. It removes only three
manually reviewed duplicate upper-body annotations from CAM-01 frames 273-275.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REFERENCE = ROOT / ".runtime/approved-bbox-reference-v5-amended-20261006-r5/reference"
SOURCE_VIDEO = ROOT / ".runtime/amc-transformer-personpresent-MtbZPbuh/amc_inputs/videos/cam_00.mp4"
DEFAULT_OUTPUT = ROOT / ".runtime/approved-bbox-reference-v6-cam01-duplicate-correction-20261010-r3"
EXPECTED_SOURCE_SHA256 = "1e2818e022b58f2f6608b9af24682f9dbc0d628ee80c7754f0c95a7c943f70f1"
EXPECTED_INNER_BOXES = {
    273: [981.375, 328.688, 101.25, 106.125],
    274: [979.125, 327.938, 99.75, 104.625],
    275: [974.625, 324.375, 102.75, 107.25],
}
EXPECTED_PTS = {273: 13_750_000_000, 274: 13_800_000_000, 275: 13_850_000_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path = DEFAULT_OUTPUT) -> dict:
    output = output.resolve()
    if not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("reference output must remain under .runtime")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference evidence: {output}")
    if sha256(SOURCE_VIDEO) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("pinned CAM-01 source video hash mismatch")
    source_manifest = SOURCE_REFERENCE / "reference-manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    output.mkdir(parents=True)
    reference_dir = output / "reference"
    reference_dir.mkdir()
    visual_dir = output / "visual-review"
    visual_dir.mkdir()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(SOURCE_VIDEO),
        "-vf", "select='between(n,275,277)'", "-vsync", "0", "-frames:v", "3",
        "-start_number", "273", str(visual_dir / "CAM01-frame-%03d.png"),
    ], check=True, capture_output=True, text=True)
    visual_images = [visual_dir / f"CAM01-frame-{frame:03d}.png" for frame in sorted(EXPECTED_INNER_BOXES)]
    if any(not path.is_file() for path in visual_images):
        raise RuntimeError("ffmpeg did not extract the three PTS-matched source frames")
    source_csv = SOURCE_REFERENCE / "CAM01-per-frame.csv"
    destination_csv = reference_dir / "CAM01-per-frame.csv"
    target_rows: dict[int, dict] = {}
    kept_rows: dict[int, dict] = {}
    with source_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError("CAM01 reference CSV has no header")
        with destination_csv.open("x", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                frame = int(row["frame"])
                if frame in EXPECTED_INNER_BOXES and row.get("v13_native_track_id") == "24":
                    actual = json.loads(row["chosen_bbox"])
                    if actual != EXPECTED_INNER_BOXES[frame] or int(row["pts"]) != EXPECTED_PTS[frame]:
                        raise RuntimeError(f"unexpected duplicate annotation at CAM-01 frame {frame}")
                    if frame in target_rows:
                        raise RuntimeError(f"multiple duplicate annotation rows at frame {frame}")
                    target_rows[frame] = row
                    continue
                writer.writerow(row)
                if frame in EXPECTED_INNER_BOXES and row.get("v13_native_track_id") == "23":
                    kept_rows[frame] = row

    if set(target_rows) != set(EXPECTED_INNER_BOXES) or set(kept_rows) != set(EXPECTED_INNER_BOXES):
        raise RuntimeError("each reviewed frame must contain exactly one nested duplicate and one full-person row")

    cam04_source = SOURCE_REFERENCE / "CAM04-per-frame.csv"
    cam04_destination = reference_dir / "CAM04-per-frame.csv"
    shutil.copyfile(cam04_source, cam04_destination)
    original_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    amendment = {
        "reference_id": "approved-bbox-reference-v6",
        "status": "staged corrected human-truth benchmark; not production configuration",
        "parent_reference_id": original_manifest.get("reference_id"),
        "parent_reference_manifest": str(source_manifest.relative_to(ROOT)),
        "parent_reference_manifest_sha256": sha256(source_manifest),
        "parent_CAM01_csv_sha256": sha256(source_csv),
        "parent_CAM04_csv_sha256": sha256(cam04_source),
        "source_video": str(SOURCE_VIDEO.relative_to(ROOT)),
        "source_video_sha256": sha256(SOURCE_VIDEO),
        "human_review": {
            "frames": [273, 274, 275],
            "pts_ns": EXPECTED_PTS,
            "visible_people_per_frame": 5,
            "review_method": "independent visual count on unannotated source frames; nested reference row checked against the same visible seated person and the full-person row",
            "finding": "The v5 reference has six active boxes while the source has five visible people. The 101x106/100x105/103x107 nested boxes and the larger full-person boxes cover the same seated person; only the nested upper-body duplicate rows are removed.",
            "visual_evidence_directory": "visual-review/",
            "visual_evidence_sha256": {path.name: sha256(path) for path in visual_images},
        },
        "removed_reference_rows": [
            {
                "camera_id": "CAM-01",
                "frame": frame,
                "pts_ns": EXPECTED_PTS[frame],
                "former_v13_reference_native_id": 24,
                "removed_bbox": EXPECTED_INNER_BOXES[frame],
                "reason": "duplicate upper-body annotation nested inside the same person's full-body box; only one visible person is present",
                "retained_full_person_box": json.loads(kept_rows[frame]["chosen_bbox"]),
                "retained_v13_reference_native_id": kept_rows[frame]["v13_native_track_id"],
            }
            for frame in sorted(EXPECTED_INNER_BOXES)
        ],
        "unchanged_CAM04_csv": True,
        "new_reference_files": {
            "CAM01-per-frame.csv": sha256(destination_csv),
            "CAM04-per-frame.csv": sha256(cam04_destination),
        },
        "rows_removed": len(target_rows),
        "no_detector_or_candidate_outputs_copied_into_reference": True,
    }
    (output / "amendment.json").write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")
    (reference_dir / "reference-manifest.json").write_text(
        json.dumps(amendment, indent=2) + "\n", encoding="utf-8")
    (visual_dir / "review-notes.md").write_text(
        "# CAM-01 source-frame review: reference correction\n\n"
        "The three PNGs are lossless, unannotated source frames extracted from the exact pinned 1920x1080 CAM-01 MP4. "
        "Frame labels and PTS are recorded in `../amendment.json`. Each frame visibly contains five people. "
        "The v5 rows for former native target 24 are compact upper-body boxes fully nested over the same seated "
        "person also covered by the full-person row (former target 23); they do not correspond to a sixth person. "
        "The amendment removes only those three duplicate rows. All other reference geometry and CAM-04 rows are retained.\n",
        encoding="utf-8")
    print(json.dumps(amendment, indent=2))
    return amendment


if __name__ == "__main__":
    build()
