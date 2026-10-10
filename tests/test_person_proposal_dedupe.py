from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native/test_person_proposal_dedupe.cpp"
NATIVE = ROOT / "services/camera_v11/native/six_camera_v13"


def test_native_person_proposal_duplicate_policy(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler, "g++ is required to validate the native proposal duplicate policy"
    executable = tmp_path / "test-person-proposal-dedupe"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT}", str(SOURCE), "-o", str(executable)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_deduper_is_after_pgie_and_before_nvdcf_input() -> None:
    detection = (NATIVE / "detection.hpp").read_text()
    infer_exit = detection.split("static GstPadProbeReturn infer_exit(", 1)[1].split(
        "static void detection_stats", 1
    )[0]
    assert infer_exit.index("suppress_same_component_person_duplicates(frame)") < infer_exit.index(
        "for (NvDsMetaList *scan = frame->obj_meta_list"
    )
    assert infer_exit.index("suppress_previous_track_supported_duplicates(frame, source_epoch)") < infer_exit.index(
        "for (NvDsMetaList *scan = frame->obj_meta_list"
    )
    assert "nvds_remove_obj_meta_from_frame(frame" in detection
    assert "tracker_record_detections(frame->source_id" in infer_exit
    assert "PGIE_NESTED_DUPLICATES geometric_suppressed=" in detection
    assert "history_supported_suppressed=" in detection
    assert "proposal-duplicate-suppressions.jsonl" in detection


def test_candidate_builder_keeps_frozen_detector_and_tracker_artifacts() -> None:
    builder = (ROOT / "scripts/build_cam05_duplicate_suppression_candidate.py").read_text()
    assert "r3-coordinate-contract" in builder
    assert "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16" in builder
    assert "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19" in builder
    assert "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355" in builder
    assert "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834" in builder
    assert "nms-iou-threshold" not in builder
