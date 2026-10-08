from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native/test_shadow_context_policy.cpp"


def test_native_shadow_context_policy(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler, "g++ is required to validate native shadow-context policy"
    executable = tmp_path / "test-shadow-context-policy"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT}", str(SOURCE), "-o", str(executable)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_shadow_refresh_is_scheduled_before_tracker_but_published_only_for_current_shadow() -> None:
    tracker = (ROOT / "services/camera_v11/native/six_camera_v13/tracker.hpp").read_text()
    entry = tracker.split("static GstPadProbeReturn tracker_enter(", 1)[1].split(
        "static GstPadProbeReturn tracker_exit(", 1)[0]
    exit_body = tracker.split("static GstPadProbeReturn tracker_exit(", 1)[1].split(
        "static void tracker_stats(", 1)[0]
    refresh_block = exit_body.split("// Shadow context requests are scheduled", 1)[1].split(
        "if (tracker_decision_evidence)", 1)[0]
    schedule = tracker.split("static void tracker_schedule_shadow_context_before_nvdcf(", 1)[1].split(
        "static GstPadProbeReturn tracker_enter(", 1)[0]
    apply = tracker.split("static void tracker_apply_verified_shadow_context(", 1)[1].split(
        "static void tracker_apply_verified_lowconf(", 1)[0]
    assert "tracker_schedule_shadow_context_before_nvdcf(buffer, frame)" in entry
    assert "lowconf_verifier_submit(buffer, frame->batch_id, key" in schedule
    assert "tracker_birth_verification.verified(source_id, native_id)" in schedule
    assert "shadow_context_pretracker_hint_eligible" in schedule
    assert "shadow_context_has_recent_primary_support" in schedule
    assert "shadow_context_group_hints(surface_width, surface_height" in schedule
    assert "candidate.shadow_targets.push_back" in schedule
    assert "candidate.shadow_context_refresh" in exit_body
    assert "for (const auto &target : candidate.shadow_targets)" in exit_body
    assert "lowconf_verifier_submit(buffer, frame->batch_id" not in refresh_block
    assert "candidate.source_frame_num != frame_num" in apply
    assert "candidate.source_pts_ns != pts_ns" in apply
    assert "shadow.state == INACTIVE" in apply
    assert "shadow.immediate_continuity" in apply
    assert "shadow_context_match_one_to_one(target_rects, proposal_rects)" in apply
    assert "candidate.verifier_boxes" in apply
