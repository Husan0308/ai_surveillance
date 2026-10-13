#!/usr/bin/env python3
"""Classify a finite localhost RTSP EOS run without applying live-FPS gates after EOS."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re


def audit(run_dir: Path, expected_camera: str, expected_duration_sec: float) -> dict:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    native_log = (run_dir / "native.log").read_text(encoding="utf-8", errors="replace")
    config = (run_dir / "six-synthetic.ini").read_text(encoding="utf-8")
    fixture_rows = []
    for path in sorted(run_dir.glob("fixture-*.log")):
        content = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"LOCAL_RTSP_EOS_FIXTURE_READY port=(\d+) bind=127\.0\.0\.1 frames=(\d+) fps=(\d+) expected_eos_sec=([0-9.]+)",
            content,
        )
        if match:
            fixture_rows.append({
                "port": int(match.group(1)), "frames": int(match.group(2)),
                "fps": int(match.group(3)), "expected_eos_sec": float(match.group(4)),
            })
    camera_ports: dict[str, int] = {}
    current_camera = None
    for line in config.splitlines():
        line = line.strip()
        section = re.fullmatch(r"\[(CAM-\d{2})\]", line)
        if section:
            current_camera = section.group(1)
            continue
        if current_camera and line.startswith("uri="):
            decoded = base64.b64decode(line.split("=", 1)[1]).decode("utf-8")
            port_match = re.search(r"127\.0\.0\.1:(\d+)", decoded)
            if port_match:
                camera_ports[current_camera] = int(port_match.group(1))

    mapped_port = camera_ports.get(expected_camera)
    fixture = next((row for row in fixture_rows if row["port"] == mapped_port), None)
    generations = [int(x) for x in re.findall(
        rf"{re.escape(expected_camera)} SOURCE_OWNER_CREATED .*?generation=(\d+)", native_log
    )]
    recovery_begins = len(re.findall(rf"{re.escape(expected_camera)} SOURCE_RECOVERY_BEGIN ", native_log))
    recovery_passes = len(re.findall(rf"{re.escape(expected_camera)} SOURCE_STALL_RECOVERY_PASS ", native_log))
    group_end = next((line for line in native_log.splitlines() if line.startswith("GROUP END")), "")
    per_camera = result.get("steady_state_fps_window", {}).get("per_camera", {})
    peers = [camera for camera in (f"CAM-{i:02d}" for i in range(1, 7))
             if camera != expected_camera]
    peer_results = {
        camera: per_camera.get(camera, {}).get("status") == "PASS"
        and per_camera.get(camera, {}).get("all_samples_fresh_online") is True
        for camera in peers
    }
    checks = {
        "fixture_is_loopback_finite_source": fixture is not None,
        "fixture_eos_duration_matches_test_spec": bool(
            fixture and abs(fixture["expected_eos_sec"] - expected_duration_sec) < 0.001
            and fixture["frames"] / fixture["fps"] == expected_duration_sec
        ),
        "camera_maps_to_finite_fixture_port": mapped_port is not None and fixture is not None,
        "candidate_exited_cleanly": result.get("exit_code") == 0,
        "source_owner_generation_increases": len(generations) >= 2 and generations == list(range(1, len(generations) + 1)),
        "each_observed_eos_recovery_reached_all_stages": recovery_begins > 0 and recovery_passes == recovery_begins,
        "one_owner_per_fixture": result.get("per_fixture_rtsp_owner_max_one") is True,
        "preview_sync_mismatches_zero": result.get("preview_sync_mismatches_zero") is True,
        "shared_pipeline_stayed_nonfatal": "fatal=0" in group_end and "shared_errors=0" in group_end,
        "all_five_continuous_peers_passed_fps_and_freshness": all(peer_results.values()),
    }
    return {
        "schema_version": 1,
        "scenario": "intentional finite RTSP EOS on one source; app-owned source-local recovery",
        "candidate_binary_sha256": result.get("binary_sha256"),
        "expected_finite_source": expected_camera,
        "finite_fixture": fixture,
        "mapped_loopback_port": mapped_port,
        "owner_generations": generations,
        "observed_eos_recovery_begins": recovery_begins,
        "observed_full_stage_recovery_passes": recovery_passes,
        "continuous_peer_status": peer_results,
        "raw_generic_runner_status": result.get("status"),
        "raw_generic_runner_fps_status_for_finite_camera": per_camera.get(expected_camera, {}).get("status"),
        "finite_camera_normal_fps_gate": "NOT_APPLICABLE_AFTER_DECLARED_EOS",
        "six_continuous_camera_fps_gate": "NOT_APPLICABLE_TO_FINITE_SOURCE_SCENARIO",
        "checks": checks,
        "scenario_status": "PASS" if all(checks.values()) else "FAIL",
        "scope_note": "This verdict covers the declared localhost finite-EOS isolation scenario only; it is not a six-camera continuous-FPS or physical-camera recovery PASS.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--camera", default="CAM-03")
    parser.add_argument("--expected-eos-sec", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir.resolve(), args.camera, args.expected_eos_sec)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["scenario_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
