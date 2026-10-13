#!/usr/bin/env python3
"""Exercise app-owned authentication quarantine/retry using localhost only."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_verified_candidate_synthetic_six as soak
from scripts.run_r11_reliability_soak_throttled import GpuSampler

CAMERA_IDS = tuple(f"CAM-{index:02d}" for index in range(1, 7))


def evaluate_peer_progress(resource_rows: list[dict], expected_offline: set[str],
                           start_sec: float, end_sec: float) -> dict:
    """Evaluate only healthy peers for an expected source-local failure test."""
    window = [row for row in resource_rows
              if start_sec <= float(row.get("monotonic_sec", -1)) <= end_sec]
    peers = [camera for camera in CAMERA_IDS if camera not in expected_offline]
    stages = ("analytics_input_frames_total", "decoded_frames", "pgie_frames_total",
              "canonical_tracker_updates_total", "preview_frames_total")
    ages = ("age_since_last_decoded_frame_ms", "age_since_last_pgie_frame_ms",
            "age_since_last_tracker_output_ms", "age_since_last_preview_ms")
    reports = {}
    for camera in peers:
        rows = [row.get("cameras", {}).get(camera, {}) for row in window]
        deltas = {}
        counters_monotonic = len(rows) >= 2
        for key in stages:
            values = [row.get(key) for row in rows]
            counters_monotonic = counters_monotonic and all(
                isinstance(value, (int, float)) for value in values
            ) and all(right >= left for left, right in zip(values, values[1:]))
            deltas[key] = (values[-1] - values[0]
                           if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values)
                           else None)
        fresh = bool(rows) and all(
            row.get("online") is True and row.get("current_state") == "ONLINE"
            and row.get("pipeline_errors") == 0
            and all(isinstance(row.get(age), (int, float)) and row[age] <= 250.0
                    for age in ages)
            for row in rows
        )
        reports[camera] = {
            "sample_count": len(rows), "fresh_online_throughout_window": fresh,
            "stage_counters_monotonic": counters_monotonic,
            "stage_deltas": deltas,
            "max_stage_age_ms": {
                age: max((row[age] for row in rows if isinstance(row.get(age), (int, float))),
                        default=None)
                for age in ages
            },
            "pass": fresh and counters_monotonic and all(
                isinstance(deltas[key], (int, float)) and deltas[key] > 0 for key in stages
            ),
        }
    return {
        "window_sec": [start_sec, end_sec],
        "expected_offline_cameras": sorted(expected_offline),
        "sample_count": len(window),
        "peers": reports,
        "pass": bool(reports) and all(row["pass"] for row in reports.values()),
    }


def evaluate_auth_scenario(auth_report: dict) -> dict:
    """Return this scenario's verdict, independent of generic-soak assertions.

    The generic six-camera runner intentionally observes a camera-local 401 and
    a long quarantine window here. Its normal continuous-FPS verdict is useful
    diagnostic context, but is not the acceptance result for this auth test.
    """
    required_true = (
        "unauthorized_seen", "authentication_recovery_started",
        "quarantine_300s_recorded", "application_created_generation_2",
        "source_epoch_advanced_to_2", "recovered_through_all_stages",
        "native_retry_warning_absent", "interval_properties_read_back_zero",
        "fixture_auth_was_corrected", "other_cameras_progressed",
        "preview_sync_mismatches_zero", "candidate_clean_exit",
    )
    checks = {key: auth_report.get(key) is True for key in required_true}
    checks["fixture_uses_single_owner"] = (
        auth_report.get("fixture_max_established_owners") == 1
        and auth_report.get("fixture_client_sessions", 0) >= 2
    )
    checks["camera_online_after_recovery"] = (
        auth_report.get("cam02_final_online") is True
        and auth_report.get("cam02_final_state") == "ONLINE"
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "scope": "localhost RTSP 401 quarantine, credential correction, and app-owned source recovery",
    }


def combine_auth_test_outcome(auth_report: dict, generic_status: str | None,
                              generic_exit_code: int) -> dict:
    """Keep the specialized scenario verdict primary and raw soak result visible."""
    scenario = evaluate_auth_scenario(auth_report)
    return {
        "status": scenario["status"],
        "result_code": 0 if scenario["status"] == "PASS" else 1,
        "scenario_checks": scenario["checks"],
        "failed_scenario_checks": scenario["failed_checks"],
        "scenario_scope": scenario["scope"],
        "generic_soak_status": generic_status,
        "generic_soak_exit_code": generic_exit_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=350)
    parser.add_argument("--accept-after-sec", type=float, default=30.0)
    args = parser.parse_args()
    if not 1 <= args.accept_after_sec <= 30:
        raise ValueError("fixture credential correction must be in 1..30 seconds")
    auth_port = 18555  # CAM-02; loopback-only fixture port

    def write_sources(path: Path) -> list[str]:
        uris = [f"rtsp://127.0.0.1:{port}/synthetic" for port in soak.PORTS]
        lines: list[str] = []
        for index, uri in enumerate(uris, 1):
            credentials = ("r11-test", "r11-secret") if index == 2 else ("", "")
            lines.extend((
                f"[CAM-{index:02d}]",
                f"uri={base64.b64encode(uri.encode()).decode()}",
                f"username={base64.b64encode(credentials[0].encode()).decode()}",
                f"password={base64.b64encode(credentials[1].encode()).decode()}",
                "latency_ms=0", "drop_on_latency=false",
                "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", "",
            ))
        lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                      "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
        fd = path.open("x", encoding="utf-8")
        with fd:
            fd.write("\n".join(lines))
        return uris

    real_popen = subprocess.Popen
    auth_fixture = ROOT / "scripts/reliability_rtsp_auth_fixture.py"

    def controlled_popen(command, *positional, **kwargs):
        if (isinstance(command, (list, tuple)) and "local_rtsp_test_fixture.py" in
                " ".join(map(str, command)) and str(auth_port) in command):
            command = ["/usr/bin/python3", str(auth_fixture), "--port", str(auth_port),
                       "--accept-after-sec", str(args.accept_after_sec)]
        return real_popen(command, *positional, **kwargs)

    soak.write_sources = write_sources
    subprocess.Popen = controlled_popen
    gpu_sampler = GpuSampler()
    gpu_sampler.start_and_wait()
    soak.used_gpu_memory_mib = gpu_sampler.current_mib
    started = time.monotonic()
    result_code = 1
    auth_exit_code = 1
    try:
        result_code = soak.run(args.candidate, args.output, args.duration, "", "")
    finally:
        gpu_sampler.stop()
        subprocess.Popen = real_popen
        if args.output.exists():
            (args.output / "gpu-monitor.json").write_text(json.dumps({
                "period_sec": gpu_sampler.period_sec,
                "sample_count": len(gpu_sampler.samples),
                "status": "PASS" if gpu_sampler.samples and all(
                    row["status"] == "PASS" for row in gpu_sampler.samples
                ) else "REVIEW_REQUIRED",
                "samples": gpu_sampler.samples,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            native_log = (args.output / "native.log").read_text(errors="replace") if (args.output / "native.log").exists() else ""
            fixture_log = (args.output / f"fixture-{auth_port}.log").read_text(errors="replace") if (args.output / f"fixture-{auth_port}.log").exists() else ""
            result_path = args.output / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            monitoring_path = args.output / "monitoring.json"
            monitoring = json.loads(monitoring_path.read_text(encoding="utf-8")) if monitoring_path.exists() else {}
            cameras = {row.get("camera_id"): row for row in monitoring.get("cameras", [])}
            camera_2 = cameras.get("CAM-02", {})
            resource_rows = []
            resource_path = args.output / "resource-samples.jsonl"
            if resource_path.exists():
                resource_rows = [json.loads(line) for line in resource_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            peer_progress = evaluate_peer_progress(resource_rows, {"CAM-02"}, 19.0, 40.0)
            quarantine_sample = next((row for row in resource_rows
                                      if row.get("cameras", {}).get("CAM-02", {}).get("current_state") == "QUARANTINED"), None)
            generation_2_sample = next((row for row in resource_rows
                                        if int(row.get("cameras", {}).get("CAM-02", {}).get("source_generation") or 0) >= 2), None)
            quarantine_to_retry_sec = None
            if quarantine_sample and generation_2_sample:
                quarantine_to_retry_sec = round(
                    float(generation_2_sample["monotonic_sec"]) - float(quarantine_sample["monotonic_sec"]), 3)
            auth_report = {
                "status": "PENDING",
                "binary_sha256": result.get("binary_sha256"),
                "unauthorized_seen": ("unauthorized" in native_log.lower() or "not authorized" in native_log.lower()),
                "authentication_recovery_started": "CAM-02 SOURCE_RECOVERY_BEGIN" in native_log and "reason=authentication_rejected" in native_log,
                "quarantine_to_retry_sec": quarantine_to_retry_sec,
                "quarantine_300s_recorded": quarantine_to_retry_sec is not None and 290.0 <= quarantine_to_retry_sec <= 315.0,
                "application_created_generation_2": "CAM-02 SOURCE_OWNER_CREATED" in native_log and "generation=2" in native_log,
                "source_epoch_advanced_to_2": bool(
                    int((generation_2_sample or {}).get("cameras", {}).get("CAM-02", {}).get("source_epoch") or 0) >= 2
                ),
                "recovered_through_all_stages": "CAM-02 SOURCE_STALL_RECOVERY_PASS" in native_log,
                "native_retry_warning_absent": "CAM-02 NATIVE_RTSP_RECONNECT_WARNING" not in native_log,
                "interval_properties_read_back_zero": (
                    "CAM-02 SOURCE_RECONNECT_PROPERTIES" in native_log and
                    "rtsp-reconnect-interval=0" in native_log and
                    "init-rtsp-reconnect-interval=0" in native_log and
                    "rtsp-reconnect-attempts-effective=-1" in native_log and
                    "retry_owner=application_watchdog" in native_log
                ),
                "fixture_auth_was_corrected": "RTSP_AUTH_ACCEPTED_CREDENTIALS_UPDATED" in fixture_log,
                "fixture_client_sessions": fixture_log.count("RTSP_CLIENT_CONNECTED"),
                "fixture_max_established_owners": result.get("max_client_owners_by_port", {}).get(str(auth_port)),
                "other_cameras_progressed": peer_progress["pass"],
                "other_camera_progress_evidence": peer_progress,
                "preview_sync_mismatches_zero": result.get("preview_sync_mismatches_zero") is True,
                "cam02_final_state": camera_2.get("current_state"),
                "cam02_final_online": camera_2.get("online"),
                "candidate_exit_code": result.get("exit_code"),
                "candidate_clean_exit": result.get("exit_code") == 0,
                "generic_soak_status": result.get("status"),
                "generic_soak_nonpass_reason": (
                    "Generic runner is not the auth scenario verdict: it expects a planned outage marker and "
                    "counts the intentionally rejected CAM-02 401 as an unqualified error. Its independent "
                    "shared-progress gap remains a separate reliability finding."
                ),
                "no_physical_camera_connections": True,
            }
            outcome = combine_auth_test_outcome(
                auth_report, result.get("status"), result_code
            )
            auth_report.update({
                "status": outcome["status"],
                "scenario_checks": outcome["scenario_checks"],
                "failed_scenario_checks": outcome["failed_scenario_checks"],
                "scenario_scope": outcome["scenario_scope"],
            })
            auth_report["shared_progress_gap_ms_max"] = max(
                (float(value) for value in re.findall(
                    r"GROUP STATS .*?max_gap_ms=([0-9.]+)", native_log)), default=None
            )
            auth_report["overall_shared_progress_status"] = "UNRESOLVED"
            auth_exit_code = 0 if auth_report["status"] == "PASS" else 1
            (args.output / "auth-recovery-verification.json").write_text(
                json.dumps(auth_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            record = {
                "test": "r11 app recovery after loopback RTSP 401 then fixture credential correction",
                "candidate": str(args.candidate.resolve().relative_to(ROOT)),
                "auth_camera": "CAM-02",
                "fixture_bind": "127.0.0.1",
                "fixture_port": auth_port,
                "invalid_credentials_initially": True,
                "credentials_corrected_after_sec": args.accept_after_sec,
                "configured_client_credentials": "r11-test/r11-secret (synthetic only)",
                "duration_sec": args.duration,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "no_physical_camera_connections": True,
                # The scenario wrapper's exit code is authoritative. Preserve
                # the generic continuous-soak result separately for audit.
                "result_code": outcome["result_code"],
                "scenario_status": auth_report["status"],
                "generic_soak_exit_code": outcome["generic_soak_exit_code"],
                "auth_scenario_exit_code": auth_exit_code,
                "auth_recovery_verification": "auth-recovery-verification.json",
                "auth_recovery_status": auth_report["status"],
                "generic_soak_status": result.get("status"),
                "overall_shared_progress_status": auth_report["overall_shared_progress_status"],
            }
            (args.output / "auth-recovery-test-plan.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return auth_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
