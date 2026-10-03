"""Run repeated occupied F5 live teardown cycles against one pinned candidate build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_room_candidate import ROOT


def _rtsp_sockets() -> list[str]:
    result = subprocess.run(["ss", "-Htnp"], capture_output=True, text=True, check=False)
    return [line for line in result.stdout.splitlines()
            if ":554" in line and ("ESTAB" in line or "ESTABLISHED" in line)]


def _candidate_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False,
    )
    return [name for name in result.stdout.splitlines()
            if name.startswith("ai-surveillance-v13-candidate-")]


def _max_gap_frames(result: dict) -> int | None:
    retention = (result or {}).get("retention") or {}
    cameras = retention.get("cameras") or {}
    if not cameras:
        return None
    gaps = []
    for camera in ("CAM-01", "CAM-04"):
        row = cameras.get(camera)
        if not row:
            return None
        windows = row.get("association_deficit_windows") or []
        gaps.append(max((int(w.get("frames", 0)) for w in windows), default=0))
    return max(gaps)


def cycle_pass(row: dict) -> bool:
    result = row.get("result") or {}
    retention = result.get("retention") or {}
    cameras = retention.get("cameras") or {}
    retention_ok = (
        set(cameras) == {"CAM-01", "CAM-04"}
        and all(cameras[camera].get("association_retention_gate") is True
                for camera in ("CAM-01", "CAM-04"))
    )
    drain = result.get("shutdown_drain") or {}
    terminal_frames_ok = (
        drain.get("status") == "PASS"
        and len(drain.get("sources") or []) == 2
        and all(source.get("pgie_frames") == source.get("tracker_frames")
                for source in drain.get("sources") or [])
    )
    max_gap = _max_gap_frames(result)
    return (
        row.get("runner_returncode") == 0
        and result.get("status") == "PASS"
        and result.get("native_exit") == 0
        and result.get("identity_exit") == 0
        and result.get("capture_exit") == 0
        and terminal_frames_ok
        and retention_ok
        and max_gap is not None
        and max_gap <= 10
        and not row.get("shutdown_backtrace_present")
        and not row.get("rtsp_sockets_after")
        and not row.get("candidate_containers_after")
    )


def safe_to_continue_after_failure(row: dict) -> bool:
    """Continue diagnostics only when a failed acceptance run shut down cleanly."""
    result = row.get("result") or {}
    drain = result.get("shutdown_drain") or {}
    return (
        result.get("native_exit") == 0
        and result.get("identity_exit") == 0
        and result.get("capture_exit") == 0
        and drain.get("status") == "PASS"
        and not row.get("shutdown_backtrace_present")
        and not row.get("rtsp_sockets_after")
        and not row.get("candidate_containers_after")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")
    output_root = args.output_root.resolve()
    output_root.relative_to(ROOT / ".runtime")
    output_root.mkdir(parents=True, exist_ok=False)

    native_build = args.native_build.resolve()
    if not native_build.is_file():
        raise SystemExit(f"missing native build record: {native_build}")

    rows: list[dict] = []
    for index in range(1, args.cycles + 1):
        cycle_dir = output_root / f"cycle-{index:02d}"
        command = [
            sys.executable, "-B", "-m", "scripts.run_v13_room_candidate",
            "--dataset", "live",
            "--duration", str(args.duration),
            "--shutdown-debug",
            "--native-build", str(native_build),
            "--output", str(cycle_dir),
        ]
        started = time.time()
        proc = subprocess.run(command, cwd=ROOT)
        result_path = cycle_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        row = {
            "cycle": index,
            "started_epoch": started,
            "runner_returncode": proc.returncode,
            "result": result,
            "shutdown_backtrace_present": (cycle_dir / "logs/shutdown-backtrace.log").exists(),
            "rtsp_sockets_after": _rtsp_sockets(),
            "candidate_containers_after": _candidate_containers(),
        }
        row["max_gap_frames"] = _max_gap_frames(result or {})
        row["pass"] = cycle_pass(row)
        rows.append(row)
        (output_root / "teardown-cycles.json").write_text(
            json.dumps({
                "required_cycles": args.cycles,
                "completed_cycles": len(rows),
                "status": "PASS" if len(rows) == args.cycles and all(r["pass"] for r in rows) else "FAIL",
                "cycles": rows,
            }, indent=2) + "\n"
        )
        print(json.dumps({
            "cycle": index,
            "pass": row["pass"],
            "native_exit": (result or {}).get("native_exit"),
            "status": (result or {}).get("status"),
            "shutdown_backtrace_present": row["shutdown_backtrace_present"],
            "max_gap_frames": row["max_gap_frames"],
            "rtsp_sockets_after": len(row["rtsp_sockets_after"]),
            "candidate_containers_after": row["candidate_containers_after"],
            "continuing_diagnostics_after_acceptance_failure": (
                not row["pass"] and safe_to_continue_after_failure(row)),
        }), flush=True)
        if not row["pass"] and not safe_to_continue_after_failure(row):
            raise SystemExit(1)

    campaign_pass = len(rows) == args.cycles and all(row["pass"] for row in rows)
    print(json.dumps({
        "status": "PASS" if campaign_pass else "FAIL",
        "cycles": len(rows),
        "required_cycles": args.cycles,
        "evidence": str(output_root / "teardown-cycles.json"),
    }), flush=True)
    if not campaign_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
