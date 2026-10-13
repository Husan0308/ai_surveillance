"""Single supervised owner for six-camera local YOLO/NvDCF plus services/UI.

The scoped MV3DT/global-identity service intentionally remains disabled here;
the canonical local tracker feed and current-frame preview remain active.
"""
from __future__ import annotations

from dataclasses import asdict
import errno
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

from services.shared.deployment import load_deployment
from services.shared.runtime_python import preflight_python

ROOT = Path(__file__).resolve().parents[1]


def application_environment(base: dict[str, str], deployment, output: Path) -> dict[str, str]:
    """Resolve child service endpoints and point ML at the native owner telemetry."""
    env = dict(base, **deployment.environment(), SURVEILLANCE_ANALYTICS_ENABLED="0",
        V11_MONITORING_TELEMETRY_PATH=str(output / "analytics" / "monitoring.json"),
        V11_TRACKED_OBJECT_FEED_DIR="/dev/shm",
        FRONTEND_USE_V11_SHARED_MEMORY="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def lowconf_verifier_enabled(base: dict[str, str]) -> bool:
    """Enable the proven fail-closed verifier in the live candidate by default.

    This verifier only sees 0.23 <= confidence < 0.25 sidecar proposals. It
    does not alter the >=0.25 proposal set sent to NvDCF. An explicit `0` is
    retained for controlled diagnostics, not as the production default.
    """
    value = base.get("FULL_STACK_LOWCONF_VERIFIER", "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("FULL_STACK_LOWCONF_VERIFIER must be a boolean value")


def yolo_context_verifier_enabled(base: dict[str, str]) -> bool:
    """Opt into candidate-local frozen-YOLO verification for this run only."""
    value = base.get("FULL_STACK_YOLO_CONTEXT_VERIFIER", "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("FULL_STACK_YOLO_CONTEXT_VERIFIER must be a boolean value")


def active_track_verifier_enabled(base: dict[str, str]) -> bool:
    """Opt into fail-closed track-birth verification for an isolated detector candidate."""
    value = base.get("FULL_STACK_ACTIVE_TRACK_VERIFIER", "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("FULL_STACK_ACTIVE_TRACK_VERIFIER must be a boolean value")


def frontend_evidence_args(output: Path, *, grid_only: bool) -> list[str]:
    """Build the opt-in UI evidence command without changing normal UI startup."""
    args = ["-m", "scripts.validate_production_wall", "--output", str(output / "ui.jsonl")]
    if grid_only:
        args.append("--grid-only")
    return args


def verifier_modes(base: dict[str, str]) -> tuple[bool, bool, bool]:
    """Select low-confidence, combined YOLO-context, or active-only verification."""
    use_yolo_context = yolo_context_verifier_enabled(base)
    use_active_track = active_track_verifier_enabled(base)
    if use_yolo_context and use_active_track:
        raise ValueError("choose only one YOLO context-verifier runtime mode")
    use_lowconf = False if (use_yolo_context or use_active_track) else lowconf_verifier_enabled(base)
    return use_lowconf, use_yolo_context, use_active_track


def child_exit_class(return_code: int | None, requested_signal: int | None) -> str:
    """Classify clean exit, expected supervisor TERM, or an unexpected child failure."""
    if return_code == 0:
        return "success"
    if requested_signal is not None and return_code == -requested_signal:
        return "expected_signal"
    return "failure"


def spawn_owned_child(command: list[str], *, cwd: Path, env: dict[str, str], stdout):
    """Start a supervised child outside the terminal's signal process group.

    Ctrl-C is delivered to the supervisor's foreground process group. Keeping
    children in their own sessions prevents the terminal SIGINT from reaching
    them before the supervisor can perform its ordered SIGTERM shutdown.
    """
    return subprocess.Popen(
        command, cwd=cwd, env=env, stdout=stdout, stderr=subprocess.STDOUT,
        close_fds=True, start_new_session=True,
    )


def verified_analytics_sigterm_shutdown(
    output: Path, return_code: int | None, requested_signal: int | None
) -> bool:
    """Accept Docker's TERM status only with explicit single-signal cleanup evidence."""
    docker_term_status = 128 + signal.SIGTERM
    if requested_signal != signal.SIGTERM or return_code != docker_term_status:
        return False
    try:
        exit_row = json.loads((output / "analytics" / "exit.json").read_text())
        native_log = (output / "analytics" / "native.log").read_text(errors="replace")
    except (OSError, ValueError):
        return False
    if (exit_row.get("return_code") != docker_term_status
            or exit_row.get("signal_received") != [signal.SIGTERM]
            or not isinstance(exit_row.get("native_exit_monotonic_ns"), int)):
        return False

    ended_sources = {
        line.split(" ", 1)[0]
        for line in native_log.splitlines()
        if line.startswith("CAM-") and " END " in line
    }
    group_end = [line for line in native_log.splitlines() if line.startswith("GROUP END ")]
    return (
        ended_sources == {f"CAM-{index:02d}" for index in range(1, 7)}
        and len(group_end) == 1
        and "fatal=0" in group_end[0].split()
        and "shared_errors=0" in group_end[0].split()
    )


def check_ports(deployment) -> None:
    for host, port in ((deployment.api_host, deployment.api_port), (deployment.ml_host, deployment.ml_port)):
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        connect_host = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}.get(host, host)
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                result = probe.connect_ex((connect_host, port))
            except OSError as exc:
                raise RuntimeError(f"service address {host}:{port} occupancy is inconclusive; no process was stopped") from exc
        if result == 0:
            raise RuntimeError(f"service address {host}:{port} is occupied; no process was stopped")
        if result != errno.ECONNREFUSED:
            raise RuntimeError(f"service address {host}:{port} occupancy is inconclusive; no process was stopped")


def main() -> int:
    python = preflight_python(ROOT)
    deployment = load_deployment()  # Resolve before credential dotenv loading.
    check_ports(deployment)
    from services.ml_service.app.config import load_settings
    from services.camera_v11.source_ownership import SourceOwnership
    cameras = load_settings().cameras
    if tuple(c.camera_id for c in cameras) != tuple(f"CAM-{i:02d}" for i in range(1, 7)):
        raise ValueError("full stack requires the canonical six enabled cameras")
    if any(not c.username or not c.password for c in cameras):
        raise ValueError("authorized camera credentials are missing")
    locks = ROOT / ".runtime/camera-owner-locks"
    with SourceOwnership(tuple(c.camera_id for c in cameras), locks):
        pass  # No RTSP is opened during preflight; producers acquire lifetime locks.
    platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    subprocess.run(["docker", "image", "inspect", platform["image"]], stdout=subprocess.DEVNULL, check=True)
    if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        raise RuntimeError("a real display is required for the production wall")
    lock_file = open("/tmp/ai_surveillance_full_live_stack.lock", "a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another full stack supervisor is running") from exc
    output = Path(os.getenv("FULL_STACK_OUT", str(ROOT / ".runtime" / time.strftime("full-stack-%Y%m%dT%H%M%SZ", time.gmtime())))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    started = time.monotonic_ns()
    env = application_environment(os.environ, deployment, output)
    if os.getenv("FULL_STACK_UI_EVIDENCE"):
        env["MV3DT_PREVIEW_LATENCY_LOG"] = str(output / "ui-frame-timings.jsonl")
    from scripts.build_six_camera_v13_wall_candidate import CANDIDATE as DEFAULT_ANALYTICS_CANDIDATE
    analytics_candidate = Path(os.getenv("FULL_STACK_ANALYTICS_CANDIDATE", str(DEFAULT_ANALYTICS_CANDIDATE))).resolve()
    if not analytics_candidate.is_relative_to(ROOT / ".runtime"):
        raise ValueError("analytics candidate must remain under .runtime")
    from scripts.run_six_camera_v13_wall_candidate import validate_candidate
    analytics_manifest = validate_candidate(analytics_candidate)
    use_lowconf_verifier, use_yolo_context_verifier, use_active_track_verifier = verifier_modes(os.environ)
    if use_lowconf_verifier:
        from scripts.run_six_camera_v13_wall_candidate import validate_lowconf_verifier_artifacts
        validate_lowconf_verifier_artifacts(analytics_candidate, analytics_manifest)
    if use_yolo_context_verifier:
        from scripts.run_six_camera_v13_wall_candidate import validate_yolo_context_verifier
        validate_yolo_context_verifier(analytics_candidate, analytics_manifest)
    if use_active_track_verifier:
        from scripts.run_six_camera_v13_wall_candidate import validate_yolo_context_verifier
        if analytics_manifest.get("detector_profile", {}).get("name") != "yolo26m-fp16-896-context-verified":
            raise RuntimeError("active-track verifier mode requires the isolated 896 detector candidate")
        validate_yolo_context_verifier(analytics_candidate, analytics_manifest)
    (output / "launch.json").write_text(json.dumps({"deployment": asdict(deployment),
        "session_start_ns": started, "session_id": output.name, "python": str(python),
        "analytics_enabled": True, "analytics_candidate": str(analytics_candidate),
        "detector_profile": analytics_manifest.get("detector_profile", {}).get(
            "name", "yolo26m-fp16-640-frozen"),
        "analytics_binary_sha256": analytics_manifest["binary_sha256"],
        "detector_engine_sha256": (analytics_manifest.get("detector_profile", {}).get("engine_sha256")
                                    or analytics_manifest["model_sha256"]["yolo26m_raw_otm_b6_fp16.engine"]),
        "yolo_engine_sha256": analytics_manifest["model_sha256"]["yolo26m_raw_otm_b6_fp16.engine"],
        "local_tracker_sha256": analytics_manifest["tracker_config_sha256"],
        "lowconf_verifier_enabled": use_lowconf_verifier,
        "yolo_context_verifier_enabled": use_yolo_context_verifier,
        "active_track_verifier_enabled": use_active_track_verifier,
        "lowconf_verifier_engine_sha256": (
            analytics_manifest.get("lowconf_verifier", {}).get("engine_sha256")
            if use_lowconf_verifier else None),
        "yolo_context_verifier_engine_sha256": (
            analytics_manifest.get("yolo_context_verifier", {}).get("engine_sha256")
            if use_yolo_context_verifier else None),
        "port_8000_owner": "AutoMagicCalib",
        "surveillance_api_port": deployment.api_port, "surveillance_ml_port": deployment.ml_port,
        "reason": "all six RTSP sources are owned by one DeepStream YOLO26m/local-NvDCF process; AMC ownership preserved"}, indent=2) + "\n")
    children: dict[str, subprocess.Popen] = {}
    logs = []
    forced_killed_children: list[str] = []
    requested_shutdown_signals: dict[str, int] = {}
    running = True

    def stop(*_):
        nonlocal running
        running = False

    def spawn(name: str, args: list[str]) -> None:
        log = (output / "logs" / f"{name}.log").open("x")
        logs.append(log)
        children[name] = spawn_owned_child(
            [str(python), "-B", "-u", *args], cwd=ROOT, env=env, stdout=log)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    result = 0
    try:
        spawn("analytics", ["-m", "scripts.run_six_camera_v13_wall_candidate",
            "--candidate", str(analytics_candidate), "--output", str(output / "analytics"),
            "--duration", "86400"] + (["--yolo-context-verifier"] if use_yolo_context_verifier else
                                      ["--active-track-verifier"] if use_active_track_verifier else
                                      ["--lowconf-verifier"] if use_lowconf_verifier else []))
        spawn("ml", ["-m", "services.ml_service.app.main"])
        spawn("api", ["-m", "services.api_service.app.main"])
        # UI starts while connecting; it does not wait on analytics or camera readiness.
        if os.getenv("FULL_STACK_UI_EVIDENCE"):
            spawn("frontend", frontend_evidence_args(
                output, grid_only=os.getenv("FULL_STACK_UI_GRID_ONLY") == "1"))
        else:
            spawn("frontend", ["-m", "services.frontend.app.main"])
        (output / "processes.json").write_text(json.dumps({k: p.pid for k, p in children.items()}, indent=2))
        deadline = time.monotonic() + 90
        ready_written = False
        while running:
            failed = [k for k, p in children.items() if p.poll() is not None]
            if failed:
                if failed == ["frontend"] and children["frontend"].returncode == 0:
                    break
                raise RuntimeError(f"owned child exited: {failed}")
            if not ready_written:
                try:
                    ml = httpx.get(deployment.ml_service_url + "/health", timeout=1).json()
                    api = httpx.get(deployment.frontend_api_base_url + "/health", timeout=1).json()
                    if ml.get("status") == api.get("status") == "ok" and ml.get("online_camera_count") == 6:
                        (output / "ready.json").write_text(json.dumps({"monotonic_ns": time.monotonic_ns(),
                            "ml": ml, "api": api}, indent=2))
                        ready_written = True
                except (httpx.HTTPError, ValueError):
                    pass
                if not ready_written and time.monotonic() > deadline:
                    raise RuntimeError("six fresh advancing producers / service health did not become ready")
            time.sleep(0.2)
    except Exception as exc:
        print(f"FULL_STACK FAIL: {exc}", file=sys.stderr, flush=True)
        result = 1
    finally:
        # Stop consumers first. TERM reaches docker's attached process and its
        # worker, which owns and closes each decoder before releasing locks.
        for name in ("frontend", "api", "ml", "analytics"):
            proc = children.get(name)
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            requested_shutdown_signals[name] = signal.SIGTERM
            try:
                proc.wait(timeout=45 if name == "analytics" else 15)
            except subprocess.TimeoutExpired:
                proc.kill()
                forced_killed_children.append(name)
                proc.wait(timeout=5)
                result = 1
        child_exit_codes = {name: proc.poll() for name, proc in children.items()}
        nonzero_children = {name: code for name, code in child_exit_codes.items() if code != 0}
        expected_signal_children = {
            name: code for name, code in child_exit_codes.items()
            if child_exit_class(code, requested_shutdown_signals.get(name)) == "expected_signal"
        }
        orderly_analytics_sigterm = (
            result == 0
            and "analytics" not in expected_signal_children
            and verified_analytics_sigterm_shutdown(
                output, child_exit_codes.get("analytics"),
                requested_shutdown_signals.get("analytics"))
        )
        if orderly_analytics_sigterm:
            expected_signal_children["analytics"] = child_exit_codes["analytics"]
        unexpected_children = {
            name: code for name, code in child_exit_codes.items()
            if child_exit_class(code, requested_shutdown_signals.get(name)) == "failure"
            and not (name == "analytics" and orderly_analytics_sigterm)
        }
        if unexpected_children:
            result = 1
        for log in logs:
            log.close()
        (output / "shutdown.json").write_text(json.dumps({
            "status": "PASS" if result == 0 else "FAIL",
            "children": child_exit_codes,
            "nonzero_children": nonzero_children,
            "requested_shutdown_signals": requested_shutdown_signals,
            "expected_signal_children": expected_signal_children,
            "verified_orderly_analytics_sigterm": orderly_analytics_sigterm,
            "unexpected_children": unexpected_children,
            "forced_kill": bool(forced_killed_children),
            "forced_killed_children": forced_killed_children,
            "monotonic_ns": time.monotonic_ns()}, indent=2))
        lock_file.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
