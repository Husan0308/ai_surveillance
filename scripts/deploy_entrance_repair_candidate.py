#!/usr/bin/env python3
"""Perform one guarded r6 -> Entrance-repair candidate restart with rollback.

This operator tool only replaces the exact known r6 owner. It never opens a
camera connection itself, never force-kills a process, and restores r6 if the
new owner cannot satisfy the startup/live-health checks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_entrance_calibration_r6 import (
    CANDIDATE as R6_CANDIDATE,
    R6_SHA256,
    containers,
    established_rtsp,
    port_is_free,
    port_listener_signature,
    proc_alive,
    stack_launch,
    supervisors,
    verify_r6_owner,
    wait_old_owner_stopped,
)
from scripts.run_six_camera_v13_wall_candidate import (
    docker_command,
    validate_candidate,
    validate_yolo_context_verifier,
)

CANDIDATE = ROOT / ".runtime/sixcam-entrance-repairs-20261009-r1"
CANDIDATE_SHA256 = "430aa23a49138b8b916abb350d1831c74abe16bbdb4572fa68759fe94b457d1f"
ENGINE_SHA256 = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
PARSER_SHA256 = "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
TRACKER_SHA256 = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
V13_SHA256 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
CAMERAS = tuple(f"CAM-{number:02d}" for number in range(1, 7))
LOCAL_FIXTURE_PORT = 18554
KEEP_ENV = {
    "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS",
    "FULL_STACK_UI_EVIDENCE", "FULL_STACK_UI_GRID_ONLY",
    "FULL_STACK_LOWCONF_VERIFIER", "FULL_STACK_YOLO_CONTEXT_VERIFIER",
    "FULL_STACK_ACTIVE_TRACK_VERIFIER", "V11_CALIBRATION_RECORDING_ENABLED",
    "V11_MUX_BATCHED_PUSH_TIMEOUT_US", "V11_MUX_WIDTH", "V11_MUX_HEIGHT",
    "V11_TRACKER_WIDTH", "V11_TRACKER_HEIGHT", "V11_VERBOSE_EVIDENCE",
    "V11_TRACKER_NAN_DIAGNOSTICS", "V11_TRACKER_DECISION_DIAGNOSTICS",
    "V11_TRACKER_FRAME_STATE_DIAGNOSTICS",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def proc_environment(pid: int) -> dict[str, str]:
    rows = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    result: dict[str, str] = {}
    for row in rows:
        if b"=" not in row:
            continue
        key, value = row.split(b"=", 1)
        name = key.decode(errors="replace")
        if name in KEEP_ENV:
            result[name] = value.decode(errors="replace")
    return result


def protected_hashes() -> dict[str, str]:
    paths = {
        "primary_engine": CANDIDATE / "models/yolo26m_raw_otm_b6_fp16.engine",
        "primary_parser": CANDIDATE / "models/libyolo26rawperson.so",
        "auxiliary_engine": CANDIDATE / "models/yolo26m_raw_otm_768_b6_fp16.engine",
        "local_tracker": CANDIDATE / "config/config_tracker_NvDCF_v13_local6.yml",
        "frozen_v13": ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml",
        "dev_room_cam_00": ROOT / "config/mv3dt_dev_room/camInfo/cam_00.yml",
        "dev_room_cam_01": ROOT / "config/mv3dt_dev_room/camInfo/cam_01.yml",
    }
    return {name: digest(path) for name, path in paths.items()}


def read_http_json(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        return response.status, json.loads(response.read())


def service_health() -> dict:
    ml_status, ml = read_http_json("http://127.0.0.1:8101/health")
    api_status, api = read_http_json("http://127.0.0.1:8100/health")
    camera_status, cameras = read_http_json("http://127.0.0.1:8100/api/v1/cameras")
    rows = cameras.get("cameras", [])
    return {
        "http_status": {"ml": ml_status, "api": api_status, "cameras": camera_status},
        "ml": {key: ml.get(key) for key in
               ("service", "status", "analytics_enabled", "monitoring_status",
                "camera_count", "online_camera_count")},
        "api": {key: api.get(key) for key in ("service", "status", "ml_status")},
        "cameras": [{key: row.get(key) for key in
                     ("camera_id", "online", "pipeline_errors", "reconnects")}
                    for row in rows],
    }


def healthy_service_summary(summary: dict) -> bool:
    return (
        summary["ml"].get("status") == "ok"
        and summary["ml"].get("online_camera_count") == 6
        and summary["api"].get("status") == "ok"
        and summary["api"].get("ml_status") == "ok"
        and [row.get("camera_id") for row in summary["cameras"]] == list(CAMERAS)
        and all(row.get("online") is True and not row.get("pipeline_errors", 0)
                for row in summary["cameras"])
    )


def socket_owner_summary() -> dict:
    rows = established_rtsp()
    owners: list[int] = []
    for row in rows:
        match = re.search(r"pid=(\d+)", row)
        if match:
            owners.append(int(match.group(1)))
    return {
        "count": len(rows),
        "owner_pids": sorted(set(owners)),
        "one_owner": len(rows) == 6 and len(owners) == 6 and len(set(owners)) == 1,
    }


def file_inventory(root: Path) -> list[dict]:
    rows = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                rows.append({"path": str(path.relative_to(ROOT)), "size_bytes": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns})
    return rows


def verify_candidate_provenance() -> dict:
    if digest(R6_CANDIDATE / "bin/cam-six-yolo-local6-posttracker") != R6_SHA256:
        raise RuntimeError("r6 rollback executable hash mismatch")
    manifest = validate_candidate(CANDIDATE)
    validate_yolo_context_verifier(CANDIDATE, manifest)
    binary = CANDIDATE / "bin/cam-six-yolo-local6-posttracker"
    if digest(binary) != CANDIDATE_SHA256 or manifest.get("binary_sha256") != CANDIDATE_SHA256:
        raise RuntimeError("repair candidate executable hash mismatch")
    profile = manifest.get("detector_profile", {})
    if (profile.get("name") != "yolo26m-fp16-640-primary-768-aux-interval-19-context-verified"
            or profile.get("batch_size") != 6 or profile.get("aux_interval") != 19
            or profile.get("primary_engine_sha256") != ENGINE_SHA256
            or profile.get("primary_parser_sha256") != PARSER_SHA256
            or manifest.get("tracker_config_sha256") != TRACKER_SHA256
            or manifest.get("frozen_v13_reference_sha256") != V13_SHA256):
        raise RuntimeError("repair candidate model/profile/tracker provenance mismatch")
    return manifest


def capture_preflight(supervisor: dict, old_output: Path, old_launch: dict,
                      protected: dict, evidence_dir: Path) -> dict:
    current_health = service_health()
    if not healthy_service_summary(current_health):
        raise RuntimeError("pre-restart ML/API/six-camera health is not clean")
    sockets = socket_owner_summary()
    if not sockets["one_owner"]:
        raise RuntimeError("pre-restart does not show exactly six RTSP sockets owned by one PID")
    amc = port_listener_signature(8000)
    if not amc:
        raise RuntimeError("AutoMagicCalib listener on port 8000 is not visible")
    if len(containers()) != 1:
        raise RuntimeError("pre-restart must have exactly one six-camera analytics container")
    child_manifest = json.loads((old_output / "processes.json").read_text())
    analytics_run = old_output / "analytics"
    analytics_launch = json.loads((analytics_run / "launch.json").read_text())
    docker_args = subprocess.run(
        ["docker", "inspect", "--format", "{{.Path}} {{json .Args}}", "surv-sixcam-analytics"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if (old_launch.get("analytics_candidate") != str(R6_CANDIDATE.resolve())
            or old_launch.get("analytics_binary_sha256") != R6_SHA256
            or analytics_launch.get("binary_sha256") != R6_SHA256):
        raise RuntimeError("running owner is not the expected r6 rollback parent")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    value = {
        "status": "PREFLIGHT_PASS",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "current_supervisor_pid": supervisor["pid"],
        "current_run_dir": str(old_output),
        "current_candidate": str(R6_CANDIDATE),
        "current_binary_sha256": R6_SHA256,
        "old_startup_manifest": old_launch,
        "old_analytics_manifest": analytics_launch,
        "old_child_pids": child_manifest,
        "old_native_command_redacted": docker_args,
        "safe_runtime_environment": proc_environment(supervisor["pid"]),
        "health": current_health,
        "rtsp": sockets,
        "container_count": len(containers()),
        "ports": {"api_8100": "occupied by surveillance", "ml_8101": "occupied by surveillance",
                  "amc_8000_listener_signature": list(amc)},
        "protected_hashes": protected,
        "calibration_inventory": file_inventory(ROOT / "config/mv3dt_dev_room"),
        "recording_inventory": file_inventory(ROOT / "calibration_recordings"),
        "rollback": {"candidate": str(R6_CANDIDATE), "binary_sha256": R6_SHA256,
                     "candidate_unchanged": True, "startup_command": "bash scripts/start_full_live_stack.sh"},
        "new_candidate": {"path": str(CANDIDATE),
                          "binary_sha256": CANDIDATE_SHA256,
                          "direct_file_binary_sha256": json.loads(
                              (CANDIDATE / "build-manifest.json").read_text())[
                                  "direct_file_binary_sha256"]},
    }
    (evidence_dir / "preflight.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def wait_ports_released(timeout_sec: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if (not containers() and not established_rtsp()
                and port_is_free(8100) and port_is_free(8101)):
            return
        time.sleep(0.25)
    raise TimeoutError("old analytics container, RTSP sockets, or 8100/8101 did not release")


def local_fixture_sessions() -> list[str]:
    result = subprocess.run(
        ["ss", "-Htnp", "state", "established", "(", "dport", "=", f":{LOCAL_FIXTURE_PORT}", ")"],
        text=True, capture_output=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def write_local_fixture_camera_config(path: Path) -> None:
    uri = f"rtsp://127.0.0.1:{LOCAL_FIXTURE_PORT}/synthetic"
    encoded_uri = base64.b64encode(uri.encode()).decode()
    encoded_empty = base64.b64encode(b"").decode()
    lines: list[str] = []
    for camera in CAMERAS:
        lines.extend((f"[{camera}]", f"uri={encoded_uri}",
                      f"username={encoded_empty}", f"password={encoded_empty}",
                      "latency_ms=20", "drop_on_latency=false",
                      "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=true", "preview_port=5600", ""))
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)


def run_offline_recovery_fixture(evidence_dir: Path) -> dict:
    """Fault-inject only a local synthetic RTSP stream using the live binary."""
    from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader

    output = evidence_dir / "offline-recovery-fixture"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite recovery fixture evidence: {output}")
    output.mkdir(parents=True)
    camera_file = output / "camera-six-local-fixture.ini"
    write_local_fixture_camera_config(camera_file)
    if local_fixture_sessions():
        raise RuntimeError("local fixture port already has RTSP clients")
    port_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        port_probe.settimeout(0.2)
        if port_probe.connect_ex(("127.0.0.1", LOCAL_FIXTURE_PORT)) == 0:
            raise RuntimeError("local fixture port is already occupied")
    finally:
        port_probe.close()

    fixture_log_path = output / "fixture-server.log"
    native_log_path = output / "native.log"
    fixture_log = fixture_log_path.open("x", encoding="utf-8")
    fixture = subprocess.Popen(
        ["/usr/bin/python3", "scripts/local_rtsp_test_fixture.py",
         "--port", str(LOCAL_FIXTURE_PORT)], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=fixture_log, stderr=subprocess.STDOUT,
        close_fds=True, start_new_session=True,
    )
    native: subprocess.Popen | None = None
    readers = {camera: PreviewFrameReader(
        f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin")
        for camera in CAMERAS}
    progress: dict[str, dict] = {}
    max_local_clients = 0
    local_owner_pids: set[int] = set()
    old_environment = {key: os.environ.get(key) for key in (
        "V11_SOURCE_STALL_RECOVERY", "V11_TEST_SOURCE_STALL_PLAN",
        "V11_CALIBRATION_RECORDING_ENABLED", "V11_TRACKER_NAN_DIAGNOSTICS",
        "V11_MUX_BATCHED_PUSH_TIMEOUT_US", "V11_VERBOSE_EVIDENCE")}
    try:
        ready_deadline = time.monotonic() + 8
        while time.monotonic() < ready_deadline:
            if fixture.poll() is not None:
                raise RuntimeError("local synthetic RTSP fixture exited during startup")
            if "LOCAL_RTSP_FIXTURE_READY" in fixture_log_path.read_text(errors="replace"):
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("local synthetic RTSP fixture did not become ready")

        os.environ.update({
            "V11_SOURCE_STALL_RECOVERY": "1",
            "V11_TEST_SOURCE_STALL_PLAN": "CAM-05@30",
            "V11_CALIBRATION_RECORDING_ENABLED": "0",
            "V11_TRACKER_NAN_DIAGNOSTICS": "1",
            "V11_MUX_BATCHED_PUSH_TIMEOUT_US": "1000",
            "V11_VERBOSE_EVIDENCE": "0",
        })
        command = docker_command(CANDIDATE, output, camera_file, 75,
                                 yolo_context_verifier=True)
        write_json(output / "probe-command.json", {
            "candidate": str(CANDIDATE), "candidate_binary_sha256": CANDIDATE_SHA256,
            "profile": "yolo26m-fp16-640-primary-768-aux-interval-19-context-verified",
            "source_recovery_enabled": True,
            "fault_plan": "CAM-05@30",
            "input": "six source IDs connected only to one localhost synthetic RTSP server",
            "production_camera_connections": 0,
            "container_network_mode": "host, URI restricted to 127.0.0.1:18554",
            "duration_sec": 75,
            "command_redacted": ["docker run", "--network=host", "<candidate mounts>",
                                 "<localhost-only camera config>", "75"],
        })
        with native_log_path.open("x", encoding="utf-8") as log:
            native = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT,
                                      close_fds=True, start_new_session=True)
            started = time.monotonic()
            while native.poll() is None:
                rows = local_fixture_sessions()
                max_local_clients = max(max_local_clients, len(rows))
                for row in rows:
                    match = re.search(r"pid=(\d+)", row)
                    if match:
                        local_owner_pids.add(int(match.group(1)))
                if len(local_owner_pids) > 1:
                    raise RuntimeError("more than one native local-fixture client process appeared")
                for camera, reader in readers.items():
                    frame = reader.read_latest(max_age_sec=1.0, metadata_only=True)
                    if frame is None or not frame.tracker_snapshot:
                        continue
                    snapshot = frame.tracker_snapshot
                    matched = (
                        int(snapshot.get("pts_ns", -1)) == frame.pts_ns
                        and int(snapshot.get("source_frame_num", -1)) == frame.source_frame_num
                        and int(snapshot.get("frame_num", -1)) == frame.tracker_frame_num
                    )
                    if not matched:
                        raise RuntimeError(f"{camera} offline recovery preview/metadata mismatch")
                    prior = progress.get(camera)
                    progress[camera] = {
                        "sequence": frame.sequence, "pts_ns": frame.pts_ns,
                        "source_frame_num": frame.source_frame_num,
                        "tracker_frame_num": frame.tracker_frame_num,
                        "sync": True,
                    }
                    if prior and (frame.sequence <= prior["sequence"]
                                  or frame.tracker_frame_num <= prior["tracker_frame_num"]):
                        # The only permitted non-advance is an unchanged latest-only read.
                        if frame.sequence != prior["sequence"]:
                            raise RuntimeError(f"{camera} preview sequence regressed")
                if time.monotonic() - started > 180:
                    raise TimeoutError("offline native recovery fixture exceeded 180 seconds")
                time.sleep(0.1)
        native_return = native.returncode
        native_log = native_log_path.read_text(errors="replace")
        if native_return != 0 or "GROUP END" not in native_log or "fatal=0" not in native_log:
            raise RuntimeError("offline native source-recovery fixture did not exit cleanly")

        event_names = ("TEST_SOURCE_STALL_BEGIN", "SOURCE_STALL_RECOVERY_BEGIN",
                       "SOURCE_STALL_OLD_OWNER_REMOVED", "SOURCE_STALL_RECOVERY_RECONNECTED",
                       "SOURCE_STALL_RECOVERY_PASS")
        event_rows = []
        for line in native_log.splitlines():
            for name in event_names:
                if name in line:
                    event_rows.append({"camera_id": line.split(maxsplit=1)[0],
                                       "event": name, "line": line})
                    break
        cam05_events = [row["event"] for row in event_rows if row["camera_id"] == "CAM-05"]
        if cam05_events != list(event_names):
            raise RuntimeError(f"CAM-05 offline recovery event order incomplete: {cam05_events}")
        old_removed = next(row["line"] for row in event_rows
                           if row["event"] == "SOURCE_STALL_OLD_OWNER_REMOVED")
        reconnected = next(row["line"] for row in event_rows
                           if row["event"] == "SOURCE_STALL_RECOVERY_RECONNECTED")
        if "owner_count=0" not in old_removed or "owner_count=1" not in reconnected:
            raise RuntimeError("offline recovery did not prove old-source removal before one replacement")
        if max_local_clients > 6 or not local_owner_pids:
            raise RuntimeError("offline fixture client count/ownership was invalid")
        metrics_path = output / "monitoring.json"
        monitoring = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        camera_rows = monitoring.get("cameras", [])
        if isinstance(camera_rows, dict):
            camera_rows = list(camera_rows.values())
        rows_by_camera = {row.get("camera_id"): row for row in camera_rows}
        if set(rows_by_camera) != set(CAMERAS):
            raise RuntimeError("offline recovery telemetry did not contain all six sources")
        cam05 = rows_by_camera["CAM-05"]
        if (int(cam05.get("source_stall_recoveries", 0)) != 1
                or int(cam05.get("tracking_epoch", 0)) < 1):
            raise RuntimeError("CAM-05 source-recovery telemetry or epoch did not advance")
        if set(progress) != set(CAMERAS):
            raise RuntimeError(f"offline fixture did not publish synchronized preview frames: {sorted(progress)}")
        result = {
            "status": "PASS_OFFLINE_LOCAL_FIXTURE_ONLY",
            "candidate_binary_sha256": CANDIDATE_SHA256,
            "camera_rtsp_disconnections": 0,
            "local_synthetic_rtsp_clients_max": max_local_clients,
            "local_native_client_owner_pids": sorted(local_owner_pids),
            "camera_id_tested": "CAM-05",
            "events": event_rows,
            "monitoring_counters": {camera: {
                key: row.get(key) for key in (
                    "decoded_frames", "pgie_frames_total", "canonical_tracker_updates_total",
                    "preview_frames_total", "rtsp_reconnect_attempts", "source_stall_recoveries",
                    "tracking_epoch", "input_pts_resets", "decoded_pts_resets", "pipeline_errors",
                )
            } for camera, row in rows_by_camera.items()},
            "exact_frame_pts_preview_sync": {camera: {**progress[camera], "mismatch_count": 0}
                                             for camera in CAMERAS},
            "pipeline_errors": {camera: int(row.get("pipeline_errors", 0))
                                for camera, row in rows_by_camera.items()},
            "native_exit": native_return,
            "evidence_dir": str(output),
        }
        write_json(output / "recovery-result.json", result)
        return result
    finally:
        for reader in readers.values():
            reader.close()
        if native is not None and native.poll() is None:
            native.send_signal(signal.SIGTERM)
            try:
                native.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Keep the no-force-kill policy even for the isolated diagnostic.
                raise TimeoutError("offline recovery container did not stop gracefully")
        if fixture.poll() is None:
            fixture.send_signal(signal.SIGTERM)
            try:
                fixture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                raise TimeoutError("local synthetic RTSP fixture did not stop gracefully")
        fixture_log.close()
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and local_fixture_sessions():
            time.sleep(0.1)
        if local_fixture_sessions():
            raise RuntimeError("local synthetic RTSP sessions remained after offline test")


def wait_preview_progress(timeout_sec: float = 50.0) -> dict:
    from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader

    readers = {camera: PreviewFrameReader(
        f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin")
        for camera in CAMERAS}
    previous: dict[str, tuple[int, int, int]] = {}
    latest_rows: dict[str, dict] = {}
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            for camera, reader in readers.items():
                frame = reader.read_latest(max_age_sec=1.0, metadata_only=True)
                if frame is None or not frame.tracker_snapshot:
                    continue
                snap = frame.tracker_snapshot
                same_frame = (
                    int(snap.get("pts_ns", -1)) == frame.pts_ns
                    and int(snap.get("source_frame_num", -1)) == frame.source_frame_num
                    and int(snap.get("frame_num", -1)) == frame.tracker_frame_num
                )
                if not same_frame:
                    raise RuntimeError(f"{camera} preview image/tracker sequence or PTS mismatch")
                objects = snap.get("objects", [])
                if not isinstance(objects, list):
                    raise RuntimeError(f"{camera} embedded canonical object list is invalid")
                ids = [int(obj.get("native_track_id", -1)) for obj in objects
                       if isinstance(obj, dict)]
                if len(ids) != len(objects) or any(native_id < 0 for native_id in ids):
                    raise RuntimeError(f"{camera} canonical tracker ID row is invalid")
                if len(ids) != len(set(ids)):
                    raise RuntimeError(f"{camera} duplicate native ID reached the canonical preview")
                point = (frame.sequence, frame.pts_ns, frame.source_frame_num)
                prior = previous.get(camera)
                if prior is None or (point[0] > prior[0] and point[1] > prior[1]
                                     and point[2] > prior[2]):
                    previous[camera] = point
                    latest_rows[camera] = {
                        "sequence": frame.sequence, "pts_ns": frame.pts_ns,
                        "source_frame_num": frame.source_frame_num,
                        "tracker_frame_num": frame.tracker_frame_num,
                        "width": frame.width, "height": frame.height,
                        "canonical_object_count": len(objects),
                        "canonical_native_ids_unique": True,
                        "age_ms": round((time.monotonic_ns() - frame.timestamp_ns) / 1e6, 3),
                    }
            if len(previous) == 6:
                # Require a second advancement from every camera, not just one fresh slot.
                baseline = dict(previous)
                advanced: set[str] = set()
                second_deadline = time.monotonic() + 4.0
                while time.monotonic() < second_deadline and len(advanced) != 6:
                    for camera, reader in readers.items():
                        if camera in advanced:
                            continue
                        prior = baseline[camera]
                        frame = reader.read_latest(max_age_sec=1.0, after_sequence=prior[0],
                                                   metadata_only=True)
                        if frame is None or not frame.tracker_snapshot:
                            continue
                        snap = frame.tracker_snapshot
                        if (frame.sequence > prior[0] and frame.pts_ns > prior[1]
                                and frame.source_frame_num > prior[2]
                                and int(snap.get("pts_ns", -1)) == frame.pts_ns
                                and int(snap.get("source_frame_num", -1)) == frame.source_frame_num
                                and int(snap.get("frame_num", -1)) == frame.tracker_frame_num):
                            advanced.add(camera)
                            latest_rows[camera] = {
                                "sequence": frame.sequence, "pts_ns": frame.pts_ns,
                                "source_frame_num": frame.source_frame_num,
                                "tracker_frame_num": frame.tracker_frame_num,
                                "width": frame.width, "height": frame.height,
                                "canonical_object_count": len(snap.get("objects", [])),
                                "canonical_native_ids_unique": True,
                                "age_ms": round((time.monotonic_ns() - frame.timestamp_ns) / 1e6, 3),
                            }
                    time.sleep(0.02)
                if len(advanced) == 6:
                    return latest_rows
            time.sleep(0.025)
    finally:
        for reader in readers.values():
            reader.close()
    raise TimeoutError("six fresh exact-frame previews did not advance")


def live_metrics(run_dir: Path) -> dict:
    monitor_path = run_dir / "analytics/monitoring.json"
    try:
        raw = json.loads(monitor_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"valid": False, "reason": "monitoring telemetry missing or malformed"}
    rows = raw.get("cameras", [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    by_camera = {row.get("camera_id"): row for row in rows}
    result = {"valid": set(by_camera) == set(CAMERAS), "cameras": {}}
    for camera in CAMERAS:
        row = by_camera.get(camera, {})
        result["cameras"][camera] = {
            key: row.get(key) for key in (
                "online", "source_fps", "decoded_frames", "analytics_input_frames_total",
                "pgie_frames_total", "canonical_tracker_updates_total", "preview_frames_total",
                "pipeline_errors", "reconnects", "rtsp_reconnect_attempts",
                "source_stall_recoveries", "input_pts_resets", "decoded_pts_resets",
                "decoder_max_interframe_gap_ms", "tracking_epoch", "queue_depth", "queue_high_water",
            )
        }
    return result


def runtime_pass(run_dir: Path, supervisor_pid: int, expected_sha: str,
                 expected_candidate: Path, amc_before: tuple[str, ...],
                 recovery_enabled: bool) -> dict:
    launch = json.loads((run_dir / "launch.json").read_text())
    if (launch.get("analytics_binary_sha256") != expected_sha
            or Path(launch.get("analytics_candidate", "")).resolve() != expected_candidate.resolve()
            or launch.get("yolo_engine_sha256") != ENGINE_SHA256
            or launch.get("local_tracker_sha256") != TRACKER_SHA256
            or launch.get("yolo_context_verifier_enabled") is not True
            or not proc_alive(supervisor_pid)):
        raise RuntimeError("live launch provenance/process does not match the requested binary")
    service = service_health()
    if not healthy_service_summary(service):
        raise RuntimeError("ML/API/camera endpoint health gate failed")
    if port_listener_signature(8000) != amc_before:
        raise RuntimeError("AutoMagicCalib port-8000 listener changed")
    socket_summary = socket_owner_summary()
    if not socket_summary["one_owner"] or len(containers()) != 1:
        raise RuntimeError("live candidate does not have exactly one owner and six RTSP sessions")
    feeds = wait_preview_progress()
    metrics = live_metrics(run_dir)
    if not metrics.get("valid"):
        raise RuntimeError("six-camera source monitoring telemetry is incomplete")
    for camera, row in metrics["cameras"].items():
        if (row.get("online") is not True or float(row.get("source_fps") or 0) < 15.0
                or int(row.get("pipeline_errors") or 0) != 0
                or int(row.get("queue_depth") if row.get("queue_depth") is not None else -1) < 0
                or int(row.get("queue_high_water") if row.get("queue_high_water") is not None else -1) > 64
                or (recovery_enabled and row.get("rtsp_reconnect_attempts") is None)
                or (recovery_enabled and row.get("source_stall_recoveries") is None)
                or (recovery_enabled and row.get("tracking_epoch") is None)):
            raise RuntimeError(f"{camera} live telemetry gate failed: {row}")
    native_log = (run_dir / "analytics/native.log").read_text(errors="replace")
    expected_recovery_line = f"SOURCE_STALL_RECOVERY enabled={1 if recovery_enabled else 0}"
    if expected_recovery_line not in native_log:
        raise RuntimeError("source-stall recovery runtime flag does not match deployment mode")
    return {"launch": launch, "health": service, "rtsp": socket_summary,
            "preview_progress": feeds, "metrics": metrics,
            "amc_listener_unchanged": True}


def spawn_stack(candidate: Path, run_dir: Path, log_path: Path,
                base_environment: dict[str, str], *, recovery: bool,
                amc_before: tuple[str, ...]) -> int:
    env = dict(base_environment)
    env.update({
        "FULL_STACK_OUT": str(run_dir.resolve()),
        "FULL_STACK_ANALYTICS_CANDIDATE": str(candidate.resolve()),
        "FULL_STACK_YOLO_CONTEXT_VERIFIER": "1",
        "FULL_STACK_ACTIVE_TRACK_VERIFIER": "0",
        "FULL_STACK_LOWCONF_VERIFIER": "0",
        "V11_SOURCE_STALL_RECOVERY": "1" if recovery else "0",
        "V11_CALIBRATION_RECORDING_ENABLED": "1",
        "V11_MUX_BATCHED_PUSH_TIMEOUT_US": "1000",
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", "scripts/start_full_live_stack.sh"], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True,
        )
    deadline = time.monotonic() + 150.0
    last_error = "readiness manifest not written"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{candidate.name} supervisor exited with {process.returncode}: {log_path}")
        if (run_dir / "ready.json").is_file() and (run_dir / "launch.json").is_file():
            try:
                launch = json.loads((run_dir / "launch.json").read_text())
                expected_sha = (CANDIDATE_SHA256 if candidate.resolve() == CANDIDATE.resolve()
                                else R6_SHA256)
                if launch.get("analytics_binary_sha256") != expected_sha:
                    raise RuntimeError("started runtime binary hash is not expected")
                runtime_pass(run_dir, process.pid, expected_sha, candidate,
                             amc_before, recovery)
                return process.pid
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise TimeoutError(f"{candidate.name} did not become ready ({last_error})")


def stop_stack(pid: int, run_dir: Path, timeout_sec: float = 75.0) -> None:
    if proc_alive(pid):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and proc_alive(pid):
        time.sleep(0.25)
    if proc_alive(pid):
        raise TimeoutError("stack supervisor failed to stop gracefully; no forced kill attempted")
    shutdown = run_dir / "shutdown.json"
    if not shutdown.is_file():
        raise RuntimeError("stack exited without shutdown evidence")
    result = json.loads(shutdown.read_text())
    if result.get("status") != "PASS" or result.get("forced_kill") is not False:
        raise RuntimeError(f"stack shutdown was not clean: {result}")
    wait_ports_released()


def run(evidence_dir: Path, *, preflight_only: bool, observe_seconds: int) -> dict:
    manifest = verify_candidate_provenance()
    if observe_seconds < 30 or observe_seconds > 1800:
        raise ValueError("observe-seconds must be between 30 and 1800")
    evidence_dir = evidence_dir.resolve()
    if not evidence_dir.is_relative_to(ROOT / ".runtime"):
        raise ValueError("deployment evidence must stay under .runtime")
    running = supervisors()
    if len(running) != 1:
        raise RuntimeError(f"expected exactly one known r6 supervisor; found {len(running)}")
    old_supervisor = running[0]
    old_output, old_launch = verify_r6_owner(old_supervisor)
    protected_before = protected_hashes()
    expected_protected = {
        "primary_engine": ENGINE_SHA256, "primary_parser": PARSER_SHA256,
        "auxiliary_engine": "84832420e6e3ad68e1664b2ceb04b38474459960990c00d76987ca8bbda153be",
        "local_tracker": TRACKER_SHA256, "frozen_v13": V13_SHA256,
    }
    if any(protected_before.get(key) != value for key, value in expected_protected.items()):
        raise RuntimeError("candidate/model/tracker/calibration protected hash preflight failed")
    preflight = capture_preflight(old_supervisor, old_output, old_launch,
                                  protected_before, evidence_dir)
    if preflight_only:
        return {"status": "PREFLIGHT_PASS_NO_RESTART", "evidence_dir": str(evidence_dir),
                "candidate_sha256": CANDIDATE_SHA256, "rollback_sha256": R6_SHA256,
                "candidate_profile": manifest["detector_profile"]["name"]}

    # Read UI/display and runtime options from the verified old supervisor only.
    base_environment = dict(os.environ)
    base_environment.update(proc_environment(old_supervisor["pid"]))
    # The repository/workstation environment can carry legacy 8000/8001
    # values, while the accepted r6 runtime explicitly preserved AMC on 8000
    # and served surveillance on 8100/8101. Rehydrate the exact deployment
    # contract from the captured live launch manifest for both candidate and
    # rollback starts; never infer it from the invoking shell or edit .env.
    deployment = old_launch.get("deployment", {})
    expected_deployment = {
        "api_host": "0.0.0.0", "api_port": 8100,
        "ml_host": "0.0.0.0", "ml_port": 8101,
        "ml_service_url": "http://127.0.0.1:8101",
        "frontend_api_base_url": "http://127.0.0.1:8100",
        "frontend_ml_video_base_url": "http://127.0.0.1:8101",
    }
    if deployment != expected_deployment:
        raise RuntimeError(f"captured r6 deployment contract is not the authorized 8100/8101 profile: {deployment}")
    base_environment.update({
        "API_HOST": deployment["api_host"],
        "API_PORT": str(deployment["api_port"]),
        "ML_HOST": deployment["ml_host"],
        "ML_PORT": str(deployment["ml_port"]),
        "ML_SERVICE_URL": deployment["ml_service_url"],
        "FRONTEND_API_BASE_URL": deployment["frontend_api_base_url"],
        "FRONTEND_ML_VIDEO_BASE_URL": deployment["frontend_ml_video_base_url"],
    })
    amc_before = port_listener_signature(8000)
    rerun = supervisors()
    if len(rerun) != 1 or rerun[0]["pid"] != old_supervisor["pid"]:
        raise RuntimeError("r6 supervisor identity changed during preflight; did not signal it")
    rechecked_output, rechecked_launch = verify_r6_owner(rerun[0])
    if rechecked_output != old_output or rechecked_launch != old_launch:
        raise RuntimeError("r6 owner provenance changed during preflight; did not signal it")

    run_dir = ROOT / ".runtime" / ("full-stack-entrance-repair-" +
                                     datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    log_path = evidence_dir / "candidate-startup.log"
    rollback_dir = ROOT / ".runtime" / ("full-stack-entrance-r6-rollback-" +
                                         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    rollback_log = evidence_dir / "rollback-startup.log"
    deployment: dict = {"preflight": preflight, "candidate_sha256": CANDIDATE_SHA256,
                        "rollback_sha256": R6_SHA256, "offline_tests": "NOT_RUN"}
    candidate_pid: int | None = None
    try:
        os.kill(old_supervisor["pid"], signal.SIGTERM)
        wait_old_owner_stopped(old_supervisor, old_output)
        if port_listener_signature(8000) != amc_before:
            raise RuntimeError("AutoMagicCalib listener changed during clean r6 shutdown")
        wait_ports_released()
        deployment["old_shutdown"] = json.loads((old_output / "shutdown.json").read_text())
        deployment["old_rtsp_sockets_released"] = True
        deployment["api_8100_released"] = port_is_free(8100)
        deployment["ml_8101_released"] = port_is_free(8101)

        # Exercise source-reset/epoch handling on a localhost-only synthetic
        # RTSP fixture while the real cameras are disconnected from the app.
        # The production owner starts only after this exact-candidate test passes.
        offline_recovery = run_offline_recovery_fixture(evidence_dir)
        deployment["offline_tests"] = offline_recovery
        candidate_pid = spawn_stack(CANDIDATE, run_dir, log_path,
                                    base_environment, recovery=True,
                                    amc_before=amc_before)
        deployment["candidate_run_dir"] = str(run_dir)
        deployment["candidate_supervisor_pid"] = candidate_pid
        deployment["candidate_startup"] = "PASS"
        deployment["candidate_initial_health"] = runtime_pass(
            run_dir, candidate_pid, CANDIDATE_SHA256, CANDIDATE, amc_before, True)
        (evidence_dir / "deployment-started.json").write_text(
            json.dumps(deployment, indent=2) + "\n")

        observations = []
        deadline = time.monotonic() + observe_seconds
        next_sample = time.monotonic()
        previous_counters: dict[str, dict[str, int]] | None = None
        while time.monotonic() < deadline:
            if not proc_alive(candidate_pid):
                raise RuntimeError("candidate supervisor exited during live observation")
            snapshot = runtime_pass(run_dir, candidate_pid, CANDIDATE_SHA256,
                                    CANDIDATE, amc_before, True)
            metrics = snapshot["metrics"]["cameras"]
            if previous_counters is not None:
                for camera in CAMERAS:
                    before = previous_counters[camera]
                    now = metrics[camera]
                    for key in ("decoded_frames", "pgie_frames_total",
                                "canonical_tracker_updates_total", "preview_frames_total"):
                        if int(now.get(key) or 0) <= before[key]:
                            raise RuntimeError(f"{camera} {key} stopped advancing")
            previous_counters = {
                camera: {key: int(metrics[camera].get(key) or 0) for key in
                         ("decoded_frames", "pgie_frames_total", "canonical_tracker_updates_total",
                          "preview_frames_total")}
                for camera in CAMERAS
            }
            if time.monotonic() >= next_sample:
                observations.append({"monotonic_ns": time.monotonic_ns(),
                                     "metrics": snapshot["metrics"],
                                     "rtsp": snapshot["rtsp"],
                                     "preview_progress": snapshot["preview_progress"]})
                next_sample = time.monotonic() + 5.0
                with (evidence_dir / "observations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(observations[-1], sort_keys=True) + "\n")
                print(f"CANDIDATE LIVE {int(deadline-time.monotonic())}s remaining; six cameras advance; one RTSP owner.",
                      flush=True)
            time.sleep(1.0)

        deployment["live_observation_status"] = "PASS"
        deployment["live_observation_seconds"] = observe_seconds
        deployment["live_observation_samples"] = len(observations)
        deployment["latest_metrics"] = live_metrics(run_dir)
        deployment["candidate_run_dir"] = str(run_dir)
        deployment["protected_hashes_after"] = protected_hashes()
        deployment["protected_hashes_unchanged"] = deployment["protected_hashes_after"] == protected_before
        deployment["amc_listener_unchanged"] = port_listener_signature(8000) == amc_before
        deployment["status"] = "CANDIDATE_DEPLOYED_LIVE_HEALTH_PASS"
        write_json(evidence_dir / "deployment-result.json", deployment)
        return deployment
    except Exception as exc:
        deployment["candidate_failure"] = f"{type(exc).__name__}: {exc}"
        deployment["status"] = "CANDIDATE_FAIL_ROLLBACK_REQUIRED"
        try:
            if candidate_pid is not None and proc_alive(candidate_pid):
                stop_stack(candidate_pid, run_dir)
            elif candidate_pid is not None and run_dir.exists() and (run_dir / "shutdown.json").is_file():
                shutdown = json.loads((run_dir / "shutdown.json").read_text())
                if shutdown.get("status") != "PASS" or shutdown.get("forced_kill") is not False:
                    raise RuntimeError("failed candidate did not leave clean shutdown evidence")
                wait_ports_released()
            else:
                wait_ports_released()
            if port_listener_signature(8000) != amc_before:
                raise RuntimeError("AutoMagicCalib listener changed before rollback")
            rollback_pid = spawn_stack(R6_CANDIDATE, rollback_dir, rollback_log,
                                       base_environment, recovery=False,
                                       amc_before=amc_before)
            deployment["rollback"] = {
                "status": "PASS", "supervisor_pid": rollback_pid,
                "run_dir": str(rollback_dir), "binary_sha256": R6_SHA256,
                "health": runtime_pass(rollback_dir, rollback_pid, R6_SHA256,
                                        R6_CANDIDATE, amc_before, False),
            }
            deployment["status"] = "ROLLED_BACK_TO_R6"
        except Exception as rollback_exc:
            deployment["rollback"] = {"status": "FAIL",
                                      "reason": f"{type(rollback_exc).__name__}: {rollback_exc}"}
        write_json(evidence_dir / "deployment-result.json", deployment)
        raise


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--observe-seconds", type=int, default=180)
    args = parser.parse_args()
    try:
        result = run(args.evidence_dir, preflight_only=args.preflight_only,
                     observe_seconds=args.observe_seconds)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"ENTRANCE_REPAIR_DEPLOY_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
