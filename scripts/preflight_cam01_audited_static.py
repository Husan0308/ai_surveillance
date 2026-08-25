#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "config/cameras.yaml",
    "requirements-trt86.txt",
    "scripts/run_cam01_trt86_audited.sh",
    "scripts/yolo26_trt86_shm_worker.py",
    "scripts/yolo26_trt86_shm_worker_v2.py",
    "scripts/yolo26_trt86_shm_worker_v3.py",
    "services/ml_service/app/config.py",
    "services/camera_v2/main.py",
    "services/camera_v2/dynamic_wall.py",
    "services/camera_v2/secure.py",
    "services/camera_v2/detection.py",
    "services/camera_v2/person_tracking.py",
    "services/camera_v2/person_tracking_final.py",
    "services/camera_v2/person_tracking_trt86_fresh.py",
    "services/camera_v2/person_tracking_trt86_audited.py",
    "services/camera_v2/detector_latency.py",
    "services/camera_v2/tracker_profile.py",
    "services/camera_v2/native_bridge.py",
    "services/camera_v2/native_meta_bridge.c",
    "services/camera_v2/native_label_style.c",
    "services/camera_v2/native_display_smoother.c",
    "services/camera_v2/native_heatmap.c",
    "services/camera_v2/yolo_trt86_shm_bridge.py",
    "services/camera_v2/yolo_trt86_fresh_bridge.py",
)

FORBIDDEN_EXACT = (
    "requirements-camera-v2-rfdetr.txt",
    "services/camera_v2/rfdetr_backend.py",
    "services/camera_v2/person_tracking_trt86.py",
    "services/camera_v2/yolo_trt86_bridge.py",
    "scripts/run_cam01_gpu_nvdcf.sh",
    "scripts/run_cam01_trt86_fixed.sh",
    "scripts/run_cam01_trt86_fresh.sh",
    "scripts/yolo26_trt86_worker.py",
)


def fail(message: str) -> None:
    raise SystemExit(f"CAMERA_V2_AUDITED_STATIC=FAIL {message}")


def main() -> None:
    missing = [rel for rel in REQUIRED if not (ROOT / rel).is_file()]
    if missing:
        fail("missing=" + ",".join(missing))

    leftovers = [rel for rel in FORBIDDEN_EXACT if (ROOT / rel).exists()]
    leftovers += [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "services/camera_v2").glob("stage*.py"))
    ]
    leftovers += [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "scripts").glob("run_camera_stage*.sh"))
    ]
    leftovers += [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "scripts").glob("rfdetr*.py"))
    ]
    if leftovers:
        fail("obsolete_files=" + ",".join(leftovers))

    for rel in REQUIRED:
        path = ROOT / rel
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                fail(f"syntax={rel}:{exc.lineno}:{exc.msg}")

    launcher = (ROOT / "scripts/run_cam01_trt86_audited.sh").read_text(encoding="utf-8")
    for needle in (
        "person_tracking_trt86_audited",
        "yolo26_trt86_shm_worker_v3.py",
        "CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01",
        "CAMERA_V2_RTSP_TRANSPORT",
        "CAMERA_V2_RTSP_LATENCY_MS",
    ):
        if needle not in launcher:
            fail(f"launcher_contract_missing={needle}")

    audited = (ROOT / "services/camera_v2/person_tracking_trt86_audited.py").read_text(encoding="utf-8")
    for needle in (
        "CameraPersonTrackingTRT86Fresh",
        "_audit_pipeline_graph",
        "person_nvdcf_tracker",
        "CAMERA_PIPELINE_AUDIT status=OK",
    ):
        if needle not in audited:
            fail(f"audited_contract_missing={needle}")

    fresh = (ROOT / "services/camera_v2/person_tracking_trt86_fresh.py").read_text(encoding="utf-8")
    if "no prefetch" not in fresh.lower() and "no-prefetch" not in fresh.lower():
        fail("fresh_runtime_missing_no_prefetch_contract")

    tracker = (ROOT / "services/camera_v2/person_tracking.py").read_text(encoding="utf-8")
    if "nvtracker" not in tracker or "self.mux.link(tracker)" not in tracker:
        fail("nvdcf_link_contract_missing")

    print(f"CAMERA_V2_AUDITED_STATIC=PASS required={len(REQUIRED)} obsolete=0")


if __name__ == "__main__":
    main()
