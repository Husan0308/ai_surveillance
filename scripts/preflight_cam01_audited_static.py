#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    ".gitignore",
    "README.md",
    "docs/ARCHITECTURE_V2.md",
    ".github/workflows/camera-v2-audited-static.yml",
    "config/cameras.yaml",
    "requirements-trt86.txt",
    "scripts/run_cam01_trt86_audited.sh",
    "scripts/run_cam01_pose_sticky.sh",
    "scripts/restore_cam01_trt86_engine.sh",
    "scripts/yolo26_trt86_shm_worker.py",
    "scripts/yolo26_trt86_shm_worker_v2.py",
    "scripts/yolo26_trt86_shm_worker_v3.py",
    "services/ml_service/app/config.py",
    "services/camera_v2/__init__.py",
    "services/camera_v2/__main__.py",
    "services/camera_v2/main.py",
    "services/camera_v2/dynamic_wall.py",
    "services/camera_v2/secure.py",
    "services/camera_v2/detection.py",
    "services/camera_v2/person_tracking.py",
    "services/camera_v2/person_tracking_final.py",
    "services/camera_v2/person_tracking_trt86_fresh.py",
    "services/camera_v2/person_tracking_trt86_audited.py",
    "services/camera_v2/person_tracking_pose_sticky.py",
    "services/camera_v2/yolo_pose_backend.py",
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
    ".github/workflows/sentinel-ui-static.yml",
    "requirements-camera-v2-rfdetr.txt",
    "services/camera_v2/rfdetr_backend.py",
    "services/camera_v2/person_tracking_trt86.py",
    "services/camera_v2/yolo_trt86_bridge.py",
    "scripts/run_cam01_gpu_nvdcf.sh",
    "scripts/run_cam01_trt86_fixed.sh",
    "scripts/run_cam01_trt86_fresh.sh",
    "scripts/yolo26_trt86_worker.py",
    "services/camera_v2/camera_wall_runtime.py",
    "services/camera_v2/pascal_safe_pipeline.py",
    "services/camera_v2/data.py",
    "services/camera_v2/qt_runtime.py",
    "services/camera_v2/ui_bridge.py",
    "services/camera_v2/native_ui_bridge.c",
    "services/camera_v2/requirements-ui.txt",
    "services/camera_v2/yolo_onnx_cpu_backend.py",
    "services/camera_v2/pose_ankle.py",
    "services/camera_v2/temporal_tracker.py",
    "services/camera_v2/sparse_tracker_contract.py",
    "services/camera_v2/native_sparse_tracker_contract.c",
    "config/tracker/config_tracker_NvDCF_pascal.yml",
)


def fail(message: str) -> None:
    raise SystemExit(f"CAMERA_V2_AUDITED_STATIC=FAIL {message}")


def rels(paths):
    return [str(path.relative_to(ROOT)) for path in sorted(paths)]


def main() -> None:
    missing = [rel for rel in REQUIRED if not (ROOT / rel).is_file()]
    if missing:
        fail("missing=" + ",".join(missing))

    leftovers = [rel for rel in FORBIDDEN_EXACT if (ROOT / rel).exists()]
    leftovers += rels((ROOT / "services/camera_v2").glob("stage*.py"))
    leftovers += rels((ROOT / "scripts").glob("run_camera_stage*.sh"))
    leftovers += rels((ROOT / "scripts").glob("rfdetr*.py"))
    leftovers += rels((ROOT / "services/camera_v2").glob("sentinel_*.py"))
    leftovers += rels((ROOT / "services/camera_v2").glob("monitor_ui*.py"))
    leftovers += rels((ROOT / "services/camera_v2").glob("*heatmap*.py"))
    leftovers += rels((ROOT / "scripts").glob("preflight_*heatmap*.py"))
    leftovers += rels((ROOT / "scripts").glob("preflight_sentinel*.py"))
    leftovers += rels((ROOT / "scripts").glob("preflight_monitor_ui*.py"))
    leftovers = sorted(set(leftovers))
    if leftovers:
        fail("obsolete_files=" + ",".join(leftovers))

    for rel in REQUIRED:
        path = ROOT / rel
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                fail(f"syntax={rel}:{exc.lineno}:{exc.msg}")

    trt_launcher = (ROOT / "scripts/run_cam01_trt86_audited.sh").read_text(encoding="utf-8")
    for needle in (
        "person_tracking_trt86_audited",
        "yolo26_trt86_shm_worker_v3.py",
        "CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01",
        "restore_cam01_trt86_engine.sh",
    ):
        if needle not in trt_launcher:
            fail(f"trt_launcher_contract_missing={needle}")

    pose_launcher = (ROOT / "scripts/run_cam01_pose_sticky.sh").read_text(encoding="utf-8")
    for needle in (
        "person_tracking_pose_sticky",
        "CAMERA_V2_POSE_IMGSZ",
        "CAMERA_V2_EMPTY_CONFIRM_MISSES",
        "CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01",
        "global-id=off",
    ):
        if needle not in pose_launcher:
            fail(f"pose_launcher_contract_missing={needle}")

    pose_backend = (ROOT / "services/camera_v2/yolo_pose_backend.py").read_text(encoding="utf-8")
    for needle in (
        "result.keypoints",
        "usable < 4",
        "same_pose",
        "classes=[0]",
        "yolo26s-pose.pt",
    ):
        if needle not in pose_backend:
            fail(f"pose_backend_contract_missing={needle}")

    sticky = (ROOT / "services/camera_v2/person_tracking_pose_sticky.py").read_text(encoding="utf-8")
    for needle in (
        "CameraPersonTrackingFinal",
        "CAMERA_POSE_EMPTY_HOLD",
        "empty_confirm_misses",
        "jit-no-prefetch=1",
        "self._request_group(group)",
    ):
        if needle not in sticky:
            fail(f"pose_sticky_contract_missing={needle}")
    if "prefetched_group" in sticky:
        fail("pose_sticky_must_not_prefetch")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for needle in ("artifacts/", "*.engine", "*.plan"):
        if needle not in gitignore:
            fail(f"gitignore_missing_local_artifact_rule={needle}")

    audited = (ROOT / "services/camera_v2/person_tracking_trt86_audited.py").read_text(encoding="utf-8")
    for needle in (
        "CameraPersonTrackingTRT86Fresh",
        "_audit_pipeline_graph",
        "person_nvdcf_tracker",
        "CAMERA_PIPELINE_AUDIT status=OK",
    ):
        if needle not in audited:
            fail(f"audited_contract_missing={needle}")

    tracker = (ROOT / "services/camera_v2/person_tracking.py").read_text(encoding="utf-8")
    if "nvtracker" not in tracker or "self.mux.link(tracker)" not in tracker:
        fail("nvdcf_link_contract_missing")

    print(f"CAMERA_V2_AUDITED_STATIC=PASS required={len(REQUIRED)} obsolete=0 pose_sticky=1")


if __name__ == "__main__":
    main()
