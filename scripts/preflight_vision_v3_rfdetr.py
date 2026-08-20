from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fail(message: str) -> None:
    print(f"VISION_V3_RFDETR_PREFLIGHT=FAIL {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def main() -> int:
    cfg_path = ROOT / "config" / "vision_v3_detector.yaml"
    if not cfg_path.exists():
        fail("missing config/vision_v3_detector.yaml")
    cfg = dict((yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("detector") or {})
    if str(cfg.get("model")) != "rfdetr-small":
        fail("detector.model must be rfdetr-small")
    if int(cfg.get("micro_batch", 0)) not in (1, 2):
        fail("micro_batch must be 1 or 2")
    threshold = float(cfg.get("threshold", -1))
    if not 0.01 <= threshold <= 0.60:
        fail(f"unreasonable person threshold {threshold}")

    roi_cfg = dict(cfg.get("roi_second_pass") or {})
    if not bool(roi_cfg.get("enabled", False)):
        fail("core-v1 ROI recovery policy is disabled")
    roi_cameras = dict(roi_cfg.get("cameras") or {})
    if not {"CAM-05", "CAM-06"}.issubset(roi_cameras):
        fail("expected proven CAM-05/CAM-06 ROI recovery configuration")

    try:
        import torch
    except Exception as exc:
        fail(f"PyTorch import failed: {type(exc).__name__}: {exc}")
    if not torch.cuda.is_available():
        fail("PyTorch CUDA is unavailable")

    try:
        import rfdetr
        from rfdetr import RFDETRSmall  # noqa: F401
    except Exception as exc:
        fail(
            "RF-DETR import failed; install with: python -m pip install 'rfdetr>=1.6.2,<2' "
            f"({type(exc).__name__}: {exc})"
        )

    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
    except Exception as exc:
        fail(f"GStreamer bindings unavailable: {type(exc).__name__}: {exc}")

    required = (
        "nvurisrcbin",
        "nvstreammux",
        "nvmultistreamtiler",
        "nveglglessink",
        "nvvideoconvert",
        "nvdsosd",
        "tee",
        "queue",
        "appsink",
        "capsfilter",
    )
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        fail("missing plugins: " + ", ".join(missing))

    try:
        from services.ml_service.vision_v3.native_boxes import ensure_bridge
        bridge = ensure_bridge()
    except Exception as exc:
        fail(f"native metadata bridge failed: {type(exc).__name__}: {exc}")

    try:
        from services.ml_service.vision_v3.core_v1_visual_adapter import CoreV1VisualAdapter
        from services.ml_service.vision_v3.rfdetr_worker_v2 import _dedupe, _filter_hard_masks, rfdetr_worker_v2  # noqa: F401
        from services.ml_service.vision_v3.sparse_visual_tracker import SparseCadenceVisualTracker
        from services.ml_service.vision_v3.visual_tracker import VisualBox

        box_cfg = dict(cfg.get("box") or {})
        manager = CoreV1VisualAdapter(1280, 720, box_cfg)
        raw = (500.0, 120.0, 620.0, 610.0)
        t0 = time.monotonic()

        # A genuinely strong RF-DETR observation must be visible immediately.
        manager.update("CAM-TEST", t0, [(raw, 0.90)])
        rows = manager.render("CAM-TEST", t0)
        if len(rows) != 1:
            fail("strong RF-DETR observation did not create an immediate visible track")
        x1, y1, x2, y2, _ = rows[0]
        if not (x1 < raw[0] and y1 < raw[1] and x2 > raw[2] and y2 > raw[3]):
            fail("display-only full-body guard did not cover raw head/feet/sides")

        # RF-DETR revisits a given camera only every ~1.5-2.0 seconds here. Verify
        # a borderline birth candidate survives that real cadence instead of the
        # old hard-coded 1.25 s YOLO window expiring it before hit #2.
        tracker = SparseCadenceVisualTracker(
            birth_candidate_ttl_ms=int(box_cfg.get("birth_candidate_ttl_ms", 5000)),
            hold_ms=int(box_cfg.get("hold_ms", 2400)),
            memory_ms=int(box_cfg.get("memory_ms", 6000)),
            prediction_ms=int(box_cfg.get("prediction_ms", 1100)),
            byte_high_conf=float(box_cfg.get("byte_high_conf", 0.08)),
            byte_low_conf=float(box_cfg.get("byte_low_conf", 0.06)),
            low_conf_confirm=float(box_cfg.get("low_conf_confirm", 0.06)),
            start_conf=float(box_cfg.get("start_conf", 0.18)),
            new_track_min_conf=float(box_cfg.get("new_track_min_conf", 0.08)),
            strong_confirm_hits=int(box_cfg.get("strong_confirm_hits", 1)),
            weak_confirm_hits=int(box_cfg.get("weak_confirm_hits", 2)),
            low_match_max_age_ms=int(box_cfg.get("low_match_max_age_ms", 2200)),
        )
        first = SimpleNamespace(
            frame_id=1,
            frame_captured_monotonic=t0,
            boxes=(VisualBox(100.0, 100.0, 180.0, 360.0, 0.12),),
        )
        second = SimpleNamespace(
            frame_id=2,
            frame_captured_monotonic=t0 + 1.80,
            boxes=(VisualBox(104.0, 102.0, 184.0, 362.0, 0.12),),
        )
        tracker.update(first, now=t0, source_width=1280, source_height=720)
        if tracker.visible(now=t0, target_time=t0):
            fail("borderline RF-DETR birth became visible before temporal confirmation")
        tracker.update(second, now=t0 + 1.80, source_width=1280, source_height=720)
        if len(tracker.visible(now=t0 + 1.80, target_time=t0 + 1.80)) != 1:
            fail("sparse-cadence RF-DETR birth candidate expired before second hit")

        weak = CoreV1VisualAdapter(1280, 720, box_cfg)
        weak.update("CAM-TEST", t0, [((100.0, 100.0, 150.0, 200.0), 0.07)])
        if weak.render("CAM-TEST", t0):
            fail("one continuation-only RF-DETR observation incorrectly created a visible person")

        fused = _dedupe(
            [
                (100.0, 100.0, 200.0, 400.0, 0.90),
                (104.0, 105.0, 198.0, 395.0, 0.70),
                (350.0, 100.0, 450.0, 400.0, 0.80),
            ],
            iou_threshold=float(cfg.get("duplicate_iou", 0.58)),
            containment_threshold=float(cfg.get("fusion_containment", 0.84)),
            center_threshold=float(cfg.get("fusion_center_distance", 0.40)),
        )
        if len(fused) != 2:
            fail(f"full/ROI confidence-first fusion regression: expected 2 boxes, got {len(fused)}")

        hard_cfg = dict(cfg.get("hard_exclusion") or {})
        _masked, rejected = _filter_hard_masks(
            "CAM-06",
            [(420.0, 10.0, 500.0, 120.0, 0.80)],
            768,
            432,
            hard_cfg,
        )
        if hard_cfg.get("cameras") and rejected != 1:
            fail("CAM-06 hard exclusion policy regression")
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"ported Core-v1 policy test failed: {type(exc).__name__}: {exc}")

    version = getattr(rfdetr, "__version__", "unknown")
    print(
        "VISION_V3_RFDETR_PREFLIGHT=PASS "
        f"rfdetr={version} gpu={torch.cuda.get_device_name(0)} threshold={threshold:.2f} "
        f"micro_batch={int(cfg.get('micro_batch', 1))} core_v1_policy=1 "
        f"adaptive_kalman_byte=1 sparse_birth=1 roi_recovery=1 fusion=1 bridge={bridge}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
