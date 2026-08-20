from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fail(message: str) -> None:
    print(f"VISION_V3_RFDETR_PREFLIGHT=FAIL {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def main() -> int:
    cfg_path = ROOT / "config" / "vision_v3_detector.yaml"
    cfg = dict((yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("detector") or {})

    if str(cfg.get("model")) != "rfdetr-small":
        fail("detector.model must be rfdetr-small")
    if (int(cfg.get("capture_width", 0)), int(cfg.get("capture_height", 0))) != (672, 384):
        fail("proven RF-DETR-S profile must use 672x384")
    if abs(float(cfg.get("threshold", -1)) - 0.18) > 1e-9:
        fail("proven RF-DETR-S profile must use threshold=0.18")
    if int(cfg.get("micro_batch", 0)) != 1:
        fail("proven GTX 1050 Ti profile must use micro_batch=1")
    if float(cfg.get("max_result_age_ms", 9999)) > 300:
        fail("detector results are allowed to become too stale")

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
        fail(f"RF-DETR import failed: {type(exc).__name__}: {exc}")

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
        # Semantic person filtering is the critical rule from the old working
        # RF-DETR backend.  A 'chair' with a numeric ID that could otherwise be
        # mistaken for person must never survive when class_name is available.
        from services.ml_service.vision_v3.rfdetr_worker_v2 import _person_rows

        fake = SimpleNamespace(
            xyxy=np.asarray(
                [
                    [10.0, 10.0, 100.0, 200.0],
                    [200.0, 20.0, 300.0, 180.0],
                ],
                dtype=np.float32,
            ),
            confidence=np.asarray([0.91, 0.99], dtype=np.float32),
            class_id=np.asarray([1, 1], dtype=np.int64),
            data={"class_name": np.asarray(["person", "chair"])},
        )
        rows = _person_rows(fake, 40)
        if len(rows) != 1 or abs(rows[0][1] - 0.91) > 0.02:
            fail("semantic RF-DETR person filtering regression")

        # The presentation smoother must show a valid strong detection immediately
        # and retain the old asymmetric full-body guard.
        from services.ml_service.vision_v3.core_v1_visual_adapter import CoreV1VisualAdapter

        manager = CoreV1VisualAdapter(1280, 720, dict(cfg.get("box") or {}))
        raw = (500.0, 120.0, 620.0, 610.0)
        t0 = time.monotonic()
        manager.update("CAM-TEST", t0, [(raw, 0.90)])
        visible = manager.render("CAM-TEST", t0)
        if len(visible) != 1:
            fail("proven motion smoother did not show a strong detection immediately")
        x1, y1, x2, y2, _ = visible[0]
        if not (x1 < raw[0] and y1 < raw[1] and x2 > raw[2] and y2 > raw[3]):
            fail("proven full-body guard no longer covers head/feet/sides")
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"proven RF-DETR policy test failed: {type(exc).__name__}: {exc}")

    version = getattr(rfdetr, "__version__", "unknown")
    print(
        "VISION_V3_RFDETR_PREFLIGHT=PASS "
        f"rfdetr={version} gpu={torch.cuda.get_device_name(0)} "
        "profile=agent-rfdetr-s-core-final shape=672x384 threshold=0.18 "
        f"micro_batch=1 semantic_person=1 proven_smoother=1 max_result_age=220ms bridge={bridge}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
