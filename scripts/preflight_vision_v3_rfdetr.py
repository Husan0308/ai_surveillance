from __future__ import annotations

import sys
from pathlib import Path

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
        from services.ml_service.vision_v3.stable_boxes import StableFullBodyManager
        box_cfg = dict(cfg.get("box") or {})
        manager = StableFullBodyManager(1280, 720, box_cfg)
        raw = (500.0, 120.0, 620.0, 610.0)
        manager.update("CAM-TEST", 1.0, [(raw, 0.90)])
        rows = manager.render("CAM-TEST", 1.0)
        if len(rows) != 1:
            fail("stable Kalman display test produced no strong box")
        x1, y1, x2, y2, _ = rows[0]
        if not (x1 < raw[0] and y1 < raw[1] and x2 > raw[2] and y2 > raw[3]):
            fail("stable display envelope did not cover raw head/feet/sides")

        weak = StableFullBodyManager(1280, 720, box_cfg)
        weak.update("CAM-TEST", 1.0, [((100.0, 100.0, 150.0, 200.0), 0.08)])
        if weak.render("CAM-TEST", 1.0):
            fail("one weak RF-DETR observation incorrectly created a visible track")
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"stable box test failed: {type(exc).__name__}: {exc}")

    version = getattr(rfdetr, "__version__", "unknown")
    print(
        "VISION_V3_RFDETR_PREFLIGHT=PASS "
        f"rfdetr={version} gpu={torch.cuda.get_device_name(0)} threshold={threshold:.2f} "
        f"micro_batch={int(cfg.get('micro_batch', 1))} stable_kalman=1 bridge={bridge}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
