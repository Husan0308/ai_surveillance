#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "yolo26m_trt86"
OUT = OUT_DIR / "yolo26m-672x384-b1-e2e.onnx"
MODEL = ROOT / "yolo26m.pt"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL) if MODEL.is_file() else "yolo26m.pt")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=(384, 672),
            batch=1,
            dynamic=False,
            simplify=True,
            end2end=True,
            opset=17,
            half=False,
            device="cpu",
            verbose=False,
        )
    )
    if not exported.is_file():
        raise SystemExit(f"YOLO26M_EXPORT_FAIL missing={exported}")
    if exported.resolve() != OUT.resolve():
        shutil.copy2(exported, OUT)
    if not OUT.is_file() or OUT.stat().st_size <= 0:
        raise SystemExit(f"YOLO26M_EXPORT_FAIL empty={OUT}")
    print(f"YOLO26M_EXPORT_OK path={OUT} bytes={OUT.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
