from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.qwen_reid import QwenReIdVerifier
from services.camera_v2.reid_embedder import AutoReIdEmbedder
from services.ml_service.app.config import load_settings


def main() -> int:
    cfg_path = ROOT / "config" / "reid.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = dict(raw.get("reid") or raw)
    settings = load_settings()
    rooms = {camera.camera_id: camera.room for camera in settings.cameras}
    print("REID_SETUP rooms=" + ",".join(f"{k}:{v}" for k, v in rooms.items()))

    # Force one real CPU embedding. This downloads/checks the selected model now,
    # not when a person first enters the camera.
    embedder = AutoReIdEmbedder(cfg, ROOT)
    y = np.arange(256, dtype=np.uint8)[:, None]
    x = np.arange(128, dtype=np.uint8)[None, :]
    crop = np.zeros((256, 128, 3), dtype=np.uint8)
    crop[..., 0] = (x + y) % 255
    crop[..., 1] = (2 * x + y) % 255
    crop[..., 2] = (x + 2 * y) % 255
    output = embedder.embed_batch([crop])
    if output.shape[0] != 1 or output.shape[1] < 128:
        raise RuntimeError(f"unexpected ReID output shape: {output.shape}")
    norm = float(np.linalg.norm(output[0]))
    if not 0.98 <= norm <= 1.02:
        raise RuntimeError(f"ReID output is not L2 normalized: norm={norm:.4f}")
    print(
        f"REID_SETUP embedder=PASS shape={output.shape} norm={norm:.4f} "
        f"metrics={embedder.metrics()}"
    )

    qwen = QwenReIdVerifier(cfg)
    if qwen.enabled:
        print(
            f"REID_SETUP qwen=configured model={qwen.model} "
            f"timeout={qwen.timeout:.1f}s"
        )
    else:
        print(
            "REID_SETUP qwen=NOT_CONFIGURED "
            "set QWEN_REID_URL for VLM verification"
        )
    print("CAMERA_V2_REID_SETUP=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
