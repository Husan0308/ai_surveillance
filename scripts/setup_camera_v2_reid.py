from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.qwen_reid import QwenReIdVerifier
from services.camera_v2.reid_embedder import AutoReIdEmbedder
from services.ml_service.app.config import load_settings


def _print_embedder_failure(exc: BaseException) -> None:
    text = f"{type(exc).__name__}: {exc}"
    print(f"REID_SETUP embedder=FAIL error={text}", file=sys.stderr)
    lower = text.lower()
    if "name resolution" in lower or "gaierror" in lower or "temporary failure" in lower:
        print(
            "REID_SETUP network=DNS_FAIL "
            "test='getent hosts storage.openvinotoolkit.org && getent hosts huggingface.co'",
            file=sys.stderr,
        )
        print(
            "REID_SETUP action=FIX_DNS_THEN_RERUN "
            "command='python scripts/setup_camera_v2_reid.py'",
            file=sys.stderr,
        )
    else:
        print(
            "REID_SETUP action=CHECK_MODEL_OR_DEPENDENCY "
            "hint='ReID model must warm successfully before production run'",
            file=sys.stderr,
        )


def _qwen_tcp_check(qwen: QwenReIdVerifier) -> tuple[bool, str]:
    if not qwen.enabled or not qwen.url:
        return False, "not-configured"
    try:
        parsed = urlparse(qwen.url)
        host = parsed.hostname
        if not host:
            return False, "invalid-url"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=min(1.5, qwen.timeout)):
            pass
        return True, f"{host}:{port}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _setup_pose_model() -> bool:
    """Resolve the small pose model before the live process starts.

    Ultralytics resolves official model names such as yolo26n-pose.pt on first use.
    Doing that here means a missing network/model is reported before six RTSP
    streams are opened, not later as a silent heatmap failure.
    """
    model_spec = os.environ.get("CAMERA_V2_POSE_MODEL", "yolo26n-pose.pt").strip()
    try:
        from ultralytics import YOLO

        model = YOLO(model_spec)
        task = str(getattr(model, "task", "") or "")
        if task and task != "pose":
            raise RuntimeError(f"model task is {task!r}, expected 'pose'")
        print(
            "POSE_SETUP model=PASS "
            f"spec={model_spec} task={task or 'pose'} "
            "anchor=left/right-ankle keypoints=15,16 device=cpu-low-rate"
        )
        return True
    except BaseException as exc:
        print(
            f"POSE_SETUP model=FAIL spec={model_spec} "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "POSE_SETUP action=FIX_MODEL_OR_NETWORK "
            "hint='yolo26n-pose.pt must resolve before ankle heatmap starts'",
            file=sys.stderr,
        )
        return False


def main() -> int:
    cfg_path = ROOT / "config" / "reid.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = dict(raw.get("reid") or raw)
    settings = load_settings()
    rooms = {camera.camera_id: camera.room for camera in settings.cameras}
    print("REID_SETUP rooms=" + ",".join(f"{k}:{v}" for k, v in rooms.items()))

    if not _setup_pose_model():
        return 5

    # Force one real CPU embedding. This downloads/checks the selected model now,
    # not when a person first enters the camera. Production must not silently start
    # with an unavailable ReID model and discover the problem on the first track.
    embedder = AutoReIdEmbedder(cfg, ROOT)
    y = np.arange(256, dtype=np.uint8)[:, None]
    x = np.arange(128, dtype=np.uint8)[None, :]
    crop = np.zeros((256, 128, 3), dtype=np.uint8)
    crop[..., 0] = (x + y) % 255
    crop[..., 1] = (2 * x + y) % 255
    crop[..., 2] = (x + 2 * y) % 255
    try:
        output = embedder.embed_batch([crop])
    except BaseException as exc:
        _print_embedder_failure(exc)
        return 2

    if output.shape[0] != 1 or output.shape[1] < 128:
        print(f"REID_SETUP embedder=FAIL unexpected_shape={output.shape}", file=sys.stderr)
        return 3
    norm = float(np.linalg.norm(output[0]))
    if not 0.98 <= norm <= 1.02:
        print(f"REID_SETUP embedder=FAIL not_l2_normalized norm={norm:.4f}", file=sys.stderr)
        return 4
    print(
        f"REID_SETUP embedder=PASS shape={output.shape} norm={norm:.4f} "
        f"metrics={embedder.metrics()}"
    )

    qwen = QwenReIdVerifier(cfg)
    if qwen.enabled:
        reachable, detail = _qwen_tcp_check(qwen)
        print(
            f"REID_SETUP qwen=configured model={qwen.model} endpoint={qwen.url} "
            f"tcp={'PASS' if reachable else 'WARN'} detail={detail} "
            f"timeout={qwen.timeout:.1f}s"
        )
    else:
        print(
            "REID_SETUP qwen=NOT_CONFIGURED "
            "set QWEN_REID_URL to a real OpenAI-compatible Qwen-VL endpoint; "
            "do not copy angle-bracket placeholders into the shell"
        )
    print("CAMERA_V2_REID_SETUP=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
