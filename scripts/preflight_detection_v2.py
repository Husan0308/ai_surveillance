from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=12, check=False)
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    print("CAMERA_V2 DETECTION PREFLIGHT")
    print(f"repo_root={ROOT}")
    print(f"python={sys.version.split()[0]}")

    for module in ("numpy", "torch", "ultralytics"):
        ok = importlib.util.find_spec(module) is not None
        print(f"python_module {module}={'OK' if ok else 'MISSING'}")
        if not ok:
            failures.append(f"missing Python module: {module}")

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
            if not torch.cuda.is_available():
                failures.append("torch.cuda.is_available() is False")
            else:
                print(f"cuda_device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
        except Exception as exc:
            failures.append(f"torch CUDA check: {type(exc).__name__}: {exc}")

    model = os.environ.get("CAMERA_V2_YOLO_MODEL", "yolo26m.pt")
    model_path = Path(model)
    if not model_path.is_absolute():
        rooted = ROOT / model
        if rooted.exists():
            model_path = rooted
    print(f"model={model_path if model_path.exists() else model}")
    if not model_path.exists():
        warnings.append(
            "YOLO model file is not in the repo; Ultralytics will try to resolve/download CAMERA_V2_YOLO_MODEL"
        )

    for tool in ("gcc", "pkg-config", "gst-inspect-1.0"):
        found = shutil.which(tool)
        print(f"tool {tool}={found or 'MISSING'}")
        if not found:
            failures.append(f"missing tool: {tool}")

    if shutil.which("gst-inspect-1.0"):
        required = [
            "nvurisrcbin", "nvstreammux", "nvmultistreamtiler", "nveglglessink",
            "nvvideoconvert", "nvdsosd", "tee", "queue", "appsink", "capsfilter",
        ]
        for plugin in required:
            code, _ = _run(["gst-inspect-1.0", plugin])
            print(f"plugin {plugin}={'OK' if code == 0 else 'MISSING'}")
            if code != 0:
                failures.append(f"missing GStreamer/DeepStream plugin: {plugin}")

    ds_roots = [Path("/opt/nvidia/deepstream/deepstream")]
    ds_roots += sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True)
    ds = next((p for p in ds_roots if (p / "sources/includes/gstnvdsmeta.h").exists()), None)
    print(f"deepstream_root={ds or 'NOT_FOUND'}")
    if ds is None:
        failures.append("DeepStream development headers not found")
    else:
        if not (ds / "lib/libnvds_meta.so").exists():
            failures.append("libnvds_meta.so not found")

    try:
        from services.camera_v2.native_bridge import ensure_bridge
        bridge = ensure_bridge()
        print(f"native_meta_bridge=OK path={bridge}")
    except Exception as exc:
        failures.append(f"native metadata bridge: {type(exc).__name__}: {exc}")

    for warning in warnings:
        print("WARNING: " + warning)
    if failures:
        for failure in failures:
            print("FAIL: " + failure)
        print("DETECTION_PREFLIGHT=FAIL")
        return 1

    print("DETECTION_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
