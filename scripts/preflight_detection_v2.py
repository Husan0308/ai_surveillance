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

    try:
        from services.ml_service.app.config import load_settings
        settings = load_settings()
        missing = [c.camera_id for c in settings.cameras if not c.username or not c.password]
        if missing:
            failures.append(
                "RTSP credentials missing for: " + ", ".join(missing)
                + ". Run: python scripts/setup_rtsp_auth.py"
            )
        else:
            print("rtsp_auth=CONFIGURED (secrets hidden)")
    except Exception as exc:
        failures.append(f"camera config: {type(exc).__name__}: {exc}")

    for module in ("numpy", "torch", "torchvision", "ultralytics"):
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
                device = torch.cuda.get_device_name(0)
                capability = torch.cuda.get_device_capability(0)
                target_arch = f"sm_{capability[0]}{capability[1]}"
                arch_list = list(torch.cuda.get_arch_list())
                print(f"cuda_device={device} capability={capability} target_arch={target_arch}")
                print(f"torch_cuda_arch_list={arch_list}")
                if target_arch not in arch_list:
                    failures.append(
                        f"installed PyTorch wheel does not contain {target_arch}; "
                        "GTX 1050 Ti/Pascal needs an official CUDA 12.6 wheel that includes sm_61"
                    )
                else:
                    try:
                        x = torch.ones((32, 32), device="cuda")
                        y = x @ x
                        torch.cuda.synchronize()
                        print(f"torch_cuda_smoke=OK value={float(y[0, 0].item()):.1f}")
                    except Exception as exc:
                        failures.append(f"PyTorch CUDA kernel smoke test failed: {type(exc).__name__}: {exc}")

                    if importlib.util.find_spec("torchvision") is not None:
                        try:
                            import torchvision
                            from torchvision.ops import nms

                            print(f"torchvision={torchvision.__version__}")
                            boxes = torch.tensor(
                                [[0.0, 0.0, 20.0, 20.0], [2.0, 2.0, 19.0, 19.0]],
                                device="cuda",
                            )
                            scores = torch.tensor([0.9, 0.8], device="cuda")
                            keep = nms(boxes, scores, 0.5)
                            torch.cuda.synchronize()
                            print(f"torchvision_cuda_nms=OK keep={keep.detach().cpu().tolist()}")
                        except Exception as exc:
                            failures.append(
                                "torchvision CUDA NMS failed; torch/torchvision wheel pair is not usable "
                                f"for this Pascal GPU: {type(exc).__name__}: {exc}"
                            )
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
        from services.camera_v2.native_bridge import NativeMetaBridge
        bridge = NativeMetaBridge()
        print(f"native_meta_bridge=OK path={bridge.path}")
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
