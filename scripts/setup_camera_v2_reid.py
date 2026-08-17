from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / ".runtime" / "camera_v2" / "models" / "reid"
DEFAULT_MODEL = DEFAULT_DIR / "resnet50_market1501_aicity156.onnx"
MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/"
    "reidentificationnet/deployable_v1.2/files/resnet50_market1501_aicity156.onnx"
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "camera-v2-reid-setup/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if tmp_path.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"downloaded file is unexpectedly small: {tmp_path.stat().st_size} bytes")
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install NVIDIA TAO ReIdentificationNet for Camera V2")
    parser.add_argument("--force", action="store_true", help="download again even if the model already exists")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="destination ONNX path")
    args = parser.parse_args()

    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.force:
        print(f"ReID model already exists: {destination}")
        print(f"size={destination.stat().st_size / (1024 * 1024):.1f} MiB")
        return 0

    print("Downloading NVIDIA TAO ReIdentificationNet v1.2...")
    print(f"source={MODEL_URL}")
    print(f"destination={destination}")
    try:
        _download(MODEL_URL, destination)
    except Exception as exc:
        print(f"ReID model download failed: {exc}", file=sys.stderr)
        return 2

    print(f"OK: {destination}")
    print(f"size={destination.stat().st_size / (1024 * 1024):.1f} MiB")
    print("Camera V2 will build and cache the FP16 TensorRT engine on first start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
