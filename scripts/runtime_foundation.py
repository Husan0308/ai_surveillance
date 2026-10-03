#!/usr/bin/env python3
"""F1 host runtime setup/preflight. Never opens cameras or starts services.

Ubuntu supplies GI/GStreamer. A declared read-only GPU package bridge can reuse
existing PyTorch/NVIDIA Python libraries, without installing or modifying them.
All other dependencies are project-local and version-pinned. User-site and
ambient PYTHONPATH packages are forbidden.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import site
import subprocess
import sys

GPU_MODULES = ("torch", "torchgen", "functorch", "torchvision", "torchvision.libs", "triton", "nvidia", "cuda")
GPU_DISTRIBUTIONS = ("torch", "torchvision", "triton", "cuda-", "nvidia-")
GPU_VERSIONS = {"torch": "2.13.0+cu126", "torchvision": "0.28.0+cu126", "triton": "3.7.1"}
ROLE_MODULES = {
    "full": ("fastapi", "httpx", "uvicorn", "numpy", "yaml", "dotenv", "gi", "PySide6",
             "cv2", "kafka", "google.protobuf", "torch", "torchvision", "torchreid"),
    "identity": ("numpy", "cv2", "torch", "torchvision", "torchreid"),
    "kafka": ("kafka", "google.protobuf"),
}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def no_trash(path: Path) -> None:
    if any(p.lower() == "trash" for p in (*path.parts, *path.resolve().parts)):
        raise ValueError("Trash paths are not valid runtime dependencies")


def pins(root: Path) -> dict[str, str]:
    result = dict(line.split("==", 1) for line in
                  (root / "requirements/runtime-host.lock.txt").read_text().splitlines()
                  if line and not line.startswith("#"))
    for name in result:
        if name.startswith(("cuda", "nvidia", "tensorrt", "pygobject")) or name in (
            "torch", "torchvision", "triton", "pycuda", "pyds", "pyservicemaker"):
            raise ValueError(f"protected platform package cannot be pip-installed: {name}")
    return result


def platform_check(prefix: Path) -> dict:
    record = prefix / "platform-libs.json"
    if not record.exists():
        return {"links": {}, "metadata": {}}
    data = json.loads(record.read_text())
    for link, target in data["links"].items():
        link_path, target_path = Path(link), Path(target)
        no_trash(target_path)
        if not link_path.is_symlink() or link_path.resolve() != target_path.resolve():
            raise ValueError(f"declared platform link drift: {link}")
    for filename, digest in data["metadata"].items():
        if sha(Path(filename)) != digest:
            raise ValueError(f"declared platform metadata drift: {filename}")
    return data


def check(root: Path, role: str = "full") -> dict:
    if sys.version_info[:2] != (3, 12) or sys.prefix == sys.base_prefix:
        raise ValueError("host runtime requires an explicit Ubuntu Python 3.12 venv")
    no_trash(Path(sys.executable))
    if site.ENABLE_USER_SITE:
        raise ValueError("user-site must be disabled: use PYTHONNOUSERSITE=1 or python -I")
    prefix = Path(sys.prefix)
    platform = platform_check(prefix)
    errors, modules = [], {}
    # Imports must come from this venv, the explicitly declared GPU links, or
    # Ubuntu's ABI-matched GI packages, not another Python environment.
    local = prefix / "lib/python3.12/site-packages"
    for name in ROLE_MODULES[role]:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                module = importlib.import_module(name)
            origin = Path(module.__file__).resolve()
            no_trash(origin)
            declared = any(origin.is_relative_to(Path(p).resolve()) for p in platform["links"].values())
            system_gi = name == "gi" and origin.is_relative_to(Path("/usr/lib/python3/dist-packages/gi"))
            if not origin.is_relative_to(local.resolve()) and not declared and not system_gi:
                raise ValueError(f"undeclared dependency origin: {name} {origin}")
            modules[name] = {"path": str(origin), "version": str(getattr(module, "__version__", "unknown"))}
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for name, expected in {**pins(root), **GPU_VERSIONS}.items():
        try:
            distribution = metadata.distribution(name)
            actual = distribution.version
            if actual != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")
            origin = Path(distribution._path).resolve()
            declared = any(origin == Path(p).resolve() for p in platform["links"].values())
            if not origin.is_relative_to(local.resolve()) and not declared:
                errors.append(f"{name}: undeclared distribution origin {origin}")
        except metadata.PackageNotFoundError:
            errors.append(f"{name}: pinned distribution missing")
    gst_version = None
    if role == "full" and "gi" in modules:
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)  # Plugin discovery only: no pipelines/RTSP/decoders.
            gst_version = Gst.version_string()
        except Exception as exc:
            errors.append(f"gi(Gst): {exc}")
    if role in ("full", "identity") and "torchreid" in modules:
        try:
            from torchreid import models
            from torchreid.reid.utils import load_pretrained_weights
            if not callable(models.build_model) or not callable(load_pretrained_weights):
                raise ValueError("OSNet API is unavailable")
            import torch
            if not torch.cuda.is_available():
                raise ValueError("existing OSNet CUDA runtime is unavailable")
        except Exception as exc:
            errors.append(f"OSNet: {exc}")
    if role in ("full", "kafka"):
        try:
            # Import/round-trip only. Never instantiate KafkaConsumer here.
            sys.path.insert(0, str(root / "services/mv3dt_room"))
            from schema_pb2 import Frame
            Frame.FromString(Frame().SerializeToString())
        except Exception as exc:
            errors.append(f"Kafka protobuf schema: {exc}")
    if errors:
        raise ValueError("runtime preflight failed:\n  " + "\n  ".join(errors))
    return {"python": sys.executable, "version": sys.version.split()[0], "prefix": sys.prefix,
            "system_site_packages": any(p == "/usr/lib/python3/dist-packages" for p in sys.path),
            "user_site_enabled": site.ENABLE_USER_SITE, "role": role, "modules": modules,
            "gstreamer": gst_version, "platform": platform,
            "requirements_sha256": sha(root / "requirements/runtime-host.lock.txt"),
            "camera_opened": False}


def bridge_gpu(prefix: Path, provider: Path) -> None:
    """Declare existing packages by read-only reference; do not pip-install GPU libs."""
    no_trash(provider)
    if not provider.is_absolute() or not provider.is_dir():
        raise ValueError("--gpu-site must name an existing absolute site-packages directory")
    available = {d.metadata["Name"].lower(): d for d in metadata.distributions(path=[str(provider)])}
    for name, expected in GPU_VERSIONS.items():
        if name not in available or available[name].version != expected:
            raise ValueError(f"GPU provider requires {name}=={expected}")
    local = prefix / "lib/python3.12/site-packages"
    targets = [provider / name for name in GPU_MODULES if (provider / name).exists()]
    targets.extend(Path(d._path) for name, d in available.items()
                   if name in GPU_DISTRIBUTIONS or name.startswith(("cuda-", "nvidia-")))
    # No API/UI packages or broad .pth search path are imported from the provider.
    links, records = {}, {}
    for target in targets:
        no_trash(target)
        link = local / target.name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise ValueError(f"refusing to overwrite existing runtime package: {link}")
        else:
            link.symlink_to(target, target_is_directory=target.is_dir())
        links[str(link)] = str(target)
        for filename in ("METADATA", "RECORD", "__init__.py"):
            path = target / filename
            if path.is_file():
                records[str(path)] = sha(path)
    (prefix / "platform-libs.json").write_text(json.dumps({"provider": str(provider),
        "links": links, "metadata": records, "nvidia_packages_installed": False}, indent=2) + "\n")


def setup(root: Path, gpu_site: Path | None) -> Path:
    prefix = root / ".runtime/full-stack-venv"
    python = prefix / "bin/python"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with (prefix.parent / "runtime-setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = [str(python), "-I", "-B", str(Path(__file__).resolve()), "check", "--root", str(root)]
        if python.is_file() and subprocess.run(command, stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL).returncode == 0:
            print("RUNTIME already satisfies pinned requirements; no install", file=sys.stderr)
            return python
        print("RUNTIME explicit setup; NVIDIA/system packages will not be changed", file=sys.stderr)
        if prefix.exists() and not (prefix / "pyvenv.cfg").is_file():
            raise ValueError("refusing to overwrite non-venv runtime directory")
        if (prefix / "pyvenv.cfg").is_file():
            cfg = (prefix / "pyvenv.cfg").read_text()
            if "include-system-site-packages = true" not in cfg or "version = 3.12." not in cfg:
                raise ValueError("existing runtime must be Python 3.12 with system-site-packages; not replaced")
        if not python.exists():
            subprocess.run(["/usr/bin/python3", "-m", "venv", "--system-site-packages", str(prefix)],
                           check=True, stdout=sys.stderr)
        cfg = (prefix / "pyvenv.cfg").read_text()
        if "include-system-site-packages = true" not in cfg or "version = 3.12." not in cfg:
            raise ValueError("existing runtime must be Python 3.12 with system-site-packages; not replaced")
        if gpu_site:
            bridge_gpu(prefix, gpu_site)
        elif not (prefix / "platform-libs.json").is_file():
            raise ValueError("first setup requires --gpu-site pointing to the existing non-Trash GPU Python libraries")
        # Only repair missing/drifted local pins; never reinstall an already
        # satisfied runtime. Complete closure and --no-deps forbid NVIDIA pulls.
        local = prefix / "lib/python3.12/site-packages"
        installed = {d.metadata["Name"].lower().replace("_", "-"): d.version
                     for d in metadata.distributions(path=[str(local)])}
        missing = [f"{name}=={version}" for name, version in pins(root).items()
                   if installed.get(name) != version]
        if missing:
            subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-deps",
                            "--disable-pip-version-check", "--ignore-installed", *missing],
                           check=True, stdout=sys.stderr)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "setup"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--role", choices=ROLE_MODULES, default="full")
    parser.add_argument("--gpu-site", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "setup":
            print(setup(args.root.resolve(), args.gpu_site))
        else:
            print(json.dumps(check(args.root.resolve(), args.role), indent=2))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"RUNTIME FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
