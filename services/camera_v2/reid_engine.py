from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _find_trtexec() -> Path:
    candidates: list[Path] = []
    found = shutil.which("trtexec")
    if found:
        candidates.append(Path(found))
    candidates.extend(
        [
            Path("/usr/src/tensorrt/bin/trtexec"),
            Path("/opt/tensorrt/bin/trtexec"),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError(
        "TensorRT trtexec was not found. Expected /usr/src/tensorrt/bin/trtexec "
        "or trtexec in PATH."
    )


def _engine_looks_valid(path: Path) -> bool:
    # A ResNet-50 ReID plan should be many megabytes. This deliberately rejects
    # zero-byte/partial files left by an interrupted TensorRT build.
    return path.exists() and path.is_file() and path.stat().st_size >= 4 * 1024 * 1024


def ensure_reid_engine(
    model: Path,
    engine: Path,
    *,
    workspace_mb: int = 256,
) -> Path:
    """Build the ReID TensorRT plan before the six-camera pipeline owns the GPU.

    NvMultiObjectTracker can build an engine itself when modelEngineFile is absent,
    but doing that after NVDEC/EGL/YOLO allocations are live is unsafe on a 4 GB
    display GPU. TensorRT engine building uses substantial temporary device memory.
    This helper runs trtexec first, with bounded workspace and CUDA lazy module
    loading, then atomically publishes the finished plan.

    The NVIDIA TAO ReIdentificationNet ONNX has a dynamic batch dimension. Our
    GTX-1050-Ti profile intentionally builds batch=1: NvDCF can still process all
    targets, while runtime/build memory stays predictable. Global cross-camera ReID
    only needs periodic embeddings, not a large instantaneous ReID batch.
    """

    model = Path(model).expanduser().resolve()
    engine = Path(engine).expanduser().resolve()
    if _engine_looks_valid(engine):
        return engine
    if engine.exists():
        engine.unlink(missing_ok=True)

    trtexec = _find_trtexec()
    engine.parent.mkdir(parents=True, exist_ok=True)
    build_path = engine.with_suffix(engine.suffix + ".building")
    log_path = engine.with_suffix(engine.suffix + ".build.log")
    build_path.unlink(missing_ok=True)

    workspace_mb = max(64, min(512, int(workspace_mb)))
    env = os.environ.copy()
    # NVIDIA recommends lazy CUDA module loading as a way to reduce peak device
    # memory during TensorRT engine construction in DeepStream/TAO deployments.
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")

    base = [
        str(trtexec),
        f"--onnx={model}",
        f"--saveEngine={build_path}",
        "--fp16",
        "--skipInference",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]

    # The current TAO ReIdentificationNet v1.2 input is named `inputs` and has a
    # dynamic batch. Explicit batch=1 shapes avoid TensorRT attempting an oversized
    # optimization profile. The fallback lets trtexec infer defaults if a future
    # compatible model changes its input name.
    attempts = [
        [
            *base,
            "--minShapes=inputs:1x3x256x128",
            "--optShapes=inputs:1x3x256x128",
            "--maxShapes=inputs:1x3x256x128",
        ],
        base,
    ]

    errors: list[str] = []
    print(
        "CAMERA_REID engine missing; prebuilding TensorRT FP16 engine before "
        "camera/YOLO startup...",
        flush=True,
    )
    print(f"CAMERA_REID trtexec={trtexec} workspace={workspace_mb}MB batch=1", flush=True)
    print(f"CAMERA_REID build_log={log_path}", flush=True)

    for attempt_no, command in enumerate(attempts, start=1):
        build_path.unlink(missing_ok=True)
        with log_path.open("w" if attempt_no == 1 else "a", encoding="utf-8") as log:
            if attempt_no > 1:
                log.write("\n\n=== fallback build without explicit input shapes ===\n")
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )

        if result.returncode == 0 and _engine_looks_valid(build_path):
            build_path.replace(engine)
            print(
                f"CAMERA_REID engine ready: {engine} "
                f"({engine.stat().st_size / (1024 * 1024):.1f} MiB)",
                flush=True,
            )
            return engine

        tail = ""
        try:
            rows = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(rows[-24:])
        except Exception:
            pass
        errors.append(f"attempt {attempt_no} rc={result.returncode}\n{tail}")

    build_path.unlink(missing_ok=True)
    raise RuntimeError(
        "Camera V2 failed to prebuild the ReID TensorRT engine. "
        f"Full log: {log_path}\n" + "\n---\n".join(errors)
    )
