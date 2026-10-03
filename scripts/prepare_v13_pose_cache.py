"""Build the unchanged V13 pose model for F4's pinned TRT runtime, in staging."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.build_room_candidate import ROOT
from scripts.dev_room_mv3dt.verify_validated_assets import sha256
from scripts.run_room_candidate import save

NAME = "bodypose3dnet_accuracy.onnx"
MODEL = Path("/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/BodyPose3DNet") / NAME


def prepare(output: Path) -> dict:
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "models/BodyPose3DNet"
    cache.mkdir(parents=True)
    shutil.copyfile(MODEL, cache / NAME)
    original = {str(p): sha256(p) for p in (MODEL, MODEL.with_name(NAME + "_b1_gpu0_fp16.engine"))}
    image = json.loads((ROOT / "config/deepstream-platform.json").read_text())["image"]
    engine_name = NAME + "_b1_gpu0_fp16.engine"
    # ONNX declares the body-pose input dimensions. trtexec reports all resolved
    # binding/profile dimensions in the saved log; no model/threshold is edited.
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "device=0",
        "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{cache}:/pose",
        "--entrypoint", "/usr/src/tensorrt/bin/trtexec", image,
        f"--onnx=/pose/{NAME}", f"--saveEngine=/pose/{engine_name}", "--fp16",
        "--skipInference", "--profilingVerbosity=detailed"]
    with (output / "build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if original != {str(p): sha256(p) for p in map(Path, original)} or sha256(cache / NAME) != original[str(MODEL)]:
        raise RuntimeError("production model changed during isolated engine build")
    result = {"production_changed": False, "original": original, "model_sha256": sha256(cache / NAME),
        "engine_sha256": sha256(cache / engine_name), "engine": str(cache / engine_name),
        "image": image, "command": command, "precision": "FP16", "batch_size": 1}
    save(output / "build.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    result = prepare(parser.parse_args().output)
    print(json.dumps(result), flush=True)
