from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def test_deepstream_source_epoch_metadata_round_trip():
    docker = shutil.which("docker")
    assert docker, "Docker with the pinned DeepStream 9.1 image is required for the metadata ABI test"
    command = [
        docker, "run", "--rm", "--pull=never", "--entrypoint", "/bin/bash",
        "-v", f"{ROOT}:/src:ro", IMAGE, "-lc",
        "g++ -std=c++17 -O2 -Wall -Wextra -Werror "
        "-I/src "
        "-I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-1.0) "
        "/src/tests/native/test_source_epoch_meta.cpp -o /tmp/test-source-epoch-meta "
        "$(pkg-config --libs gstreamer-1.0) "
        "-L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib && "
        "/tmp/test-source-epoch-meta",
    ]
    subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=False)
